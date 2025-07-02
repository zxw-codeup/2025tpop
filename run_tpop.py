import json
import os

import yaml
import scipy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from rdkit import Chem

from pretrain_clf import train_clf_one_seed
from utils import Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, \
    set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, \
    init_metric_dict


class GSAT(nn.Module):
    def __init__(self, clf, extractor, optimizer, scheduler, device, model_dir, dataset_name, num_class,
                 multi_label, random_state,
                 method_config, shared_config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.7)

        self.metrics = {
            'clf_acc': [],
            'clf_roc': [],
            'att_auroc': [],
            'precision_at_k': [],
            'clf_f1': [],
            'y_preds': [],
            'clf_labels': []
        }

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                     init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                # trans_idx, trans_val = transpose(data.edge_index, att)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        return edge_att, loss, loss_dict, clf_logits

    def eval_with_removed_edges(self, data, epoch, remove_ratio=0.1):  # 修改参数为比例
        att, _, clf_logits = self.eval_one_batch(data, epoch)


        num_edges = att.shape[0]
        remove_k = max(1, int(num_edges * remove_ratio))  # 确保至少移除1条边

        sorted_idx = np.argsort(-att)
        remove_idx = sorted_idx[:remove_k]  # 使用动态计算的k值

        mask = torch.ones_like(data.edge_index[0], dtype=torch.bool)
        mask[remove_idx] = False
        new_edge_index = data.edge_index[:, mask]

        new_clf_logits = self.clf(data.x, new_edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=None)
        new_loss, new_loss_dict = self.__loss__(att[mask], new_clf_logits, data.y, epoch)
        new_clf_preds, _ = get_preds(new_clf_logits, self.multi_label)
        new_clf_acc = (new_clf_preds == data.y).sum().item() / data.y.shape[0] if not self.multi_label else 0
        new_clf_roc = roc_auc_score(data.y.cpu().numpy(), new_clf_logits.cpu().numpy(), multi_class='ovr')
        new_clf_f1 = f1_score(data.y.cpu().numpy(), new_clf_preds.cpu().numpy(), average='weighted')
        new_clf_recall = recall_score(data.y.cpu().numpy(), new_clf_preds.cpu().numpy(), average='weighted')

        return new_clf_acc, new_clf_roc, new_clf_f1, new_clf_recall, new_loss_dict['loss']


    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        (all_removed_clf_acc, all_removed_clf_roc, all_removed_clf_f1) = ([] for i in range(3))

        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, att_auroc, precision, clf_acc, clf_roc, clf_f1 = self.log_epoch(epoch, phase, loss_dict, exp_labels,
                                                                                  att, precision_at_k,
                                                                                  data.y.data.cpu(), clf_logits,
                                                                                  batch=False)
            att_auroc = att_auroc if att_auroc is not None else 0
            precision = precision if precision is not None else 0
            clf_acc = clf_acc if clf_acc is not None else 0
            clf_roc = clf_roc if clf_roc is not None else 0
            clf_f1 = clf_f1 if clf_f1 is not None else 0

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            (all_clf_labels.append(data.y.data.cpu()),
             all_clf_logits.append(clf_logits))

            if phase == 'test':
                removed_clf_acc, removed_clf_roc, removed_clf_f1, removed_clf_recall, removed_loss = self.eval_with_removed_edges(
                    data, epoch, remove_k=self.k)
                all_removed_clf_acc.append(removed_clf_acc)
                all_removed_clf_roc.append(removed_clf_roc)
                all_removed_clf_f1.append(removed_clf_f1)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, clf_f1 = self.log_epoch(epoch, phase, all_loss_dict,
                                                                                      all_exp_labels, all_att,
                                                                                      all_precision_at_k,
                                                                                      all_clf_labels, all_clf_logits,
                                                                                      batch=False)

            if phase == 'test':
                avg_removed_clf_acc = np.mean(all_removed_clf_acc)
                avg_removed_clf_roc = np.mean(all_removed_clf_roc)
                avg_removed_clf_f1 = np.mean(all_removed_clf_f1)


                desc += f', removed_clf_acc: {avg_removed_clf_acc:.3f}, removed_clf_roc: {avg_removed_clf_roc:.3f}, removed_clf_f1: {avg_removed_clf_f1:.3f}'

            pbar.set_description(desc)
        # self.save_epoch_loss_dict(epoch, all_loss_dict)
        if phase == 'test':
            return (att_auroc, precision, clf_acc, clf_roc, clf_f1, all_loss_dict['loss'],
                    avg_removed_clf_acc, avg_removed_clf_roc, avg_removed_clf_f1)
        else:
            return att_auroc, precision, clf_acc, clf_roc, clf_f1, all_loss_dict['loss']
        # return att_auroc, precision, clf_acc, clf_roc, clf_f1, all_loss_dict['loss']
        # return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            # 获取训练、验证和测试集的结果
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)

            assert len(train_res) == 6, f"Expected 6 values, but got {len(train_res)}: {train_res}"

            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc

            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                         init_r=self.init_r)
            if (r == self.final_r or self.fix_r) and epoch > 10 and (
                    (valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                    or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                        and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):
                metric_dict = {'metric/best_clf_epoch': epoch,
                               'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx],
                               'metric/best_clf_valid': valid_res[main_metric_idx],
                               'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0],
                               'metric/best_x_roc_valid': valid_res[0],
                               'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1],
                               'metric/best_x_precision_valid': valid_res[1],
                               'metric/best_x_precision_test': test_res[1]}
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))




            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: '
                  f'Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, '
                  f'Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')

        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        eval_desc, att_auroc, precision, clf_acc, clf_roc, clf_f1 = self.get_eval_score(epoch, phase, exp_labels, att,
                                                                                        precision_at_k, clf_labels,
                                                                                        clf_logits, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, clf_f1

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds, y_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        try:
            clf_roc = roc_auc_score(clf_labels.cpu().numpy(), clf_logits.cpu().numpy(), multi_class='ovr')
        except ValueError:
            clf_roc = 0

        clf_f1 = f1_score(clf_labels.cpu().numpy(), clf_preds.cpu().numpy(), average='weighted')
        clf_recall = recall_score(clf_labels.cpu().numpy(), clf_preds.cpu().numpy(), average='weighted')


        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att

        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]


        self.metrics['clf_acc'].append(clf_acc)
        self.metrics['clf_roc'].append(clf_roc)
        self.metrics['att_auroc'].append(att_auroc)
        self.metrics['precision_at_k'].append(precision_at_k)
        self.metrics['clf_f1'].append(clf_f1)
        self.metrics['y_preds'].append(y_preds)
        self.metrics['clf_labels'].append(clf_labels)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, clf_recall: {clf_recall:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc, clf_f1

    def save_average_metrics_to_file(self, filename):
        average_metrics = {
            'clf_acc': np.mean(self.metrics['clf_acc']),
            'clf_roc': np.mean(self.metrics['clf_roc']),
            'att_auroc': np.mean(self.metrics['att_auroc']),
            'precision_at_k': np.mean(self.metrics['precision_at_k']),
            'clf_f1': np.mean(self.metrics['clf_f1'])
        }
        with open(filename, 'w') as f:
            json.dump(average_metrics, f, indent=4)

    def save_predictions_to_csv(self, filename):

        all_preds = np.concatenate(self.metrics['y_preds'], axis=0)  # 合并所有批次的预测值
        all_labels = np.concatenate(self.metrics['clf_labels'], axis=0)  # 合并所有批次的真实标签


        if all_preds.ndim == 2:
            all_preds = all_preds.flatten()
        if all_labels.ndim == 2:
            all_labels = all_labels.flatten()

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)


        df_preds = pd.DataFrame({
            'True_Labels': all_labels,
            'Predicted_Prob': all_preds
        })


        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")


        df_preds.to_csv(filename, index=False)

        print(f"Predictions and true labels saved to {filename}")

    def save_epoch_loss_dict(self, epoch, loss_dict):
        save_path = os.path.join(self.model_dir, f'loss_dict_epoch_{epoch}.json')


        with open(save_path, 'w') as f:
            json.dump(loss_dict, f, indent=4)

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max() + 1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name in ['Graph-SST2', 'sp2020', 'bpi2020pl', 'bpi2018al', 'bpi2017w']:

                condi = (y_dist == each_class) * (num_nodes > 10) * (num_nodes < 70)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na',
                             11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name in ['Graph-SST2', 'sp2020', 'bpi2020pl', 'bpi2018al', 'bpi2017w']:
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v + 1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in
                            element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label',
                                                                             None) is not None else torch.zeros(
                viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name,
                                         norm=self.viz_norm_att, mol_type=mol_type, coor=coor)




    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def compute_node_weights(edge_index, edge_weights, num_nodes):

        node_weights = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            node_weights[src] += edge_weights[i]
            node_weights[dst] += edge_weights[i]
        return node_weights

    @staticmethod
    def plot_node_weights(node_weights, node_labels=None):

        # 转换为 numpy 数组
        node_weights = node_weights.cpu().numpy()

        # 设置标签
        if node_labels is None:
            node_labels = [f"活动 {i + 1}" for i in range(len(node_weights))]

        # 绘制饼图
        plt.figure(figsize=(8, 8))
        plt.pie(node_weights, labels=node_labels, autopct='%1.1f%%', startangle=140)
        plt.title("活动权重分布（信息量贡献）")
        plt.show()


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size,
                                                                                    splits, random_state,
                                                                                    data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(model, extractor, optimizer, scheduler, device, log_dir, dataset_name, num_class,
                aux_info['multi_label'], random_state, method_config, shared_config)

    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    # gsat.save_average_metrics_to_file(f'{data_dir}/{dataset_name}/metrics_log.json')
    gsat.save_predictions_to_csv(f'{data_dir}/{dataset_name}/y_preds/y_preds.csv')

    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', default='bpi2017w', type=str, help='dataset used')
    parser.add_argument('--backbone', default='GIN', type=str, help='backbone model used')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (
                time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name,
                                                       method_name, device, random_state)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (
            time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)



if __name__ == '__main__':
    main()
