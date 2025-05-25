import os
import numpy as np
import json
# from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import pickle
import pandas as pd
import torch
import torch.nn as nn


def load_and_process_event_log(filepath):
    df = pd.read_csv(filepath)
    df = df[['CASE_ID', 'ACTIVITY', 'TIMESTAMP', '...']] # ... add new columns
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.sort_values(by=['CASE_ID', 'TIMESTAMP']).reset_index(drop=True)
    df['CASE_ID'] = df['CASE_ID'].astype('category').cat.codes + 1
    return df

def generate_edge_index(df, output_dir, dataset_name):
    edge_index = []
    for case_id, case_df in df.groupby("CASE_ID"):
        node_indices = case_df.index.tolist()
        edges = [[node_indices[i], node_indices[i + 1]] for i in range(len(node_indices) - 1)]
        edge_index.extend(edges)

    output_path = os.path.join(output_dir, f"{dataset_name}_edge_index.txt")
    with open(output_path, "w") as f:
        for edge in edge_index:
            f.write(f"{edge[0]} {edge[1]}\n")
    print("Edge index file generated.")
def generate_node_indicator(df, output_dir, dataset_name):
    node_indicator = df['CASE_ID'].values
    output_path = os.path.join(output_dir, f"{dataset_name}_node_indicator.txt")
    np.savetxt(output_path, node_indicator, fmt="%d")
    print("Node indicator file generated.")


def generate_graph_labels(df, output_dir, dataset_name):
    graph_labels = df.drop_duplicates("CASE_ID")[['CASE_ID', 'Labels']]
    output_path = os.path.join(output_dir, f"{dataset_name}_graph_labels.txt")
    np.savetxt(output_path, graph_labels['Labels'].values, fmt="%d")
    print("Graph labels file generated.")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def generate_node_features_bert_mlp(df, output_dir, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()

    # 初始化MLP模型
    mlp_model = MLP(input_dim=10, hidden_dim=256, output_dim=768).to(device)
    all_embeddings = []

    # 标准化器
    scaler = StandardScaler()

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), desc="Processing events", total=len(df)):
            activity = row["ACTIVITY"]


            inputs = tokenizer(activity, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()


            additional_features = row[['FROM_FIRST', 'OverspentAmount', 'DURATION', 'DAY']].values

            additional_features = pd.to_numeric(additional_features, errors='coerce')  # 将非数值值变成NaN
            additional_features = np.nan_to_num(additional_features, nan=0.0)  # 将NaN替换为0

            extended_features = [
                additional_features[0],
                additional_features[1],
                additional_features[2],
                additional_features[3],
                additional_features[4],  # add new_features

            ]

            extended_features = scaler.fit_transform(np.array(extended_features).reshape(1, -1)).flatten()

            extended_features_tensor = torch.tensor(extended_features, dtype=torch.float32).to(device)

            mapped_features = mlp_model(extended_features_tensor.unsqueeze(0)).cpu().numpy().squeeze()

            combined_features = cls_embedding + mapped_features  # 使用加法融合
            all_embeddings.append(combined_features)


    all_embeddings = np.array(all_embeddings)

    # 保存最终的node_features.pkl
    output_path = os.path.join(output_dir, f"{dataset_name}_node_features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"Node features file generated with shape {all_embeddings.shape} using BERT embeddings and enhanced features.")


def generate_sentence_tokens(df, output_dir, dataset_name):
    sentence_tokens = {
        str(case_id): [extract_abbreviation(activity) for activity in case_df["ACTIVITY"].tolist()]
        for case_id, case_df in df.groupby("CASE_ID")
    }
    output_path = os.path.join(output_dir, f"{dataset_name}_sentence_tokens.json")
    with open(output_path, "w") as f:
        json.dump(sentence_tokens, f, indent=2)
    print("Sentence tokens JSON file generated.")


def generate_case_split_indices(df, output_dir, dataset_name, test_size=0.2, val_size=0.2, random_seed=42):
    case_ids = df['CASE_ID'].unique()
    train_cases, test_cases = train_test_split(case_ids, test_size=test_size, random_state=random_seed)
    train_cases, val_cases = train_test_split(train_cases, test_size=val_size / (1 - test_size), random_state=random_seed)

    split_indices = np.zeros(len(case_ids), dtype=int)
    split_indices[np.isin(case_ids, val_cases)] = 1  # 验证集标记为 1
    split_indices[np.isin(case_ids, test_cases)] = 2  # 测试集标记为 2

    output_path = os.path.join(output_dir, f"{dataset_name}_split_indices.txt")
    np.savetxt(output_path, split_indices, fmt="%d")
    print("Split indices file generated for cases.")

# 主函数：生成所有文件
def main(filepath, dataset_name):
    output_dir = create_output_dir(dataset_name)
    df = load_and_process_event_log(filepath)
    generate_edge_index(df, output_dir, dataset_name)
    generate_node_indicator(df, output_dir, dataset_name)
    generate_graph_labels(df, output_dir, dataset_name)
    generate_node_features_bert_mlp(df, output_dir, dataset_name)
    generate_sentence_tokens(df, output_dir, dataset_name)
    generate_case_split_indices(df, output_dir, dataset_name)
    print("All files generated successfully.")



