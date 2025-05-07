#!/usr/bin/env python3

import sys
import argparse
import json
import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel

from omegaconf import OmegaConf

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    return parser.parse_args(argv)

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch_geometric.data import Data, Batch
import networkx as nx
import json

# Переменные окружения
DEVICE = torch.device("cuda" if torch.device("cuda").is_available() else "cpu")
CHECKPOINT_PATH = 'checkpoint.pt'
DATA_FILE = 'data.tsv'
MAX_LENGTH = 128

# Загрузка чекпоинта
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    return checkpoint

# Класс для датасета
class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['question']
        graph_json = row['graph']
        json_str = graph_json.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('nan', '\"nan\"')
        graph_json = json.loads(json_str)
        
        graph_nx = nx.node_link_graph(graph_json, edges="links")
        nodes = list(graph_nx.nodes)
        edges = [(u, v) for u, v in graph_nx.edges()]
        
        question_encoding = self.tokenizer(question, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            'question_encoding': question_encoding,
            'graph': Data(
                x=torch.zeros((len(nodes), 1)),
                edge_index=torch.tensor(edges).t().contiguous(),
                num_nodes=len(nodes)
            )
        }

# Генерация предсказаний
def generate_predictions(model, loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)
            results.extend(predictions.squeeze().tolist())  # Фиксируем предсказания
    return results

# Основной рабочий цикл
def main():
    # Загрузка чекпоинта
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['name'])
    model.load_state_dict(model_state_dict)
    model.to(DEVICE)

    # Загрузка данных
    df = pd.read_csv(DATA_FILE, sep='\t')
    dataset = InferenceDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], collate_fn=lambda x: x)

    # Генерируем предсказания
    predictions = generate_predictions(model, loader)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()