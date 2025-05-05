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
from datasets import load_dataset

from tqdm.auto import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    return parser.parse_args(argv)


class MultiLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super(MultiLayerGCN, self).__init__()
        # Первый слой должен принимать на вход 1 признак и выводить hidden_channels
        first_layer = GCNConv(in_channels=1, out_channels=hidden_channels)
        last_layers = [GCNConv(in_channels=hidden_channels, out_channels=hidden_channels) for _ in range(num_layers - 1)]
        self.convs = torch.nn.ModuleList([first_layer] + last_layers)
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index)
        return self.fc_out(x)    


class TextGraphClassifier(torch.nn.Module):
    def __init__(self, config):
        super(TextGraphClassifier, self).__init__()
        self.config = config
        self.transformer = AutoModel.from_pretrained(config.name)
        self.gcn = MultiLayerGCN(config.hidden_dim, config.hidden_dim, config.num_layers)
        self.adapter = torch.nn.Linear(config.hidden_dim, config.text_embedding_dim)
        self.final_classifier = torch.nn.Linear(config.text_embedding_dim * 2, 1)

        for param in self.transformer.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, batch):
        text_embedding = self.transformer(
            input_ids=batch['question_encoding']['input_ids'].to(device),
            attention_mask=batch['question_encoding']['attention_mask'].to(device)
        ).last_hidden_state.mean(dim=1)        

        gcn_output = self.gcn(batch['graph'].x.to(device), batch['graph'].edge_index.to(device))
        gcn_embedding = gcn_output.mean(dim=0)
        
        gcn_embedding_expanded = self.adapter(gcn_embedding.unsqueeze(0)).expand_as(text_embedding)
        
        combined = torch.cat([text_embedding, gcn_embedding_expanded], dim=1)
        
        return self.final_classifier(combined)


class TextGraphDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.data = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        question = example['question']
        json_str = example['graph']
        json_str = json_str.replace("'", '\"').replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('nan', '\"nan\"')
        json_str = json_str.replace("\'source", '"source')
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
            ),
            'label': torch.tensor(example['correct'], dtype=torch.long)
        }


class DataCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        question_encodings = [item['question_encoding'] for item in samples]
        graphs = [item['graph'] for item in samples]
        labels = [item['label'] for item in samples]
        
        max_seq_len = min(max(len(enc.input_ids[0]) for enc in question_encodings), self.max_length)
        
        padded_input_ids = []
        attention_masks = []
        for enc in question_encodings:
            seq_len = len(enc.input_ids[0])
            pad_len = max_seq_len - seq_len
            
            if seq_len > max_seq_len:
                input_ids = enc.input_ids[0][:max_seq_len]
                attention_mask = enc.attention_mask[0][:max_seq_len]
            else:
                # manual padding
                padding_tensor = torch.full((pad_len,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([enc.input_ids[0], padding_tensor])
                
                mask_padding_tensor = torch.zeros((pad_len,), dtype=torch.long)
                attention_mask = torch.cat([enc.attention_mask[0], mask_padding_tensor])
            
            padded_input_ids.append(input_ids)
            attention_masks.append(attention_mask)
        
        padded_inputs = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks)
        }
        
        batch_graphs = Batch.from_data_list(graphs)
        batch_graphs.edge_index = batch_graphs.edge_index.type(torch.long) 
        
        return {
            'question_encoding': padded_inputs,
            'graph': batch_graphs,
            'label': torch.stack(labels)
        }


def is_valid_json(example):
    """функция для чистки неправильных описаний графов в формате json"""
    try:
        json_str = example['graph']
        json_str = json_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('nan', '\"nan\"')
        _ = json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def compute_accuracy(predictions, targets):
    """
    Compute accuracy score.
    :param predictions: Raw prediction scores (logits)
    :param targets: Ground-truth labels
    :return: Accuracy value
    """
    predicted_labels = (predictions >= 0.5).float()  # Бинаризация предсказаний
    correct_predictions = (predicted_labels == targets).sum().item()
    total_samples = targets.size(0)
    return correct_predictions / total_samples


def main(argv):
    args = get_args(argv)

    cfg = OmegaConf.load(args.config)
    torch.manual_seed(cfg.seed)
    
    dataset = load_dataset(path=cfg.data.name, split='train').filter(is_valid_json)
    
    train_val_split = dataset.train_test_split(test_size=0.2, seed=cfg.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    train_dataset = TextGraphDataset(train_val_split['train'], tokenizer, max_length=cfg.data.max_length)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, 
                              collate_fn=DataCollator(tokenizer))

    val_dataset = TextGraphDataset(train_val_split['test'], tokenizer, max_length=cfg.data.max_length)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, collate_fn=DataCollator(tokenizer))
    
    model = DataParallel(TextGraphClassifier(cfg.model).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in trange(cfg.train.epochs):
        model.train()
        total_loss = 0
        train_acc = 0
        for batch in tqdm(train_loader, desc='Train'):
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch['label'].float().view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            acc = compute_accuracy(predictions.detach().cpu(), batch['label'].float().view(-1, 1))
            train_acc += acc
        
        avg_train_acc = train_acc / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for val_batch in tqdm(val_loader, desc='Val'):
                predictions = model(val_batch)
                loss = criterion(predictions, val_batch['label'].float().view(-1, 1).to(device))
                val_loss += loss.item()
            
                acc = compute_accuracy(predictions.cpu(), val_batch['label'].float().view(-1, 1))
                val_acc += acc
    
        avg_val_acc = val_acc / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)    

        print(f'Epoch [{epoch+1}/{cfg.train.epochs}], '
              f'train loss: {total_loss / len(train_loader):.4f}, '
              f'train acc: {avg_train_acc:.4f}, '
              f'val loss: {val_loss / len(val_loader):.4f}, '
              f'val acc: {avg_val_acc:.4f}')        

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }, 'best_checkpoint.pth')


if __name__ == '__main__':
    main(sys.argv[1:])
