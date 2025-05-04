import sys
import argparse
import json
from functools import partial
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    return parser.parse_args(argv)



class MultiLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super(MultiLayerGCN, self).__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.convs[-1](x, edge_index)
        return self.fc_out(x)
    

# Main classifier combining text and graph processing
class TextGraphClassifier(torch.nn.Module):
    def __init__(self, config):
        super(TextGraphClassifier, self).__init__()
        self.config = config
        self.transformer = AutoModel.from_pretrained(config.name)
        self.gcn = MultiLayerGCN(config.hidden_dim, config.hidden_dim, config.num_layers)
        self.final_classifier = torch.nn.Linear(config.hidden_dim * 2, 1)
    
    def forward(self, batch):
        # Process text via transformer
        text_embedding = self.transformer(**batch['question_encoding'].to(device)).last_hidden_state.mean(dim=1)
        
        # Process graph via GCN
        gcn_output = self.gcn(batch['graph'].x.to(device), batch['graph'].edge_index.to(device))
        gcn_embedding = gcn_output.mean(dim=0)
        
        # Concatenate text and graph embeddings
        combined = torch.cat([text_embedding, gcn_embedding.unsqueeze(0)], dim=1)
        
        # Predict label
        return torch.sigmoid(self.final_classifier(combined))


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
        
        # Find maximum sequence length across the batch
        max_seq_len = min(max(len(enc.input_ids[0]) for enc in question_encodings), self.max_length)
        
        # Manual padding and truncation
        padded_input_ids = []
        attention_masks = []
        for enc in question_encodings:
            seq_len = len(enc.input_ids[0])
            pad_len = max_seq_len - seq_len
            
            # Truncate long sequences
            if seq_len > max_seq_len:
                input_ids = enc.input_ids[0][:max_seq_len]
                attention_mask = enc.attention_mask[0][:max_seq_len]
            else:
                # Create padding tensor
                padding_tensor = torch.full((pad_len,), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([enc.input_ids[0], padding_tensor])
                
                # Same operation for attention masks
                mask_padding_tensor = torch.zeros((pad_len,), dtype=torch.long)
                attention_mask = torch.cat([enc.attention_mask[0], mask_padding_tensor])
            
            padded_input_ids.append(input_ids)
            attention_masks.append(attention_mask)
        
        # Stack tensors together
        padded_inputs = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks)
        }
        
        # Pack graphs using PyTorch Geometric's Batch
        batch_graphs = Batch.from_data_list(graphs)
        
        return {
            'question_encoding': padded_inputs,
            'graph': batch_graphs,
            'label': torch.stack(labels)
        }


def is_valid_json(example):
    try:
        json_str = example['graph']
        json_str = json_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('nan', '\"nan\"')
        _ = json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def main(argv):
    args = get_args(argv)

    cfg = OmegaConf.load(args.config)
    
    dataset = load_dataset(path=cfg.data.name, split='train').filter(is_valid_json)
    
    train_val_split = dataset.train_test_split(test_size=0.2, seed=cfg.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    train_dataset = TextGraphDataset(train_val_split['train'], tokenizer, max_length=cfg.data.max_length)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, 
                              collate_fn=DataCollator(tokenizer))

    val_dataset = TextGraphDataset(train_val_split['test'], tokenizer, max_length=cfg.data.max_length)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, collate_fn=DataCollator(tokenizer))
    
    model = TextGraphClassifier(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            predictions = model(batch)
            loss = criterion(predictions, batch['labels'].view(-1, 1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_loader:
                predictions = model(val_batch)
                loss = criterion(predictions, val_batch['labels'].view(-1, 1).to(device))
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{cfg.epochs}], train loss: {total_loss / len(train_loader):.4f}, val loss: {val_loss / len(val_loader):.4f}')


if __name__ == '__main__':
    main(sys.argv[1:])
