import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import networkx as nx

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
    def __init__(self, tokenizer, max_length=128, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = not inference

    def __call__(self, samples):
        question_encodings = [item['question_encoding'] for item in samples]
        graphs = [item['graph'] for item in samples]
        if self.train:
            labels = [item['label'] for item in samples]
        else:
            sample_ids = [item['sample_id'] for item in samples]
        
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
        
        result = {
            'question_encoding': padded_inputs,
            'graph': batch_graphs,
        }
        
        if self.train:
            result['label'] = torch.stack(labels)
        else:
            result['sample_id'] = torch.tensor(sample_ids)

        return result


def is_valid_json(example, column_name='graph'):
    """функция для чистки неправильных описаний графов в формате json"""
    try:
        json_str = example[column_name]
        json_str = json_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null').replace('nan', '\"nan\"')
        _ = json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


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
            'sample_id': row['sample_id'],
            'question_encoding': question_encoding,
            'graph': Data(
                x=torch.zeros((len(nodes), 1)),
                edge_index=torch.tensor(edges).t().contiguous(),
                num_nodes=len(nodes)
            )
        }

