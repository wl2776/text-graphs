import torch
from torch_geometric.nn import GCNConv
from transformers import AutoModel


class GCNConvBlock1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(GCNConvBlock1, self).__init__()
        self.conv = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.dropout = torch.nn.Dropout(p=p)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class GCNConvBlock2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(GCNConvBlock2, self).__init__()
        self.conv = GCNConv(in_channels=in_channels, out_channels=out_channels)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.dropout(x)
        return x


class MultiLayerGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super(MultiLayerGCN, self).__init__()

        # первые слои - с dropout и relu
        first_layer = GCNConvBlock1(in_channels=1, out_channels=hidden_channels)
        mid_layers = [GCNConvBlock1(in_channels=hidden_channels, out_channels=hidden_channels) for _ in range(num_layers - 2)]

        last_layer = GCNConvBlock2(in_channels=hidden_channels, out_channels=hidden_channels)

        self.convs = torch.nn.ModuleList([first_layer] + mid_layers + [last_layer])
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for c in self.convs:
            x = c(x, edge_index)
        return self.fc_out(x)    


class TextGraphClassifier(torch.nn.Module):
    def __init__(self, config, device=None, freeze_transformer=True):
        super(TextGraphClassifier, self).__init__()
        self.config = config
        self.transformer = AutoModel.from_pretrained(config.name)
        self.gcn = MultiLayerGCN(config.hidden_dim, config.hidden_dim, config.num_layers)
        self.adapter = torch.nn.Linear(config.hidden_dim, config.text_embedding_dim)
        self.final_classifier = torch.nn.Linear(config.text_embedding_dim * 2, 1)

        if freeze_transformer:
            for param in self.transformer.base_model.parameters():
                param.requires_grad = False

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
    
    def forward(self, batch):
        text_embedding = self.transformer(
            input_ids=batch['question_encoding']['input_ids'].to(self.device),
            attention_mask=batch['question_encoding']['attention_mask'].to(self.device)
        ).last_hidden_state.mean(dim=1)        

        gcn_output = self.gcn(batch['graph'].x.to(self.device), batch['graph'].edge_index.to(self.device))
        gcn_embedding = gcn_output.mean(dim=0)
        
        gcn_embedding_expanded = self.adapter(gcn_embedding.unsqueeze(0)).expand_as(text_embedding)
        
        combined = torch.cat([text_embedding, gcn_embedding_expanded], dim=1)
        
        return self.final_classifier(combined)

