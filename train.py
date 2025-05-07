#!/usr/bin/env python3

import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from transformers import AutoTokenizer
from omegaconf import OmegaConf

from datasets import load_dataset

from tqdm.auto import tqdm, trange

from model import TextGraphClassifier
from dataset import TextGraphDataset, DataCollator, is_valid_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    return parser.parse_args(argv)


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
