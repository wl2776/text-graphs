#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

import pandas as pd
from omegaconf import OmegaConf

from model import TextGraphClassifier
from dataset import InferenceDataset, DataCollator, is_valid_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(argv):
    parser = argparse.ArgumentParser(description='Run model')
    parser.add_argument('-c', '--checkpoint', type=Path, required=True,
                        help='Path to checkpoint')
    parser.add_argument('-i', '--input', type=Path, help='Input data')
    parser.add_argument('-o', '--output', type=Path, help='Output path')
    parser.add_argument('--config', type=Path, default=None)
    return parser.parse_args(argv)


def generate_predictions(model, loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)
            results.extend(predictions.squeeze().tolist())
    return results


def clean_df_by_json_column(df, column_name='graph'):
    """
    Функция очищает DataFrame, удаляя строки с некорректным JSON в указанном столбце.
    
    :param df: Pandas DataFrame
    :param column_name: Название колонки с JSON-данными
    :return: Очищенный DataFrame
    """
    
    valid_rows = [is_valid_json(row, column_name) for _, row in df.iterrows()]
    
    return df.loc[valid_rows]
    

def main(argv):
    args = get_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif args.config is not None:
        config = OmegaConf.load(args.config)
    else:
        raise RuntimeError("Cannot find config")

    model_state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in model_state_dict.keys()):
        model_state_dict = {key[7:]: value for key, value in model_state_dict.items()}  # удалить module., оставшиеся от DDP

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = TextGraphClassifier(config=config.model, device=device)
    model.load_state_dict(model_state_dict)
    model.to(device)

    df = pd.read_csv(args.input, sep='\t')
    df = clean_df_by_json_column(df, 'graph')
    dataset = InferenceDataset(df, tokenizer, config['data']['max_length'])
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], collate_fn=DataCollator(tokenizer, inference=True))

    predictions = generate_predictions(model, loader)
    print("Predictions:", predictions)


if __name__ == '__main__':
    main(sys.argv[1:])