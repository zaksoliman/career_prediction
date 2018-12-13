from helpers.loader import load_data
import argparse
from prediction_models.title_seq_rnn import Model
from helpers.batcher import Batcher
import numpy as np
import os


if __name__ == '__main__':
    path = f"/data/rali7/Tmp/solimanz/LBJ/dataset"
    emb_path = os.path.join(path, 'embeddings_reduced.npy')

    title_to_id, train_data, train_max_seq_len, _, _, _ = load_data(os.path.join(path, 'train', 'train.json'))
    _, valid_data, valid_max_seq_len, _, _, _ = load_data(os.path.join(path, 'valid', 'valid.json'))

    config = {
        "train_data": train_data,
        "test_data": valid_data,
        "batcher": Batcher,
        "n_titles":len(title_to_id),
        "max_timesteps": max(train_max_seq_len, valid_max_seq_len),
        "use_dropout": True,
        "num_layers": 1,
        "keep_prob": 0.5,
        "hidden_dim": 250,
        "use_attention": False,
        "attention_dim": 100,
        "use_embedding": True,
        "embedding_dim": 300,
        "use_fasttext": True,
        "freeze_emb": False,
        "max_grad_norm": 5,
        "rnn_cell_type": 'LSTM',
        "use_bow": False,
        "vocab_size": -1,
        "learning_rate": 0.001,
        "batch_size": 200,
        "n_epochs": 1000,
        "log_interval": 100,
        "store_model": True,
        "restore": True,
        "store_dir": "/data/rali7/Tmp/solimanz/data/models/",
        "log_dir": ".log/",
        "name": "CANAI",
        "emb_path": emb_path
    }

    model = Model(**config)
    model.train()


