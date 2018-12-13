from helpers.loader import load_data
import argparse
from prediction_models.cnn_lstm_classif import Model
from helpers.batcher import SkillBatcher
import os


if __name__ == '__main__':

    path = f"/data/rali7/Tmp/solimanz/LBJ/dataset"
    model_name = 'EncDec'
    emb_path = os.path.join(path, 'embeddings_reduced.npy')
    title_to_id, train, train_max_seq_len, token_id, max_skills, skill_embs_path = load_data(os.path.join(path, 'train', 'train.json'))
    _, valid, valid_max_seq_len, _, _, _ = load_data(os.path.join(path, 'valid', 'valid.json'))

    config = {
        "train_data": train,
        "test_data": valid,
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
        "name": model_name,
        "emb_path": emb_path
    }


    config["n_titles"] = len(title_to_id)
    config["n_skills"] = len(token_id)
    config["max_skills"] = max_skills
    config["batcher"] = SkillBatcher
    config["emb_path"] = emb_path
    config["use_fasttext"] = True
    config["embedding_dim"] = 300
    config["skill_emb_path"] = skill_embs_path

    model = Model(**config)
    model.train()

