from helpers.loader import load_dat_skills as load_data
import argparse
from prediction_models.cnn_lstm_classif import Model
from helpers.batcher import SkillBatcher
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run models')
    parser.add_argument('-ds', '--dataset', required=True, help='Choose dataset',
                    choices=['top550', 'reduced7000'])
    parser.add_argument('-t', '--task', required=True,
                        choices=['train', 'test', 'tsne'])

    args = parser.parse_args()
    path = f"/data/rali7/Tmp/solimanz/data/datasets/{args.dataset}"
    model_name = 'title_emb_and_skills'

    config = {
        "train_targets": None,
        "test_targets": None,
        "use_dropout": True,
        "num_layers": 1,
        "keep_prob": 0.5,
        "hidden_dim": 100,
        "use_attention": False,
        "attention_dim": 100,
        "use_embedding": True,
        "embedding_dim": 100,
        "use_fasttext": False,
        "freeze_emb": False,
        "max_grad_norm": 5,
        "rnn_cell_type": 'LSTM',
        "use_bow": False,
        "vocab_size": -1,
        "learning_rate": 0.001,
        "batch_size": 100,
        "n_epochs": 1000,
        "log_interval": 100,
        "n_filters": 2,
        "kernel_sizes": list(range(2,5)),
        "store_model": True,
        "restore": True,
        "store_dir": "/data/rali7/Tmp/solimanz/data/models/",
        "log_dir": ".log/",
        "name": model_name,
        "emb_path": ''
    }

    emb_path = os.path.join(path, 'fasttext', 'embeddings.npy')
    title_id, train, test, max_seq_len, token_id, max_skills, skill_embs_path = load_data(os.path.join(path,
                                                                                                       'skill_embs',
                                                                                                       'data.json'))

    config["train_data"] = train
    config["n_titles"] = len(title_id)
    config["n_skills"] = len(token_id)
    config["max_skills"] = max_skills
    config["test_data"] = test
    config["batcher"] = SkillBatcher
    config["max_timesteps"] = max_seq_len
    config["emb_path"] = emb_path
    config["use_fasttext"] = True
    config["embedding_dim"] = 300
    config["skill_emb_path"] = skill_embs_path

    model = Model(**config)

    if model and args.task == 'train':
        model.train()
    elif model and args.task =='test':
        model.test()
    elif model and args.task =='tsne':
        model.tSNE(args.dataset, args.representation)

