from helpers.loader import load_data
import argparse
from prediction_models.title_seq_rnn import Model
#from prediction_models.lstm_rnn_classifier import Model
from prediction_models.MLP import FeedFowardModel
from helpers.batcher import Batcher, BOW_Batcher
import numpy as np
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run models')
    parser.add_argument('-ds', '--dataset', required=True, help='Choose dataset',
                    choices=['top550', 'reduced7000'])
    parser.add_argument('-r', '--representation', required=True, help='Choose a method to represent textual data',
                        choices=['jobid', 'bow', 'fasttext', 'emb'])
    parser.add_argument('-m', '--model', required=True, choices=['mlp', 'simple_rnn'])
    parser.add_argument('-t', '--task', required=True,
                        choices=['train', 'test', 'tsne'])

    args = parser.parse_args()
    path = f"/data/rali7/Tmp/solimanz/data/datasets/{args.dataset}"
    model = None

    if args.representation == 'fasttext':
        model_name = 'title_emb'
    else:
        model_name = args.representation

    config = {
        "train_targets": None,
        "test_targets": None,
        "use_dropout": True,
        "num_layers": 1,
        "keep_prob": 0.5,
        "hidden_dim": 50,
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
        "batch_size": 200,
        "n_epochs": 1000,
        "log_interval": 100,
        "store_model": True,
        "restore": True,
        "store_dir": "/data/rali7/Tmp/solimanz/data/models/",
        "log_dir": ".log/",
        "name": model_name,
        "emb_path": ''
    }


    if args.model == 'simple_rnn':
        if args.representation == 'jobid':
            title_to_id, train_data, test_data, max_seq_len = load_data(os.path.join(path, 'jobid', 'data.json'))

            config["train_data"] = train_data
            config["n_titles"] = len(title_to_id)
            config["test_data"] = test_data
            config["batcher"] = Batcher
            config["max_timesteps"] = max_seq_len

            model = Model(**config)

        elif args.representation == 'bow':
            print("Loading bow data...")
            title_to_id, train_data, train_targets, test_data, test_targets, max_seq_len, vocab_size = load_data(
                os.path.join(path, args.representation, 'data.json'),
                bow=True)

            config["train_data"] = train_data
            config["n_titles"] = len(title_to_id)
            config["test_data"] = test_data
            config["batcher"] = BOW_Batcher
            config["max_timesteps"] = max_seq_len
            config["train_targets"] = train_targets
            config["test_targets"] = test_targets
            config["vocab_size"] = vocab_size
            config["use_bow"] = True

            model = Model(**config)
        elif args.representation == 'fasttext':
            emb_path = os.path.join(path, args.representation, 'embeddings.npy')
            title_to_id, train_data, test_data, max_seq_len = load_data(os.path.join(path, 'jobid', 'data.json'))

            config["train_data"] = train_data
            config["n_titles"] = len(title_to_id)
            config["test_data"] = test_data
            config["batcher"] = Batcher
            config["max_timesteps"] = max_seq_len
            config["emb_path"] = emb_path
            config["use_fasttext"] = True
            config["embedding_dim"] = 300

            model = Model(**config)

    elif args.model == 'mlp':

        data_path = f"/data/rali7/Tmp/solimanz/data/datasets/feed_forward/{args.dataset}/{args.representation}"

        with open(os.path.join(data_path, "train.npy"), "rb") as f:
            X_train = np.load(f)
        with open(os.path.join(data_path, "test.npy"), "rb") as f:
            X_test = np.load(f)
        with open(os.path.join(data_path, "train_targets.npy"), "rb") as f:
            train_targets = np.load(f)
        with open(os.path.join(data_path, "test_targets.npy"), "rb") as f:
            test_targets = np.load(f)

        model = FeedFowardModel(
            train_data=X_train,
            test_data=X_test,
            train_targets=train_targets,
            test_targets=test_targets,
            input_dim=X_train.shape[1],
            n_labels=train_targets.shape[1],
            learning_rate=0.01,
            n_epochs=1000,
            batch_size=500,
            n_layers=1,
            hiddden_dim=252,
            use_emb=args.representation == 'emb',
            ds_name=args.dataset
        )

    if model and args.task == 'train':
        model.train()
    elif model and args.task =='test':
        model.test()
    elif model and args.task =='tsne':
        model.tSNE(args.dataset, args.representation)

