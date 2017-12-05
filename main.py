from prediction_models.title_seq_rnn import Model
from helpers.loader import load_data
import sys

def main():

    ds = sys.argv[1]
    names = {'1': '',
             '2': 'start_tags',
             '3': 'bow'}

    path = f"/data/rali7/Tmp/solimanz/data/datasets/{ds}/title_sequences.json"
    mapping, train_data, test_data, max_seq_len = load_data(path)

    seq_model = Model(
            name=names[ds],
            train_data=train_data,
            test_data=test_data,
            embedding_dim=100,
            n_titles=len(mapping),
            max_timesteps=max_seq_len,
            class_mapping=mapping,
            num_layers=1,
            n_epochs=1500,
            learning_rate=0.001)

    seq_model.train()

if __name__ == "__main__":
    main()
