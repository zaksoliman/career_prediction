from prediction_models.title_seq_rnn import Model as simple_model
from prediction_models.bow_lstm import Model as bow_model
from helpers.loader import load_data
import sys

def main():

    ds = sys.argv[1]
    #model = sys.argv[1]

    model_names = {
        '1': '',
        '2': 'start_tags',
        '3': '',
        '4': 'durations'
    }
    dataset_names = {
        '1': 'title_sequences',
        '2': 'title_sequences',
        '3': 'title_sequences',
        '4': 'title_sequences_durations'
    }

    path = f"/data/rali7/Tmp/solimanz/data/datasets/{ds}/{dataset_names[ds]}.json"

    if ds in "12":
        mapping, train_data, test_data, max_seq_len = load_data(path)
        seq_model = simple_model(
            name=model_names[ds],
            train_data=train_data,
            test_data=test_data,
            embedding_dim=100,
            n_titles=len(mapping),
            max_timesteps=max_seq_len,
            class_mapping=mapping,
            num_layers=1,
            n_epochs=1500,
            learning_rate=0.001
        )

        seq_model.train()
    elif ds in "34":
        title_id, train_inputs, train_targets, test_inputs, test_targets, vocab_id, max_seq_len = load_data(path, bow=True)
        seq_model = bow_model(
            name=model_names[ds],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            embedding_dim=100,
            n_titles=len(title_id),
            vocab_size=len(vocab_id),
            max_timesteps=max_seq_len,
            num_layers=1,
            n_epochs=1500,
            learning_rate=0.001,
            batch_size=100
        )

        seq_model.train()

if __name__ == "__main__":
    main()
