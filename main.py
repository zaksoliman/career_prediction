from prediction_models.title_seq_rnn import Model as simple_model
from prediction_models.bow_lstm import Model as bow_model
from prediction_models.multi_label_fasttext import  Model as multi_label_model
from helpers.loader import load_data
import sys

def main():

    seq_model = None
    ds = sys.argv[1]
    #model = sys.argv[1]

    model_names = {
        '1': '',
        '2': 'start_tags',
        '3': 'bow',
        '4': 'bow_durations',
        '5': 'multi_fasttext',
        '6': 'big_multi_ft'
    }
    dataset_names = {
        '1': 'title_sequences',
        '2': 'title_sequences',
        '3': 'title_sequences',
        '4': 'title_sequences_durations',
        '5': 'title_embedding_sequences_multi_label',
        '6': 'big_title_embedding_sequences_multi_label'
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
    elif ds in "5":
        print("Multi label model selected...")
        title_id, train_inputs, train_targets, test_inputs, test_targets, token_id, emb_dim, max_seq_len = load_data(path, multi=True)
        seq_model = multi_label_model(
            name=model_names[ds],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            embedding_dim=emb_dim,
            n_titles=len(title_id),
            n_tokens=len(token_id),
            max_timesteps=max_seq_len,
            num_layers=1,
            freeze_emb=False,
            n_epochs=1500,
            learning_rate=0.001,
            batch_size=100
        )
    elif ds in "6":
        print("Big Multi label model selected...")
        title_id, train_inputs, train_targets, test_inputs, test_targets, token_id, emb_dim, max_seq_len = load_data(path, multi=True)
        seq_model = multi_label_model(
            name=model_names[ds],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            embedding_dim=emb_dim,
            emb_path="/data/rali7/Tmp/solimanz/data/datasets/6/embeddings_big.npy",
            n_titles=len(title_id),
            n_tokens=len(token_id),
            max_timesteps=max_seq_len,
            num_layers=1,
            freeze_emb=True,
            n_epochs=1500,
            learning_rate=0.001,
            batch_size=100
        )

    if seq_model:
        seq_model.train()

if __name__ == "__main__":
    main()
