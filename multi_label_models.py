from prediction_models.mutli_label_sub_seq import Model as multi_model
from helpers.loader import load_data
import sys

def main():

    seq_model = None
    ds = sys.argv[1]
    #model = sys.argv[1]

    model_names = {
        '1': 'top7000_multi'
    }
    dataset_names = {
        '1': 'multilabel_title_sequences'
    }

    path = f"/data/rali7/Tmp/solimanz/data/datasets/multilabel/{ds}/{dataset_names[ds]}.json"

    if ds in "1":
        mapping, train_inputs, train_targets, test_inputs, max_seq_len = load_data(path, multi=True)
        seq_model = multi_model(
            name=model_names[ds],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            embedding_dim=500,
            n_titles=len(mapping),
            max_timesteps=max_seq_len,
            class_mapping=mapping,
            num_layers=1,
            n_epochs=800,
            learning_rate=0.01,
            store_model=True,
            restore=True
        )

    if seq_model:
        seq_model.train()

if __name__ == "__main__":
    main()
