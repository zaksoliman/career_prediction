from prediction_models.title_seq_rnn import Model
from helpers.loader import load_data

def main():
    path = "/data/rali7/Tmp/solimanz/data/datasets/1/title_sequences.json"
    mapping, train_data, test_data = load_data(path)
    seq_model = Model(train_data=train_data, test_data=test_data, class_mapping=mapping, num_layers=2)
    seq_model.train()

if __name__ == "__main__":
    main()
