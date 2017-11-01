from prediction_models.title_seq_rnn import Model
from helpers.loader import load_data

def main():
    path = "/data/rali7/Tmp/solimanz/data/dataset/title_seq.json"
    mapping, train_data, test_data = load_data()
    seq_model = Model(train_data=train_data)
    seq_model.train()

if __name__ == "__main__":
    main()