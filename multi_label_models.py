#from prediction_models.multi_label_fasttext import Model
from prediction_models.multi_label_cnn_lstm import Model
from helpers.loader import load_data
import sys

def main():

    seq_model = None
    ds = sys.argv[1]
    #model = sys.argv[1]

    model_names = {
        'reduced7k': 'reduced7k_multilabel',
        'top550': 'top550_mutlilabel'
    }
    dataset_names = {
        'reduced7k': 'data2',
        'top550': 'data2'
    }
    embedding_paths = {
        'reduced7k': "/data/rali7/Tmp/solimanz/data/datasets/reduced7000/fasttext/embeddings.npy",
        'top550': "/data/rali7/Tmp/solimanz/data/datasets/top550/fasttext/embeddings.npy"
    }

    path = f"/data/rali7/Tmp/solimanz/data/datasets/multilabel/{ds}/{dataset_names[ds]}.json"

    title_id, train_inputs, train_targets, test_inputs, test_targets, max_seq_len, n_labels, n_skills, max_skills, skill_embs_path = load_data(path, multi=True)

    print(len(title_id))

    train_skills = [d[2] for d in train_inputs]
    test_skills = [d[2] for d in test_inputs]

    train_inputs = [d[1] for d in train_inputs]
    test_inputs = [d[1] for d in test_inputs]

    train_targets = [d[1] for d in train_targets]
    test_targets = [d[1] for d in test_targets]

    emb_path = embedding_paths[ds]
    emb_dim = 300
    n_layer = 1
    n_epochs = 1000
    lr = 0.001
    batch_size = 1000

    seq_model = Model(
        name=model_names[ds],
        train_inputs=train_inputs,
        train_targets=train_targets,
        train_skills=train_skills,
        test_inputs=test_inputs,
        test_targets=test_targets,
        test_skills=test_skills,
        embedding_dim=emb_dim,
        emb_path=emb_path,
        batch_size=batch_size,
        freeze_emb=False,
        use_dropout=True,
        n_labels=n_labels,
        n_skills=n_skills,
        max_skills=max_skills,
        n_titles=len(title_id),
        max_timesteps=max_seq_len,
        class_mapping=title_id,
        num_layers=n_layer,
        n_epochs=n_epochs,
        learning_rate=lr,
        skill_emb_path=skill_embs_path,
        store_model=True,
        restore=True
    )

    if seq_model:
        seq_model.test()

if __name__ == "__main__":
    main()
