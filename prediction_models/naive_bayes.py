import numpy as np
import re
from bidict import bidict
import json
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

def get_subsequences(sequences):
    sub_seqs = []
    for seq in sequences:
        for i in range(2,len(seq)):
            sub_seqs.append(seq[:i])
    return sub_seqs

def apply_stemming(data):
    """data is a list of sequences"""

    stemmer = SnowballStemmer('english')
    pattrn = re.compile(r"[-/,\.\\\s_]")

    output = []

    for sequence in data:
        stemmed_seq = []
        for title in sequence:
            stemmed = ""
            for token in re.split(pattrn, title):
                if stemmed:
                    stemmed += " " + stemmer.stem(token)
                else:
                    stemmed += stemmer.stem(token)
            stemmed_seq.append(stemmed)

        output.append(stemmed_seq)

    return [" ".join(seq) for seq in output]


def get_data_sets(json_path, tokenize_titles=True, use_stemmer=True, use_sub_seq=False):
    with open(json_path, 'r') as f:
        data = json.load(f)

    title_id = bidict(data["title_to_id"])
    train_data = data["train_data"]
    test_data = data["test_data"]

    if use_sub_seq:
        train_data = get_subsequences(train_data)
        # test_data = get_subsequences(test_data)

    if tokenize_titles:
        train_seq = [[title_id.inv[i] for i in title_seq[1][:-1]] for title_seq in train_data]
        test_seq = [[title_id.inv[i] for i in title_seq[1][:-1]] for title_seq in test_data]
    else:
        # replace whitespace by underscore to prevent breaking up multi word job titles
        train_seq = [[str(i) for i in title_seq[1][:-1]] for title_seq in train_data]
        test_seq = [[str(i) for i in title_seq[1][:-1]] for title_seq in test_data]

    train_targets = [seq[1][-1] for seq in train_data]
    test_targets = [seq[1][-1] for seq in test_data]

    if tokenize_titles and not use_stemmer:
        train_text = [" ".join(title_seq).replace("_", " ") for title_seq in train_seq]
        test_text = [" ".join(title_seq).replace("_", " ") for title_seq in test_seq]
    elif tokenize_titles and use_stemmer:
        train_text = apply_stemming(train_seq)
        test_text = apply_stemming(test_seq)
    else:
        train_text = [" ".join(title_seq) for title_seq in train_seq]
        test_text = [" ".join(title_seq) for title_seq in test_seq]

    return train_text, train_targets, test_text, test_targets


def run_model(model, X_train, train_targets, X_test, test_targets):
    # Train
    print(f"Training Naive Bayes...")
    model.fit(X_train, train_targets)

    # Test
    print("Running trained model on test dataset")
    predicted = model.predict(X_test)
    acc = np.mean(predicted == test_targets)

    print("Model Accuracy: " + str(acc))

    return model


def run_experiment(data_path, tokenized_titles=False, use_stemmer=False, tf_idf=False, sub_seq=False):
    # Fetch and preprocess data
    train, train_targets, test, test_targets = get_data_sets(data_path, tokenized_titles, use_stemmer, sub_seq)

    # Construct Models
    multi_nb = MultinomialNB()
    nb = BernoulliNB()

    # Construct vectorizer
    if tokenized_titles:
        sw = stopwords.words('english')
        vect = CountVectorizer(stop_words=sw)
        vect = vect.fit(train + test)
    else:
        vect = CountVectorizer(token_pattern=r"\b\d+\b")
        vect = vect.fit(train + test)

    print(f"Vocab Size: {len(vect.vocabulary_)}")

    # Construct document matices
    X_train = vect.transform(train)
    X_test = vect.transform(test)

    # Run Models
    multi_nb = run_model(multi_nb, X_train, train_targets, X_test, test_targets)
    nb = run_model(nb, X_train, train_targets, X_test, test_targets)

    return multi_nb, nb, X_train, X_test, train_targets, test_targets

def get_preds(model, X, targets):
    preds = model.predict_proba(X, targets)
    return preds

def top_k_acc(preds, targets, k=1):
    sorted_args = (-preds).argsort(axis=1)[:,:k]
    tt = np.tile(targets, (k,1)).T
    acc = np.mean(np.sum(sorted_args == tt, axis=1))
    return acc

def print_top_k_accs(model, X_test, targets):
    print(f"acc: {top_k_acc(model, X_test, targets, k=1)[0]*100:.2f}")
    print(f"top 2: {top_k_acc(model, X_test, targets, k=2)[0]*100:.2f}")
    print(f"top 3: {top_k_acc(model, X_test, targets, k=3)[0]*100:.2f}")
    print(f"top 4: {top_k_acc(model, X_test, targets, k=4)[0]*100:.2f}")
    print(f"top 5: {top_k_acc(model, X_test, targets, k=5)[0]*100:.2f}")

if __name__ == "__main__":

    reduced7000_path = "/data/rali7/Tmp/solimanz/data/datasets/reduced7000/jobid/data.json"
    top550_path = "/data/rali7/Tmp/solimanz/data/datasets/top550/jobid/data.json"

    accs = {
        'dataset': [],
        'top_k': [],
        'model': [],
        'accuracy': []
    }

    def add_to_dict(accs, preds, targets, dataset, model):
        for k in range(1, 11):
            accs['dataset'].append(dataset)
            accs['top_k'].append(k)
            accs['model'].append(model)
            accs['accuracy'].append(top_k_acc(preds, targets, k))

    # multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(top550_path,
    #                                                                             tokenized_titles=True,
    #                                                                             use_stemmer=False,
    #                                                                             sub_seq=False)
    #
    # add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, '550_titles', "mult_nb_bow_no_stem")
    # add_to_dict(accs, nb.predict_proba(X_test), test_targets, '550_titles', "bern_nb_bow_no_stem")
    #
    # multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(top550_path,
    #                                                                             tokenized_titles=True,
    #                                                                             use_stemmer=True,
    #                                                                            sub_seq=False)
    #
    # add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, '550_titles', "mult_nb_bow_stem")
    # add_to_dict(accs, nb.predict_proba(X_test), test_targets, '550_titles', "bern_nb_bow_stem")
    #
    #
    # multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(top550_path,
    #                                                                             tokenized_titles=False,
    #                                                                             use_stemmer=False,
    #                                                                             sub_seq=False)
    #
    # add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, '550_titles', "mult_nb_titles")
    # add_to_dict(accs, nb.predict_proba(X_test), test_targets, '550_titles', "bern_nb_titles")


    multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(reduced7000_path,
                                                                                tokenized_titles=True,
                                                                                use_stemmer=False,
                                                                                sub_seq=False)
    add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, 'reduced7k', "mult_nb_bow_no_stem")
    add_to_dict(accs, nb.predict_proba(X_test), test_targets, 'reduced7k', "bern_nb_bow_no_stem")

    multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(reduced7000_path,
                                                                                tokenized_titles=True,
                                                                                use_stemmer=True,
                                                                                sub_seq=False)
    add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, 'reduced7k', "mult_nb_bow_stem")
    add_to_dict(accs, nb.predict_proba(X_test), test_targets, 'reduced7k', "bern_nb_bow_stem")

    multi_nb, nb, X_train, X_test, train_targets, test_targets = run_experiment(reduced7000_path,
                                                                                tokenized_titles=False,
                                                                                use_stemmer=False,
                                                                                sub_seq=False)
    add_to_dict(accs, multi_nb.predict_proba(X_test), test_targets, 'reduced7k', "mult_nb_titles")
    add_to_dict(accs, nb.predict_proba(X_test), test_targets, 'reduced7k', "bern_nb_titles")
