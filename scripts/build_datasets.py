from bidict import bidict
from bson.objectid import ObjectId
from collections import Counter, defaultdict
from math import ceil, floor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pprint import pprint
from pyfasttext import FastText
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import json
import errno
import random
import re

def get_top_k_funcs(k, df, df_col):
    """
    @:returns The the sub series of the top-k unique job title counts
    """
    return df[df_col].value_counts()[:k]

def get_valid_ids(df, toplist, df_col):
    """
    @returns the set of user profile IDs that only use job titles within the toplist.
    """
    bad_ids = df[~df[df_col].isin(toplist.index)]["_id"].unique()
    all_ids = df["_id"].unique()
    return  list(set(all_ids) - set(bad_ids))

def get_id_sets(dataset_ids, path):
    """
    splits dataset ids into profile ids that will be used for
    training the models and ids for testing the models
    @:returns tuple (train_ids, test_ids)
    """
    if os.path.exists(os.path.dirname(os.path.join(path, 'train_ids.pkl'))):
        train_ids = load(os.path.join(path, 'train_ids.pkl'), pickle)
        test_ids = load(os.path.join(path, 'test_ids.pkl'), pickle)
    else:
        random.seed(1234)
        train_size = ceil(0.8 * len(dataset_ids))
        random.shuffle(dataset_ids)
        train_ids = dataset_ids[:train_size]
        test_ids = dataset_ids[train_size:]

    return train_ids, test_ids

def dump(path, serializer, obj):
    """
    Saves 'obj' to 'path' using 'serializer' (either pickle or json)
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    mode = "wb" if serializer.__name__ == "pickle" else "w"
    print(f"Writing to {path}")
    with open(path, mode) as f:
        serializer.dump(obj, f)

def load(path, serializer):
    """
    Loads and returns an object from 'path' by using 'serializer' (pickle or json)
    """
    mode = "rb" if serializer.__name__ == "pickle" else "r"
    with open(path, mode) as f:
        dat = serializer.load(f)

    return dat

def get_sequences(df, df_col, title_id, train_ids, test_ids, use_ids=True):
    """
    Returns tuple of (train_sequences, test_sequences) which are both
    lists of lists of job title ids if use_ids == True else: they are lists of job
    title strings

    @:returns tuple of (train_sequences, test_sequences)
    """
    func_series = df.groupby('_id')[df_col].apply(lambda x: list(reversed(list(x))))

    if use_ids:
        train_data = [[title_id[title] for title in func_series[i]] for i in train_ids]
        test_data = [[title_id[title] for title in func_series[i]] for i in test_ids]
    else:
        train_data = [[title for title in func_series[i]] for i in train_ids]
        test_data = [[title for title in func_series[i]] for i in test_ids]

    return train_data, test_data

def _get_bow(funcs):
    """
    Retruns (vocabulary to id mapping, tokenizer_pattrn)
    """
    sw = set(stopwords.words("english"))
    joined = " ".join(funcs)
    pattrn = re.compile(r"[-/,\.\\\s]")
    tokens = re.split(pattrn, joined)
    vocab = set(tokens)

    vocab = vocab - sw
    vocab = vocab - {''}

    vocab_id = {token: i for i, token in enumerate(vocab)}

    return vocab_id, pattrn

def as_bows(df, df_col, title_id, train_ids, test_id, tf_idf=True):
    """
    Represent each word in a job title as a bag-of-words.
    @return: list of bow sequences
    """
    train, test = get_sequences(df, df_col, title_id, train_ids, test_id, use_ids=False)
    vocab_id, pattrn = _get_bow(list(title_id.keys()))
    bow = {}

    for title in title_id.keys():
        tokens = re.split(pattrn, title)
        token_indices = [vocab_id[tok] for tok in tokens if tok in vocab_id]
        bow[title] = sorted(token_indices)

    train = [[bow[title] for title in seq] for seq in train]
    test = [[bow[title] for title in seq] for seq in test]

    return train, test

def get_job_embs(title_id, emb_dim=300, model_path="/data/rali7/Tmp/solimanz/data/wikipedia/wiki.en.bin"):
    """
    Returns a matrix of job title embeddings
    """
    print("Loading fastText model...")
    model = FastText(model_path)
    print("Model successfull loaded :-D")

    embeddings = np.zeros((len(title_id), emb_dim), dtype=np.float32)

    for title in title_id.keys():
        vec = model.get_sentence_vector(title)
        embeddings[title_id[title], :] = vec

    return embeddings

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to help generate datasets')
    parser.add_argument('-ds', '--dataset', required=True, help='method to standardize dataset',
                    choices=['top550', 'reduced7000'])

    parser.add_argument('-r', '--representation', required=True, help='Choose a method to represent textual data',
                        choices=['jobid', 'bow', 'bowst', 'title_emb', 'all'])

    args = parser.parse_args()

    col_options = {
        'top550': 'transformed',
        'reduced7000': 'reduced'
    }
    data_repr = ['jobid', 'bow', 'bowst', 'title_emb']

    ds_path = f"/data/rali7/Tmp/solimanz/data/datasets/{args.dataset}"
    #ds_path = f"/data/rali7/Tmp/solimanz/data/datasets/{args.dataset}/{args.representation}"
    ds_file_name = "title_sequences"

    k = 550 if args.dataset == 'top550' else 7000
    df_col = col_options[args.dataset]
    ds = args.dataset
    df_path = "/data/rali7/Tmp/solimanz/data/pickles/excerpt-2017-02-20_reduced.pkl"

    print("Reading dataframe...")
    df = pd.read_pickle(df_path)
    print("Successfully loaded dataframe... :-D")
    top = get_top_k_funcs(k, df, df_col)

    dataset_ids = get_valid_ids(df, top, df_col)
    # Get the relevent job experiences
    print("Filtering out irrelevent users...")
    df = df[df._id.isin(dataset_ids)]
    print(f"Size of entire dataset: {len(dataset_ids)}")

    print("Mapping job titles to integer ids...")
    job_titles = top.index.values
    title_id = {title: i for i, title in enumerate(job_titles)}
    print(f"Number of unique job titles: {len(title_id)}")

    print("Splitting up dataset (test/train)...")
    train_ids, test_ids = get_id_sets(dataset_ids, ds_path)
    print(f"Size of train: {len(train_ids)}\nSize of test: {len(test_ids)}")

    print("Saving test and train id lists...")
    dump(os.path.join(ds_path, 'train_ids.pkl'), pickle, train_ids)
    dump(os.path.join(ds_path, 'test_ids.pkl'), pickle, test_ids)

    print("Getting job sequences for train and test datasets...")

    if args.representation == 'jobid' or args.representation == 'all':
        print("Getting job title id sequences...")
        train, test = get_sequences(df, df_col, title_id, train_ids, test_ids)

        max_train_seq = max([len(seq) for seq in train])
        max_test_seq = max([len(seq) for seq in test])

        print(f"Maximum length of training sequences : {max_train_seq}"
              f"\nMaximum length of test sequences: {max_test_seq}")

        print("Dumping job title id sequences...")

        data = {
            'title_to_id': title_id,
            'train_data': train,
            'test_data': test,
            'maximum_seq_len': max(max_train_seq, max_test_seq)
        }

        dump(os.path.join(ds_path, 'jobid', 'data.json'), json, data)

    if args.representation == 'bow' or args.representation == 'all':
        print("Getting job title bow sequences...")
        train_bow, test_bow = as_bows(df, df_col, title_id, train_ids, test_ids)

        max_train_seq = max([len(seq) for seq in train_bow])
        max_test_seq = max([len(seq) for seq in test_bow])
        print(f"Maximum length of training sequences : {max_train_seq}"
              f"\nMaximum length of test sequences: {max_test_seq}")
        print("Dumping job title id sequences...")
        data = {
            'title_to_id': title_id,
            'train_data': train_bow,
            'test_data': test_bow,
            'maximum_seq_len': max(max_train_seq, max_test_seq)
        }
        dump(os.path.join(ds_path, 'bow', 'data.json'), json, data)

    if args.representation == 'title_emb' or args.representation == 'all':
        print("Getting embeddings...")
        embeddings = get_job_embs(title_id)

        if not os.path.exists(os.path.dirname(os.path.join(ds_path, "title_emb"))):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        print("Saving embedding matix...")
        np.save(os.path.join(ds_path, "title_emb", "embeddings_small.npy"), embeddings)
        print("All done! :-D")