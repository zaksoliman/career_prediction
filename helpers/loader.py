import json
import pickle
from bidict import bidict


def load_data(data_path, bow=False, multi=False):
    """

    :param data_path: full path to the json serialized file
    :return: tuple:
        - title_to_id: bidict mapping
        - train_data: list -> sequences of title ids
        - test_dat: list -> sequences of title ids

    """

# """
# data = {
#         'title_to_id': dict(title_id),
#         'token_id': token_id,
#         'train_inputs': train_inputs,
#         'test_inputs': test_inputs,
#         'train_targets': train_targets,
#         'test_targets': test_targets,
#         'maximum_seq_len': max(max_train_seq, max_test_seq),
#         'emb_dim': emb_dim
#
#     }
#     """
    print("Loading mapping and data")
    print(f"Reading {data_path}")

    if multi:
        with open(data_path, 'r') as data_file:
            data = json.load(data_file)
    else:
        with open(data_path) as data_file:
            data = json.load(data_file)

    title_to_id = bidict(data['title_to_id'])

    if "maximum_seq_len" in data:
        max_seq_len = data["maximum_seq_len"]
    else:
        max_seq_len = 33

    if bow:
        train_inputs = data["train_data"]
        train_targets = data["train_targets"]
        test_inputs = data["test_data"]
        test_targets = data["test_targets"]
        vocab_size = len(data["vocab"])
        return  title_to_id, train_inputs, train_targets, test_inputs, test_targets, max_seq_len, vocab_size
    if multi:
        train_inputs = data["train_data"]
        train_targets = data["train_targets"]
        test_inputs = data["test_data"]
        test_targets = data["test_targets"]
        n_labels = data['n_labels']
        return title_to_id, train_inputs, train_targets, test_inputs, test_targets, max_seq_len, n_labels
    else:
        train_data, test_data = data['train_data'], data['test_data']
        return title_to_id, train_data, test_data, max_seq_len

def load_dat_skills(data_path):

    with open(data_path, 'r') as data_file:
        data =  json.load(data_file)

    #'title_to_id', 'train_data', 'test_data', 'maximum_seq_len', 'skill_token_id', 'max_skills_num', 'skills_embs'

    title_id = data["title_to_id"]
    train = data["train_data"]
    test = data["test_data"]
    max_seq_len = data["maximum_seq_len"]
    token_id = data["skill_token_id"]
    max_skills = data["max_skills_num"]
    skill_embs_path = data["skills_embs"]

    return title_id, train, test, max_seq_len, token_id, max_skills, skill_embs_path