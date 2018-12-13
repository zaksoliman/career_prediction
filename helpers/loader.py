import json
import pickle
from bidict import bidict


def load_data(data_path):
    """

    :param data_path: full path to the json serialized file
    :return: tuple:
        - title_to_id: bidict mapping
        - train_data: list -> sequences of title ids
        - test_dat: list -> sequences of title ids

    dict_keys(['sequences', 'maximum_seq_len', 'title_id', 'n_labels', 'max_n_skills', 'emb_path', 'skill_id'])
    """

    print("Loading mapping and data")
    print(f"Reading {data_path}")


    with open(data_path) as data_file:
        data = json.load(data_file)

    title_to_id = bidict(data["title_id"])
    max_seq_len = data["maximum_seq_len"]
    n_labels = data["n_labels"]
    sequences = data["sequences"]
    token_id = data["skill_id"]
    max_skills = data["max_n_skills"]
    skill_embs_path = data["emb_path"]

    return title_to_id, sequences, max_seq_len, token_id, max_skills, skill_embs_path

def load_dat_skills(data_path):

    with open(data_path, 'r') as data_file:
        data =  json.load(data_file)

    title_id = data["title_to_id"]
    train = data["train_data"]
    test = data["test_data"]
    max_seq_len = data["maximum_seq_len"]
    token_id = data["skill_token_id"]
    max_skills = data["max_skills_num"]
    skill_embs_path = data["skills_embs"]

    return title_id, train, test, max_seq_len, token_id, max_skills, skill_embs_path