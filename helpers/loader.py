import json
from bidict import bidict


def load_data(data_path):
    """

    :param data_path: full path to the json serialized file
    :return: tuple:
        - title_to_id: bidict mapping
        - train_data: list -> sequences of title ids
        - test_dat: list -> sequences of title ids

    """

    print("Loading mapping and data")
    with open(data_path) as data_file:
        data = json.load(data_file)
    title_to_id = bidict(data['title_to_id'])
    train_data, test_data = data['train_data'], data['test_data']

    if "maximum_seq_length" in data:
        max_seq_len = data["maximum_seq_length"]
    else:
        max_seq_len = 33

    return title_to_id, train_data, test_data, max_seq_len
