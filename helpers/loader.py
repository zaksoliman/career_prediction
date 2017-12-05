import json
from bidict import bidict


def load_data(data_path, bow=False):
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

    if "maximum_seq_len" in data:
        max_seq_len = data["maximum_seq_len"]
    else:
        max_seq_len = 33

    if bow:
        train_inputs = data["train_inputs"]
        train_targets = data["train_targets"]
        test_inputs = data["test_inputs"]
        test_targets = data["test_targets"]
        vocab_id = data["vocab_id"]
        return  title_to_id, train_inputs, train_targets, test_inputs, test_targets, vocab_id, max_seq_len
    else:
        train_data, test_data = data['train_data'], data['test_data']
        return title_to_id, train_data, test_data, max_seq_len
