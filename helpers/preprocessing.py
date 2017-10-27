import pandas as pd
import os
import json
import pickle


def preprocess_job_title_sequences(data_path):

    print('Reading test and train ids...')
    with open(os.path.join(data_path, "train_ids.pkl"), "rb") as f:
        train_ids = pickle.load(f)
    with open(os.path.join(data_path, "test_ids.pkl"), 'rb') as f:
        test_ids = pickle.load(f)

    print('Loading dataframe...')
    data = pd.read_pickle(os.path.join(data_path, "top_data.pkl"))

    print('Building mapping between job title name and a job title id...')
    job_titles = data.function.unique()
    title_id = {title: i + 1 for i, title in enumerate(job_titles)}

    print('Getting list of job titles for every profile id')
    func_series = data.groupby('_id')['function'].apply(lambda x: list(reversed(list(x))))

    print('Building training data list...')
    train_data = [[title_id[title] for title in func_series[i]] for i in train_ids]
    print('Build test data...')
    test_data = [[title_id[title] for title in func_series[i]] for i in test_ids]

    print('dumping json...')
    data = {
        'title_to_id': title_id,
        'train_data': train_data,
        'test_data': test_data
    }

    with open(os.path.join(data_path, "title_seq.json"), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    data_path = "../../data/datasets/"
    preprocess_job_title_sequences(data_path)