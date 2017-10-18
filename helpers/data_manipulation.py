import pandas as pd
import json
import pickle


def build_job_title_sequences():
    print('Reading test and train ids')
    with open('/data/rali7/Tmp/solimanz/data/pickles/train_ids.pkl', 'rb') as f:
        train_ids = pickle.load(f)

    with open('/data/rali7/Tmp/solimanz/data/pickles/test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)

    print('reading dataframe')
    data = pd.read_pickle('/data/rali7/Tmp/solimanz/data/pickles/top_data.pkl')

    print('Building mapping between job title name and a job title id')
    job_titles = data.function.unique()
    title_id = {title: i for i, title in enumerate(job_titles)}

    print('Getting list of job titles for every profile id')
    func_series = data.groupby('_id')['function'].apply(list)

    print('Building training data list')
    train_data = [list(map(lambda title: title_id[title], func_series[i])) for i in train_ids]

    print('Build test data')
    test_data = [list(map(lambda title: title_id[title], func_series[i])) for i in test_ids]

    print('dumping json...')
    data = {
        'title_to_id': title_id,
        'train_data': train_data,
        'test_data': test_data
    }

    with open('/data/rali7/Tmp/solimanz/data/datasets/title_seq.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    build_job_title_sequences()