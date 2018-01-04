from pymongo import MongoClient
import pandas as pd
from collections import defaultdict
import numpy as np


def _get_element(doc, field_name):
    if field_name in doc:
        return doc[field_name]
    return np.nan

def make_dataframe(curr):
    data = defaultdict(list)
    for doc in curr:
        data['_id'].append(doc['_id'])
        data['industry'].append(_get_element(doc,'industry'))
        data['job_index'].append(doc['job_index'])
        data['place'].append(_get_element(doc, 'place'))
        data['function'].append(_get_element(doc, 'function'))
        data['start_date'].append(_get_element(doc, 'start_date'))
        data['end_date'].append(_get_element(doc, 'end_date'))

    return data

if __name__ == "__main__":
    client = MongoClient()
    candidates = client.lbj["2016-09-15"]
    pipeline = [
                {"$match": {"$and": [
                    {"language": "en"},
                    {"$nor": [
                        {"experiences": {"$exists": False}},
                        {"experiences": {"$size": 0}},
                        {"experiences": {"$size": 1}}
                    ]}]}},
                {"$unwind": {
                    "path": "$experiences",
                    "includeArrayIndex": "job_index"
                    }
                },
                {"$project": {
                    "industry": 1,
                    "job_index": 1,
                    "place": "$experiences.place",
                    "function": "$experiences.function",
                    "start_date": "$experiences.startDate",
                    "end_date": "$experiences.endDate"
                    #"duration": "$experiences.duration"
                    #"mission": "$experiences.missions"
                    }}
                ]

    agg = candidates.aggregate(pipeline)
    data = make_dataframe(agg)
    df = pd.DataFrame(data)
    print(df[pd.isna(df["function"])]._id.nunique())
    df.to_pickle("/data/rali7/Tmp/solimanz/data/pickles/2016-09-15.pkl")
