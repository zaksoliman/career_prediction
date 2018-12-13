import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson import ObjectId
from collections import defaultdict, Counter
import os

client = MongoClient()
collection = client.LBJ.candidats

def _get_element(doc, field_name):
    if field_name in doc:
        return doc[field_name]
    return np.nan

def make_dataframe(curr):
    data = defaultdict(list)

    for doc in curr:
        data['_id'].append(str(doc['_id']))
        data['function'].append(_get_element(doc, 'function'))
        data['company_name'].append(_get_element(doc,'company_name'))
        data['industry'].append(_get_element(doc,'industry'))
        data['job_index'].append(doc['job_index'])
        data['place'].append(_get_element(doc, 'place'))
        data['language'].append(_get_element(doc, 'language'))
        #data['mission'].append(_get_element(doc, 'mission'))
        data['start_date'].append(_get_element(doc, 'start_date'))
        data['end_date'].append(_get_element(doc, 'end_date'))
    return data

if __name__ == '__main__':

    pipeline = [
            {"$match": {'experiences': {"$exists": True}} },
            {"$unwind": {
                "path": "$experiences",
                "includeArrayIndex": "job_index"
                }
            },
            {"$project": {
                "industry": 1,
                "job_index": 1,
                "language": "$language",
                "company_name": "$experiences.companyName",
                "place": "$experiences.place",
                "function": "$experiences.function",
                "start_date": "$experiences.startDate",
                "end_date": "$experiences.endDate"
                #"mission": "$experiences.missions"
                }}
            ]

    agg = collection.aggregate(pipeline)
    print("Building dataframe...")
    df_dict = make_dataframe(agg)
    df = pd.DataFrame(df_dict)
    print("Saving dataframe...")
    df.to_pickle('/data/rali7/Tmp/solimanz/LBJ/dataframe.pkl')
    print("Done :)")
