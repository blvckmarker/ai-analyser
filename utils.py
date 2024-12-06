from torch.utils.data import Dataset
import numpy as np
from sentence_transformers import util
import pandas as pd
import sqlite3
import zipfile
import os


class QueryDataset(Dataset):
    def __init__(self, queries : pd.DataFrame):
        super().__init__()
        self.queries = queries
        self.__prepare()

    def __len__(self):
        return self.queries.__len__()
    
    def __prepare(self):
        data = []
        for i in range(len(self.queries)):
            row = self.queries[self.queries.index == i]
            data.append({
                'question_ru' : row['question'][i]['ru'],
                'question_en' : row['question'][i]['en'],
                'query_ru' : row['query'][i]['ru'],
                'query_en' : row['query'][i]['en']
            })
        
        self.prepared_data = data

    def __iter__(self):
        return iter(self.prepared_data)

    def __getitem__(self, index):
        return self.prepared_data[index]
    

def find_similar_sentences(sentence_model, target_sentence : str, sentences : list[str], count : int = 3):
    emb_target = sentence_model.encode(target_sentence)

    sims = []
    for i, sentence in enumerate(sentences):
        emb_sentence = sentence_model.encode(sentence)
        sim = util.pytorch_cos_sim(emb_sentence, emb_target)
        sims.append([i, np.float16(sim.squeeze())])

    nearest = sorted(sims, key=lambda pair : pair[1], reverse=True)
    similar_questions = [sentences[pair[0]] for pair in nearest if pair[1] != 1.0][:count]
    return similar_questions

def table_similarity(dataframe1 : pd.DataFrame, dataframe2 : pd.DataFrame, mode : str) -> int:
    if dataframe1.columns.shape != dataframe2.columns.shape:
        return False
    if not (dataframe1.columns == dataframe2.columns).all():
        return False
    
    match mode:
        case 'soft':
            return int(dataframe1.sort_index().equals(dataframe2.sort_index()))
        case 'strict':
            return int(dataframe1.equals(dataframe2))
        case 'flexible':
            hash_1 = set(pd.util.hash_pandas_object(dataframe1, index=False))
            hash_2 = set(pd.util.hash_pandas_object(dataframe2, index=False))
            intersection = hash_1 & hash_2
            union = hash_1 | hash_2

            return len(intersection) / len(union) if len(union) != 0 else 1
        case _:
            raise Exception('Incorrect mode value')
     

def load_table(database_path : str, queries_table_path : str, db_id : str):
    queries = pd.read_json(queries_table_path)
    queries = queries[queries['db_id'] == db_id]
    queries = queries.reset_index(drop=True)
    dataset = QueryDataset(queries)

    sqlite_conn = sqlite3.connect(os.path.join(database_path, f'{db_id}.sqlite'))
    schema = open(os.path.join(database_path, 'schema.sql')).read()

    db = sqlite_conn.cursor()
    try:
        db.executescript(schema)
    except:
        print('Some problem occured during schema execution')
    return sqlite_conn, dataset


def unzip_file(path, path_to):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path_to)