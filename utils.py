from torch.utils.data import Dataset
import numpy as np
from sentence_transformers import util
import pandas as pd
import sqlite3
import zipfile
import os


class QueryDataset(Dataset):
    """
    A class that simplifies working with query dataset by translating the original dataframe into a list of dictionaries.

    It is closely related to the structure of the pauq dataset
    """
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
    """
    The algorithm for searching for ``count`` sentences from ``sentences``, semantically similar to ``target sentence`'.

    Parameters
    ----------
    sentence_model : Aní
    Model that has an interface for vectorizing input tokens (sentences)

    target_sentence : structure
        A sentence for which looking similar sentences
    sentences : list[str]
        The body (list, dataset) of sentences
    count : int = 3
        The number of sentences that are most similar in meaning that need to be found
    """
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
    """
    The function of comparing two dataframes

    Three modes are available: ```soft, strict, flexible```. 
    
    In the ```soft``` mode, two tables are equivalent, if they contain the same data in any order. 
    
    The ```strict``` mode have condition of orderliness.

    The `flexible` mode is the ratio of the intersection of two tables to their union (IoU metrics)
    """
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
    """
    Loading tables from the pauq dataset

    Parameter
    ----------
    database_path : str
        The path to a specific database in the pauq dataset 
        (for example, the folder ./pub/academic, which stores two files with the extensions .sqlite and .sql)

    queries_table_path : str
        Path to dataset with queries 
    db_id : str
        Name of the specific database contained in the `queries table`. For example, db_id = academic
    """
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