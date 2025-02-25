from torch.utils.data import Dataset
import pandas as pd
import sqlite3
from sqlalchemy import text
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


def tables_from_connection(conn : sqlite3.Connection):
    master = pd.DataFrame(conn.execute(text('SELECT * FROM sqlite_master')).fetchall())
    tables = list(master[master['type'] == 'table']['name'])
    return tables


def structure_from_connection(conn : sqlite3.Connection):
    tables = tables_from_connection(conn)
    structure = []
    for table in tables:
        columns = list(pd.DataFrame(conn.execute(text(f'SELECT * FROM {table}')).fetchall()).columns)[1:]
        structure.append(
            {
                'table_name' : table,
                'columns' : columns
            })
        
    return structure


def prepare_column_names(conn : sqlite3.Connection):
    structure = structure_from_connection(conn)
    for table in structure:
        for column in table['columns']:
            if ' ' in column:
                new_name = ''.join([char for char in column if str.isalnum(char)])
                conn.execute(text(
                    f'''ALTER TABLE {table['table_name']} RENAME COLUMN "{column}" TO {new_name}'''
                ))

    return True