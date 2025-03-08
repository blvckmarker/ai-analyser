import numpy as np
from sentence_transformers import util
import pandas as pd
import zipfile
import sqlparse


def find_similar_sentences(sentence_model, target_sentence : str, sentences : list[str], count : int = 3):
    """
    Функция поиска похожих по смыслу предложений из набора `sentences` для указанного предложения `target_sentence`

    Parameters
    ----------
    sentence_model : Any
        Модель, позволяющая векторизовать текст
    target_sentence: str
        Предложение, для которого нужно найти похожие по смыслу предложения
    sentences : List[str]
        Набор предложений
    count : int
        Количество ожидаемых предложений
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
    Функция сравнения двух таблиц

    Parameters
    ----------
    dataframe1 : pd.DataFrame
        Первая таблица
    dataframe2 : pd.DataFrame
        Вторая таблица
    mode : str
        Режим сравнения. Допустимы режимы soft, strict, flexible
    """

    # if dataframe1.columns.shape != dataframe2.columns.shape:
    #     return False
    # if not (dataframe1.columns == dataframe2.columns).all():
    #     return False
    
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
     

def unzip_file(path, path_to):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path_to)


def parse_literals(sql : str, table_structure : list[dict]):
    """
    Функция, вытягивающая все названия таблиц и столбцов, которые упомянуты в запросе `sql`

    Parameters
    ----------
    sql : str
        SQL запрос
    table_structure : List[dict]
        Структура таблицы, которая может быть получена при помощи функции `structure_from_connection`
    """

    root = sqlparse.parse(sql)[0]
    names = []

    def __get_all_names_helper(node : sqlparse.sql.Token):
        if issubclass(type(node), sqlparse.sql.TokenList):
            for token in node.tokens:
                __get_all_names_helper(token)
        elif node.ttype != sqlparse.sql.T.Punctuation and node.ttype != sqlparse.sql.T.Whitespace:
            names.append(node.value)

    __get_all_names_helper(root)
    
    tables = set([table['table_name'] for table in table_structure])
    visited_tables = set([])
    buckets = []

    # В этом говне не рекомендую особо купаться. Хотя блочная схема для алгоритма вполне примитивна
    for name in names:
        if name in tables and name not in visited_tables:
            buckets.append({
                'table_name' : name,
                'columns' : []
            })
            visited_tables.add(name)
        elif name not in tables:
            for table in table_structure:
                if name in table['columns']:
                    if table['table_name'] not in visited_tables:
                        buckets.append({
                            'table_name' : table['table_name'],
                            'columns' : [name]
                        })
                        visited_tables.add(table['table_name'])
                    else:
                        instance = [bucket for bucket in buckets if bucket['table_name'] == table['table_name']][0]
                        if name not in instance['columns']:
                            instance['columns'].append(name)
    
    return buckets