from dataclasses import dataclass
from abc import ABC
import numpy as np
from sentence_transformers import util
import pandas as pd
import zipfile
from sqlglot import exp
import sqlglot.optimizer
import re
from spans import *
from pandas.testing import assert_frame_equal, assert_series_equal

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


def find_sql(text : str, start_keyword='SELECT'):
    """
    Функция, которая ищет в строке `text` первое вхождение самого длинного, правильного SQL запроса
    """

    matches = re.search(f'({start_keyword}).*', text, flags=re.IGNORECASE)
    if not matches:
        return ''

    begin_sql = matches.group()
    splitted = begin_sql.split()

    maybe_sql = ''
    last_success_pos = 0
    for i, word in enumerate(splitted):
        maybe_sql += f' {word}'
        try:
            sqlglot.transpile(maybe_sql)
            last_success_pos = i
        except:
            pass

    found_sql = ' '.join(splitted[:last_success_pos + 1])
    return found_sql



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


# def parse_literals(sql : str, table_structure : list[dict]):
#     """
#     Функция, вытягивающая все названия таблиц и столбцов, которые упомянуты в запросе `sql`

#     Parameters
#     ----------
#     sql : str
#         SQL запрос
#     table_structure : List[dict]
#         Структура таблицы, которая может быть получена при помощи функции `structure_from_connection`
#     """

#     root = sqlparse.parse(sql)[0]
#     names = []

#     def __get_all_names_helper(node : sqlparse.sql.Token):
#         if issubclass(type(node), sqlparse.sql.TokenList):
#             for token in node.tokens:
#                 __get_all_names_helper(token)
#         elif node.ttype != sqlparse.sql.T.Punctuation and node.ttype != sqlparse.sql.T.Whitespace:
#             names.append(node.value)

#     __get_all_names_helper(root)
    
#     tables = set([table['table_name'] for table in table_structure])
#     visited_tables = set([])
#     buckets = []

#     # В этом говне не рекомендую особо купаться. Хотя блочная схема для алгоритма вполне примитивна
#     for name in names:
#         if name in tables and name not in visited_tables:
#             buckets.append({
#                 'table_name' : name,
#                 'columns' : []
#             })
#             visited_tables.add(name)
#         elif name not in tables:
#             for table in table_structure:
#                 if name in table['columns']:
#                     if table['table_name'] not in visited_tables:
#                         buckets.append({
#                             'table_name' : table['table_name'],
#                             'columns' : [name]
#                         })
#                         visited_tables.add(table['table_name'])
#                     else:
#                         instance = [bucket for bucket in buckets if bucket['table_name'] == table['table_name']][0]
#                         if name not in instance['columns']:
#                             instance['columns'].append(name)


#     if '*' in sql:
#         for bucket in buckets:
#             columns_instance = [table['columns'] for table in table_structure if table['table_name'] == bucket['table_name']][0]
#             bucket['columns'] = columns_instance
        
    
#     return buckets


def schema_parse(sql : str, structure_dict : dict):
    """
    Функция, вытягивающая все названия таблиц и столбцов, которые упомянуты в запросе `sql`

    Parameters
    ----------
    sql : str
        SQL запрос
    table_structure : List[dict]
        Структура таблицы, которая может быть получена при помощи функции `structure_from_connection`
    """

    optimized_sql = sqlglot.optimizer.optimize(
        sqlglot.parse_one(sql),
        schema=structure_dict
    )

    buckets = {table.name : set([]) for table in optimized_sql.find_all(exp.Table)}
    for column in optimized_sql.find_all(exp.Column):
        table_of_col = column.table
        buckets[table_of_col].add(column.name)

    as_default = []
    for k, v in buckets.items():
        as_default.append({'table_name' : k, 'columns' : list(v)})

    return as_default


class ExcelIO(object):
    @staticmethod
    def write_spans(spans : list[Span], path : str):
        asdict = [span.__dict__ for span in spans]
        df = pd.DataFrame(asdict)
        df.to_excel(excel_writer=path, index=False)

    @staticmethod
    def read_excel(path : str):
        df = pd.read_excel(path)
        return df

def normalize_table(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. sorting columns in alphabetical order
    2. sorting rows using values from first column to last
    3. resetting index
    """
    # sort columns in alphabetical order
    sorted_df = df.reindex(sorted(df.columns), axis=1)
    # sort rows using values from first column to last
    sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))
    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df

def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    verbose: bool = False,
) -> bool:
    """
    Checks if df_sub is a subset of df_super
    """
    if df_sub.empty:
        return True  # trivial case
    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )
            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue
        if col_match == False:
            if verbose:
                print(f"no match for {col_sub_name}")
            return False
    df_sub_normalized = normalize_table(df_sub)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched)

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False
