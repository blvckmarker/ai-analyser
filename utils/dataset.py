import pandas as pd
from sqlalchemy import text, Connection, inspect


class IterableDataFrame:
    """
    Класс, позволяющий итерироваться в таблице типа `pd.DataFrame`
    """

    def __init__(self, df : pd.DataFrame):
        self.df = df
        self.__series = {}
        for idx in self.df.index:
            sample = {
                column : self.df[self.df.index == idx][column][idx] for column in self.df.keys()
            }
            self.__series[idx] = sample

    def __len__(self):
        return self.df.shape[0]

    def __as_list(self):
        return list(self.__series.values())
    
    def __iter__(self):
        return iter(self.__as_list())

    def __getitem__(self, index):
        return self.__as_list()[index]
    
    def at_index(self, index):
        return self.__series[index]


def tables_from_connection(conn : Connection):
    """
    Функция, возвращающая список названий всех таблиц для данного соединения `conn`

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Соединение с базой данных
    """

    master = pd.DataFrame(conn.execute(text('SELECT * FROM sqlite_master')).fetchall())
    tables = list(master[master['type'] == 'table']['name'])
    return tables


def structure_from_connection(conn : Connection):
    """
    Функция, возвращающая список словарей вида {table_name, columns}, где table_name - str, а columns - List[str]

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Соединение с базой данных
    """

    tables = tables_from_connection(conn)
    structure = []
    for table in tables:
        columns = pd.DataFrame(conn.execute(text(f'SELECT * FROM "{table}"')).fetchall()).columns.to_list()
        structure.append(
            {
                'table_name' : table,
                'columns' : columns
            })
        
    return structure


def structure_from_connection_dict(conn : Connection):
    """
    Функция, возвращающая словарь словарей вида {"Table" : {"Col" : "INT", ...}}

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Соединение с базой данных
    """

    tables = tables_from_connection(conn)
    structure = {}
    for table in tables:
        columns = inspect(conn).get_columns(table)
        columns_meta = {column['name'] : column['type'] for column in columns}
        structure[table] = columns_meta

    return structure


def prepare_column_names(conn : Connection):
    """
    Функция, обрабатывающая базу данных из соединения `conn`. Функция переименовывает названия всех таблиц и их столбцов, 
    которые содержат whitespace и punctuation символы. Возвращает True, если переименовывание прошло успешно

    Parameters
    ----------
    conn : sqlalchemy.Connection
        Соединение с базой данных
    """
    
    structure = structure_from_connection(conn)
    for table in structure:
        for column in table['columns']:
            new_name = str.lower(''.join([char for char in column if str.isalnum(char)]))
            if new_name != column:
                conn.execute(text(
                    f'''ALTER TABLE "{table['table_name']}" RENAME COLUMN "{column}" TO "{new_name}"'''
                ))

        new_table_name = str.lower(''.join([char for char in table['table_name'] if str.isalnum(char)]))
        if new_table_name != table['table_name']:
            conn.execute(text(f'''ALTER TABLE "{table['table_name']}" RENAME TO "{new_table_name}"'''))

    return True