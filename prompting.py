import pandas as pd
import numpy as np
import sqlalchemy
from utils.general import find_similar_sentences
from utils.dataset import structure_from_connection, tables_from_connection, IterableDataFrame


class PromptBuilder:
    """
    Класс, отвечающий за создание промпта на основе указанных фичей
    """

    def __init__(self):
        self.__prompt = ''
        self.schema_linking = False


    def add_schema_linking(self, table_structure=None):
        """
        Метод, добавляющий режим использования фичи Schema Linking. 
        
        Parameters
        ----------
        table_structure : Any
            Структура таблицы, которая может быть получена с помощью функции `structure_from_connection`
        """

        self.table_structure = table_structure
        self.schema_linking = True
        return self


    def add_few_shot(self, 
                     queries : IterableDataFrame, 
                     target_question : str, 
                     sentence_model, 
                     count : int = 1):
        """
        Метод, отвечающий за добавление фичи Few-Shot в промпт

        Parameters
        ----------

        sentence_model : Any
            Модель, позволяющая векторизовать текст
        target_question : str
            Вопрос, для которого нужно найти похожие по смыслу вопросы
        queries : IterableDataFrame
            Набор вопросов и запросов, среди которых нужно найти ближайшие по смыслу вопросы. Объект должен являться матрицей Nx2
        count : int
        """

        questions = [sample['question'] for sample in queries]

        input_examples = []
        similar = find_similar_sentences(sentence_model, target_question, questions, count)
        for sample in queries:
            curr_qs = sample['question']
            if curr_qs in similar:
                input_examples.append([curr_qs, sample['query']])

        few_shot_template = ''
        for ex in input_examples:
            few_shot_template += f'Q: {ex[0]}\n'
            few_shot_template += f'A: {ex[1]}\n'

        self.__prompt += few_shot_template + '\n'
        return self
    

    def add_schema_template(self, db_conn : sqlalchemy.Connection):
        """
        Метод, отвечающий за добавление фичи Schema Template в промпт

        Parameters
        ----------
        db_conn : sqlalchemy.Connection
            Соединение с базой данных
        """

        if self.schema_linking:
            structure = self.table_structure
        else:
            structure = structure_from_connection(db_conn)

        schema_template = ''
        for table in structure:
            schema_template += f"{table['table_name']}({', '.join(table['columns'])});\n"

        self.__prompt += schema_template + '\n'
        return self


    def add_cell_value_referencing(self, db_conn : sqlalchemy.Connection, count=1):
        """
        Метод, отвечающий за добавление фичи Cell Value Referencing в промпт

        Parameters
        ----------
        db_conn : sqlalchemy.Connection
            Соединение с базой данных
        count : int
            Ожидаемое количество примеров для добавления. По умолчанию равно 1
        """

        if self.schema_linking:
            tables = [table['table_name'] for table in self.table_structure]
        else:
            tables = tables_from_connection(db_conn)

        data_information = []
        for table in tables:
            if self.schema_linking:
                instance = [bucket for bucket in self.table_structure if bucket['table_name'] == table][0]
                pd_table = pd.read_sql(f'SELECT * FROM {table}', db_conn)[instance['columns']]
            else:
                pd_table = pd.read_sql(f'SELECT * FROM {table}', db_conn)
            
            indexes = np.random.randint(0, pd_table.shape[0], size=count)
            series = [pd_table[pd_table.index == idx].to_numpy() for idx in indexes]

            data_information.append({
                'table_name' : table,
                'examples' : [f"[{', '.join(map(str,list(ser.reshape(ser.shape[1]))))}]" for ser in series]
            })

        value_template = ''
        for data in data_information:
            value_template += f"{data['table_name']}({', '.join(data['examples'])});\n"

        self.__prompt += value_template + '\n'
        return self


    def add_message(self, message : str):
        self.__prompt += message + '\n'
        return self


    def build_prompt(self):
        return self.__prompt