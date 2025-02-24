from utils import find_similar_sentences
import pandas as pd
import numpy as np


class PromptBuilder:
    def __init__(self, question):
        self.__prompt = ''
        self.__question = question
        self.__few_shot = None
        self.__schema_template = None
        self.__cell_value_referencing = None


    def add_few_shot(self, sentence_model, target_question, queries):
        questions = [sample['question_ru'] for sample in queries]

        input_examples = []
        similar = find_similar_sentences(sentence_model, target_question, questions, count=3)
        for sample in queries:
            curr_qs = sample['question_ru']
            if curr_qs in similar:
                input_examples.append([curr_qs, sample['query_ru']])

        few_shot_template = ''
        for ex in input_examples:
            few_shot_template += f'Q: {ex[0]}\n'
            few_shot_template += f'A: {ex[1]}\n'

        self.__few_shot = few_shot_template
        return self

    def add_schema_template(self, db_conn):
        sql_master = pd.read_sql('SELECT * FROM sqlite_master', db_conn)
        tables = sql_master[sql_master['type'] == 'table']['name']
        hier = []
        for table in tables:
            temp = pd.read_sql(f'SELECT * FROM {table}', db_conn)
            hier.append({
                'table_name' : table,
                'struct' : list(temp)
                })
        
        schema_template = ''
        for table in hier:
            schema_template += f"{table['table_name']}({', '.join(table['struct'])});\n"
        self.__schema_template = schema_template
        return self

    def add_cell_value_referencing(self, db_conn, count=1):
        sql_master = pd.read_sql('SELECT * FROM sqlite_master', db_conn)
        tables = sql_master[sql_master['type'] == 'table']['name']

        data_information = []
        for table in tables:
            pd_table = pd.read_sql(f'SELECT * FROM {table}', db_conn)
            indexes = np.random.randint(0, pd_table.shape[0], size=count)
            series = [pd_table[pd_table.index == idx] for idx in indexes]

            data_information.append({
                'table_name' : table,
                'examples' : [f"[{', '.join(map(str,list(ser.values.squeeze())))}]" for ser in series]
            })

        value_template = ''
        for data in data_information:
            value_template += f"{data['table_name']}({', '.join(data['examples'])});\n"

        self.__cell_value_referencing = value_template
        return self

    def include_target(self, number: int):
        Variations = {
            1: 'Ответь на вопрос SQLite sql-запросом и без объяснений.\n',
            2: ''
        }
        return Variations[number]

    def include_few_shot(self, number: int):
        if self.__few_shot is None:
            raise IndexError('Не добавлен few_shot')

        Variations = {
            1: f'### Примеры похожих запросов и ответы на них:\n{self.__few_shot}\n',
            2: ''
        }
        return Variations[number]

    def include_schema_template(self, number: int):
        if self.__schema_template is None:
            raise IndexError('Не добавлен schema_template')

        Variations = {
            1: f'### Схема таблиц:\n{self.__schema_template}\n',
            2: ''
        }
        return Variations[number]

    def include_cell_value_referencing(self, number: int):
        if self.__cell_value_referencing is None:
            raise IndexError('Не добавлен cell_value_referencing')

        Variations = {
            1: f'### Примеры данных в таблице:\n{self.__cell_value_referencing}\n',
            2: ''
        }
        return Variations[number]

    def include_question(self, number: int):
        Variations = {
            1: f'### Вопрос: {self.__question}\n### SQL:\n\n',
            2: ''
        }
        return Variations[number]

    def build_prompt(self, number: int):
        Variations = {
            1: {
                'include_target': 1,
                'include_few_shot': 1,
                'include_schema_template': 1,
                'include_cell_value_referencing': 1,
                'include_question': 1
            },
            2: {
                'include_question': 1
            }
        }

        for func_name, value in Variations[number].items():
            if hasattr(self, func_name):  # Проверяем, существует ли метод у объекта
                func = getattr(self, func_name) 
                self.__prompt += func(value)  
            else:
                print(f"Функция {func_name} не найдена")

        return self.__prompt
