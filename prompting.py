import pandas as pd
import numpy as np
import sqlalchemy
from utils.general import find_similar_sentences
from utils.dataset import structure_from_connection, tables_from_connection


class PromptBuilder:
    """
    Класс, отвечающий за создание промпта на основе указанных фичей
    """

    def __init__(self, question):
        self.__prompt = ''
        self.schema_linking = False
        self.__question = question
        self.__few_shot = None
        self.__schema_template = None
        self.__cell_value_referencing = None


    def switch_schema_linking(self, table_structure=None):
        """
        Метод, переключащий режим использования фичи Schema Linking. По умолчанию фича отключена. 
        
        Parameters
        ----------
        table_structure : Any
            Структура таблицы, которая может быть получена с помощью функции `structure_from_connection`
        """

        self.table_structure = table_structure
        self.schema_linking = not self.schema_linking
        return self


    def add_few_shot(self, queries, target_question : str, sentence_model):
        """
        Метод, отвечающий за добавление фичи Few-Shot в промпт

        Parameters
        ----------

        sentence_model : Any
            Модель, позволяющая векторизовать текст
        target_question : str
            Вопрос, для которого нужно найти похожие по смыслу вопросы
        queries : Any
            Набор вопросов и запросов, среди которых нужно найти ближайшие по смыслу вопросы. Объект должен являться матрицей Nx2
        """

        questions = [sample['question'] for sample in queries]

        input_examples = []
        similar = find_similar_sentences(sentence_model, target_question, questions, count=3)
        for sample in queries:
            curr_qs = sample['question']
            if curr_qs in similar:
                input_examples.append([curr_qs, sample['query']])

        few_shot_template = ''
        for ex in input_examples:
            few_shot_template += f'Q: {ex[0]}\n'
            few_shot_template += f'A: {ex[1]}\n'

        self.__few_shot = few_shot_template
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
        self.__schema_template = schema_template
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

        self.__cell_value_referencing = value_template
        return self


    def __include_target(self, number: int):
        Variations = {
            1: '### Ответь на вопрос SQLite sql-запросом и без объяснений.\n',
            2: 'Ответь на вопрос SQLite sql-запросом и без объяснений.\n',
            3: '* Ответь на вопрос SQLite sql-запросом и без объяснений.\n',
            4: 'Ответь на вопрос SQLite sql-запросом и без объяснений\n'
        }
        return Variations[number]


    def __include_few_shot(self, number: int):
        if self.__few_shot is None:
            raise RuntimeError('Не добавлен few_shot')

        Variations = {
            1: f'### Примеры похожих запросов и ответы на них:\n{self.__few_shot}\n',
            2: f'Примеры похожих запросов и ответы на них:\n{self.__few_shot}\n',
            3: f'* Примеры похожих запросов и ответы на них:\n{self.__few_shot}\n',
            4: f'Примеры похожих запросов и ответы на них\n{self.__few_shot}\n'
        }
        return Variations[number]


    def __include_schema_template(self, number: int):
        if self.__schema_template is None:
            raise RuntimeError('Не добавлен schema_template')

        Variations = {
            1: f'### Схема таблиц:\n{self.__schema_template}\n',
            2: f'Схема таблиц:\n{self.__schema_template}\n',
            3: f'* Схема таблиц:\n{self.__schema_template}\n',
            4: f'Схема таблиц\n{self.__schema_template}\n'
        }
        return Variations[number]


    def __include_cell_value_referencing(self, number: int):
        if self.__cell_value_referencing is None:
            raise RuntimeError('Не добавлен cell_value_referencing')

        Variations = {
            1: f'### Примеры данных в таблице:\n{self.__cell_value_referencing}\n',
            2: f'Примеры данных в таблице:\n{self.__cell_value_referencing}\n',
            3: f'* Примеры данных в таблице:\n{self.__cell_value_referencing}\n',
            4: f'Примеры данных в таблице\n{self.__cell_value_referencing}\n'
        }
        return Variations[number]


    def __include_question(self, number: int):
        Variations = {
            1: f'### Вопрос: {self.__question}\n### SQL:\n\n',
            2: f'Вопрос: {self.__question}\nSQL:\n\n',
            3: f'* Вопрос: {self.__question}\n* SQL:\n\n',
            4: f'Вопрос\n {self.__question}\nSQL\n\n',
        }
        return Variations[number]


    def build_prompt(self, number: int):
        """
        Метод, создающий промпт для конкретного случая, определенного числом `number`. 
        Метод должен быть финальным в цепочке вызовов. Цепочка вызовов должна совпадать с соотвествующим случаем

        Parameters
        ----------
        number : int
            Номер случая
        """

        Variations = {
            1: {
                self.__include_target : 1,
                self.__include_few_shot : 1,
                self.__include_schema_template : 1,
                self.__include_cell_value_referencing : 1,
                self.__include_question : 1
            },
            2: {
                self.__include_question : 1
            },
            3:
            {
                self.__include_cell_value_referencing : 1,
                self.__include_question : 1
            },
            4:
            {
                self.__include_question : 1,
                self.__include_schema_template : 1
            }
        }

        for func, value in Variations[number].items():
                self.__prompt += func(value)  

        return self.__prompt
