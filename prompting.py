from utils import find_similar_sentences
import pandas as pd


class PromptBuilder:
    def __init__(self, prompt_prefix=''):
        self.__prompt = prompt_prefix

    def add_few_shot(self, sentence_model, target_question, queries, message='Example of similar tasks and answers for them'):
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

        self.__prompt += f'\n### {message}\n'
        self.__prompt += few_shot_template
        return self

    def add_schema_template(self, db_conn, message='Tables schema:'):
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
        self.__prompt += f'\n### {message}\n'
        self.__prompt += schema_template
        return self

    def add_cell_value_referencing(self, db_conn, count=1, message='Example of rows in table:'):
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

        self.__prompt += f'\n### {message}\n'
        self.__prompt += value_template
        return self
        
    def add_message(self, message):
        self.__prompt += f'\n### {message}\n'
        return self

    def build_prompt(self):
        return self.__prompt