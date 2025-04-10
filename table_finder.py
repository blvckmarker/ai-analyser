import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DtoColumn:
    Name : str
    Description : str


@dataclass
class DtoTable:
    Name : str
    Description : str
    Columns : list[DtoColumn]


def prepare_df(df: pd.DataFrame) -> list[DtoTable]:
    tables = []

    for table in tqdm(df['table'].unique()):
        t : pd.DataFrame = df[df['table'] == table]
        columns : list[DtoColumn] = []
        for idx in t.index:
            name = str(t[t.index == idx]['field'][idx]).strip()
            desc = str(t[t.index == idx]['field_description'][idx]).strip()

            column = DtoColumn(name, desc)
            columns.append(column)

        dto_table = DtoTable(table, str(t['table_description'][idx]).strip(), columns)
        tables.append(dto_table)

    return tables


def generate_table_profile(table : DtoTable) -> str:
    profile = []
    
    profile.append(f"Таблица: {table.Name}")
    profile.append(f"Описание таблицы: {table.Description}")

    profile.append("Колонки:")
    for col in table.Columns:
        profile.append(f"- {col.Name} - {col.Description}")
    
    return "\n".join(profile)


class TableFinder:
    def __init__(self, tables):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.table_profiles = [generate_table_profile(t) for t in tables]
        self.table_embeddings = self.model.encode(self.table_profiles)
        self.tables = tables
    
    def find_tables(self, question: str, top_k: int = 5) :
        question_embedding = self.model.encode(question)
        
        similarities = []
        for emb in self.table_embeddings:
            cos_sim = np.dot(question_embedding, emb) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(emb)
            )
            similarities.append(cos_sim)
        
        sorted_indices = np.argsort(similarities)[::-1]
        return [(self.tables[i], similarities[i]) for i in sorted_indices[:top_k]]


class HybridFinder(TableFinder):
    def __init__(self, tables):
        super().__init__(tables)
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.table_profiles)
    
    def find_tables(self, question: str, top_k: int = 5, alpha: float = 0.7, verbose : bool = False):
        semantic_scores = np.array([ex[1] for ex in super().find_tables(question, top_k=len(self.tables))])
        
        question_tfidf = self.tfidf.transform([question])
        keyword_scores = np.dot(question_tfidf, self.tfidf_matrix.T).toarray()[0]
        
        combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
        sorted_indices = np.argsort(combined_scores)[::-1]
        if verbose:
            return [(self.tables[i], combined_scores[i]) for i in sorted_indices[:top_k]]
        else:
            return [self.tables[i] for i in sorted_indices[:top_k]]