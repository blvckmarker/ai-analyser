from abc import ABC
from dataclasses import dataclass

@dataclass
class Span(ABC):
    pass


@dataclass
class ExtendedSqlSpan(Span):
    NL : str
    sql_gold : str
    sql_pred : str
    df_soft : int
    df_flexible : int
    df_gold_IN_df_pred : bool
    df_pred_IN_df_gold : bool
    df_gold_columns : list[str]
    df_pred_columns : list[str]
    TED : int
    Error : str | None
