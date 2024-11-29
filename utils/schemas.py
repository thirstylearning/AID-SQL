from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union


class MyCollectionSetting(BaseModel):
    collection_name: str = ""

    rag_dataset_paths: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)

    embedding_method: str = ""
    similarity_method: str = ""
    used_embedding_fields: str = ""


class PredictedSQL(BaseModel):
    sql: str

    difficulty_type: Optional[str] = None
    spider_difficulty: Optional[str] = None


class PreprocessedDatasetSample(BaseModel):
    id: Optional[str] = None
    question: str 

    # bird evidence
    evidence: Optional[str] = None

    db_id: str
    sql: str
    norm_sql: str
    sql_skeleton: str
    natsql: Optional[str]
    norm_natsql: Optional[str]
    natsql_skeleton: Optional[str]
    schema_links: List[str]



    # over methods
    domain: Optional[str] = None
    difficulty_type: Union[List[str], str]
    spider_difficulty: Optional[str] = None
    sub_queries: Optional[List[str]] = None

    predicted_sql_list: Optional[List[PredictedSQL]] = Field(default_factory=list)

    db_schema: List[Dict]
    pk: List[Dict]
    fk: List[Dict]
    table_labels: List[int]
    column_labels: List[List]

    column_pred_probs: Optional[List[List[float]]] = None
    table_pred_probs: Optional[List[float]] = None
    ranked_schema_links: Optional[List[str]] = None


class PreprocessedWithProbsDatasetSample(PreprocessedDatasetSample):
    pass


class PreprocessedRankedSchemaDatasetSample(PreprocessedWithProbsDatasetSample):
    pass


class RAGPreprocessedDatasetSample(BaseModel):
    id: str
    question: str 

    # bird evidence
    evidence: Optional[str] = None

    sql: str
    sql_skeleton: str

    db_schema_sequence: str
    golden_schema_link_sequence: str

    db_schema: List[Dict]
    pk: List[Dict]
    fk: List[Dict]
    table_labels: List[int]
    column_labels: List[List[int]]

    db_id: str
    domain: Optional[str] = None
    difficulty_type: Union[List[str], str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        


class Instructions(BaseModel):
    prefix_instruction: Optional[str] = None
    sql_instruction: Optional[str] = None


class QuestionPrompt(BaseModel):
    input_prompt: str

    db_id: str
    domain: Optional[str] = None
    difficulty_type: str
    db_schema: List[Dict]
    pk: List[Dict]
    fk: List[Dict]

    golden_sql: str
    golden_schema_links: List[str]

    few_shot_examples_num: int
    few_shot_examples: Optional[List[Dict]] = None
    few_shot_examples_similarities: Optional[List[float]] = None
    avg_examples_similarity: Optional[float] = None
