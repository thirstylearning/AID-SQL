from typing import List

DATASETS_ARGS: List[str] = ['spider', 'bird', 'spider-bird']


class DB_PATH:
    SPIDER = "dataset/spider-dataset/spider/database"
    # TODO: add bird db path
    BIRD = "dataset"


class RAG_PREPROCESSED_DATA_PATH:
    PREFIX = "./preprocessed_data/rag_preprocessed_data/"

    SPIDER = f"{PREFIX}rag_preprocessed_spider_train.json"


class DATASET_LIST:
    SPIDER = ["spider"]
    BIRD = ["bird"]
    SPIDER_BIRD = ["spider", "bird"]


class PREPROCESSED_DATASET_PATH:
    SPIDER_TRAIN = "./preprocessed_data/preprocessed_train_spider_natsql.json"
    SPIDER_DEV = "./preprocessed_data/preprocessed_ranked_dev_spider_natsql.json"

    BIRD_TRAIN = ""
    BIRD_DEV = ""


class DB_SCHAME_TYPE:
    CODE_REPR = "CODE_REPR"
    OPENAI_REPR = "OPENAI_REPR"
    BASIC_REPR = "BASIC_REPR"


class EXAMPLE_SELECTOR_TYPE:
    RANDOM = "random"

    DEF_EMBEDDING_SIMILARITY_EUCLIDEAN = "def_embedding_similarity_euclidean"
    DEF_EMBEDDING_SIMILARITY_COSINE = "def_embedding_similarity_cosine"

    RERANK_BIG2SMALL_DEF_EMBEDDING_SIMILARITY_EUCLIDEAN = "rerank_big2small_def_embedding_similarity_euclidean"
    RERANK_BIG2SMALL_DEF_EMBEDDING_SIMILARITY_COSINE = "rerank_big2small_def_embedding_similarity_cosine"

    RERANK_SMALL_SORT_DEF_EMBEDDING_SIMILARITY_EUCLIDEAN = "rerank_small_sort_def_embedding_similarity_euclidean"


class EXAMPLE_ORG_TYPE:
    FULL = "FULL"
    Q_SQL = "Q_SQL"
    ONLY_SQL = "ONLY_SQL"


class INSTRUCT_TYPE:
    ZEROSHOT_BASIC = "ZEROSHOT_BASIC_INSTRUCT"

    BASIC = "BASIC_INSTRUCT"


class EMBEDDING_TYPE:
    DEFAULT = "default"


class SIMILARITY_TYPE:
    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"
    IP = "IP"


class EMBEDDING_FIELD:
    QUESTION = "QUESTION"
    SQL = "SQL"
    SQL_SKELETON = "SQL_SKELETON"
    DB_SCHEMA = "DB_SCHEMA"
    SCHEMA_LINK = "SCHEMA_LINK"

    QUESTION_SQL = "QUESTION_SQL"
    QUESTION_SQL_SKELETON = "QUESTION_SQL_SKELETON"

    FULL = "FULL"  # QUESTION_SCHEMA_LINK_SQL


class LLM_TYPE:
    DUMP = "dump"

    GPT_3_5_TURBO = "gpt-3.5-turbo"
