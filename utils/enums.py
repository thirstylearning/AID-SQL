from typing import List

DATASETS_ARGS: List[str] = ['spider', 'bird', 'spider-bird']


class DB_PATH:
    SPIDER = "dataset/spider-dataset/spider/database"
    # TODO: add bird db path
    BIRD = "dataset"


class RAG_PREPROCESSED_DATA_PATH:
    PREFIX = "./preprocessed_data/rag_preprocessed_data/"

    SPIDER = f"{PREFIX}rag_preprocessed_spider_train.json"
    BIRD = f"{PREFIX}/rag_preprocessed_bird_train.json"


class DATASET_LIST:
    SPIDER = ["spider"]
    BIRD = ["bird"]
    SPIDER_BIRD = ["spider", "bird"]


class QUERY_TYPE:
    class SPIDER_DIFFICULTY:
        EASY = "easy"
        MEDIUM = "medium"
        HARD = "hard"
        EXTRA = "extra"

    # TODO: change name
    class SQL_TYPE:
        EASY = "easy"
        NON_NESTED = "non-nested"
        NESTED = "nested"


class PREPROCESSED_DATASET_PATH:
    SPIDER_TRAIN = "./preprocessed_data/preprocessed_train_spider.json"
    SPIDER_DEV = "./preprocessed_data/preprocessed_ranked_dev_spider.json"
    SPIDER_TEST = "./preprocessed_data/spider_test/preprocessed_ranked_test_spider.json"

    BIRD_TRAIN = "./preprocessed_data/bird/preprocessed_train_bird.json"
    BIRD_DEV = "./preprocessed_data/bird/preprocessed_dev_bird.json"


class DB_SCHAME_TYPE:
    CODE_REPR = "CODE_REPR"
    OPENAI_REPR = "OPENAI_REPR"
    BASIC_REPR = "BASIC_REPR"


class EXAMPLE_SELECTOR_TYPE:
    RANDOM = "random"

    DEF_EMBEDDING_SIMI_FULL_EUCLIDEAN = "def_embedding_simi_full_euclidean"
    DEF_EMBEDDING_SIMI_FULL_COSINE = "def_embedding_simi_full_cosine"

    DEF_EMBEDDING_SIMI_SQL_EUCLIDEAN_CHEAT = "def_embedding_simi_sql_euclidean_cheat"
    DEF_EMBEDDING_SIMI_SQL_EUC_TYPE_CHEAT = "def_embedding_simi_sql_euc_type_cheat"
    DEF_EMBEDDING_SIMI_SQL_EUC_BYPRE = "def_embedding_simi_sql_euc_bypre"
    DEF_EMBEDDING_SIMI_SQL_EUC_BYPRE_TYPE_BYPRE = "def_embedding_simi_sql_euc_bypre_type_bypre"
    DEF_EMBEDDING_SIMI_QUES_SQL_EUCLIDEAN_CHEAT = "def_embedding_simi_ques_sql_euclidean_cheat"
    DEF_EMBEDDING_SIMI_QUES_EUCLIDEAN = "def_embedding_simi_ques_euclidean"

    RERANK_BIG2SMALL_DEF_EMBEDDING_SIMILARITY_EUCLIDEAN = "rerank_big2small_def_embedding_similarity_euclidean"
    RERANK_BIG2SMALL_DEF_EMBEDDING_SIMILARITY_COSINE = "rerank_big2small_def_embedding_similarity_cosine"

    DEF_RERANK_SMALL_SORT_DEF_EMBEDDING_SIMI_FULL_EUCLIDEAN = "def_rerank_small_sort_def_emb_simi_full_euc"
    DEF_RERANK_SMALL_SORT_DEF_EMBEDDING_SIMI_SQL_EUCLIDEAN_CHEAT = "def_rerank_small_sort_def_emb_simi_sql_euc_cheat"
    DEF_RERANK_SMALL_SORT_DEF_EMBEDDING_SIMI_SQL_EUCLIDEAN_REV_CHEAT = "def_rerank_small_sort_def_emb_simi_sql_euc_rev_cheat"

    DEF_RERANK_500BIG2SMALL_DEF_EMBEDDING_SIMI_FULL_EUCLIDEAN = "def_rerank_500big2small_def_emb_simi_full_euc"
    DEF_RERANK_500BIG2SMALL_DEF_EMBEDDING_SIMI_SQL_EUCLIDEAN_CHEAT = "def_rerank_500big2small_def_emb_simi_sql_euc_cheat"
    DEF_RERANK_100BIG2SAMLL_DEF_EMBEDDING_SIMI_SQL_EUCLIDEAN_CHEAT = "def_rerank_100big2small_def_emb_simi_sql_euc_cheat"
    DEF_RERANK_500BIG2SMALL_DEF_EMBEDDING_SIMI_QUES_EUCLIDEAN = "def_rerank_500big2small_def_emb_simi_ques_euc"
    DEF_RERANK_500BIG2SMALL_DEF_EMBEDDING_SIMI_QUES_SQL_EUCLIDEAN_CHEAT = "def_rerank_500big2small_def_emb_simi_ques_sql_euc_cheat"

    COLBERT_RERANK_3000BIG2SMALL_DEF_EMBEDDING_SIMI_QUES_EUCLIDEAN = "colbert_rerank_3000big2small_def_emb_simi_ques_euc"
    COLBERT_RERANK_QUES2SQL3000BIG2SMALL_DEF_EMB_SIMI_SQL_EUC_BYPRE = "colbert_rerank_ques2sql3000big2small_def_emb_simi_sql_euc_bypre"

    FTCOLBERT_RERANK_QUES2SQLALL2SMALL = "ft_colbert_rerank_ques2sql_all2small"


class EXAMPLE_ORG_TYPE:
    FULL = "FULL"
    Q_SQL = "Q_SQL"
    ONLY_SQL = "ONLY_SQL"
    ONLY_QUES = "ONLY_QUES"


class INSTRUCT_TYPE:
    ZEROSHOT_BASIC = "ZEROSHOT_BASIC_INSTRUCT"
    FEWSHOT_BASIC = "FEWSHOT_BASIC_INSTRUCT"

    FEWSHOT_SQLTYPE_CHEAT = "FEWSHOT_SQLTYPE_INS_CHEAT"
    FEWSHOT_SQLTYPE_BYPRE = "FEWSHOT_SQLTYPE_BYPRE_INS"

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


class MY_COLLECTION_SET_PATH:
    DEF_EUC_FULL_SPIDER = "./vectorDB/my_collection_settings/default_EUCLIDEAN_FULL_spider/setting.json"
    DEF_EUC_SQL_SPIDER = "./vectorDB/my_collection_settings/default_EUCLIDEAN_SQL_spider/setting.json"
    DEF_EUC_QUES_SQL_SPIDER = "./vectorDB/my_collection_settings/default_EUCLIDEAN_QUESTION_SQL_spider/setting.json"
    DEF_EUC_QUES_SPIDER = "./vectorDB/my_collection_settings/default_EUCLIDEAN_QUESTION_spider/setting.json"

    DEF_EUC_SQL_BIRD = "./vectorDB/my_collection_settings/default_EUCLIDEAN_SQL_bird/setting.json"
    DEF_COS_SQL_BIRD = "./vectorDB/my_collection_settings/default_COSINE_SQL_bird/setting.json"

    # TODO: construct
    DEF_EUC_FULL_NO_SQL_SPIDER = ""


class RERANK_MODEL_PATH:
    DEF_RERANKER = "model/rerank/ms-marco-MiniLM-L-6-v2"


class LLM_TYPE:
    DUMP = "dump"

    GPT_3_5_TURBO = "gpt-3.5-turbo"
