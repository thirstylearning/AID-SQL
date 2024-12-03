import os
import json
import datetime
import torch

from typing import Dict, List

from utils.enums import RAG_PREPROCESSED_DATA_PATH


from utils.schemas import PredictedSQL, PreprocessedDatasetSample
from utils.sql_parse_classify import get_difficulty_type

from third_party.test_suite.evaluation import count_component1, count_component2, count_others


def get_rag_dataset_path_by_name(dataset_name: str) -> str:
    if dataset_name == "spider":
        return RAG_PREPROCESSED_DATA_PATH.SPIDER
    elif dataset_name == "bird":
        # raise ValueError("Bird dataset is not supported yet")
        return RAG_PREPROCESSED_DATA_PATH.BIRD
    else:
        raise ValueError(f"RAG Dataset {dataset_name} is not supported yet")


def get_tc_dict_from_db_schema(db_schema: List[Dict]) -> Dict:
    """
    get key-table, value-columns dict, from db_schema field
    """
    table_cols_dict: Dict = {}
    for table in db_schema:
        table_cols_dict[table["table_name_original"]] = [col_name for col_name in table["column_names_original"]]

    return table_cols_dict


def get_tc_schema_link_dict_from_db_schema(
    db_schema: List[Dict], table_labels: List[int], column_labels: List[List[int]]
) -> Dict:
    """
    get key-table, value-columns dict, from db_schema field with used tables and columns
    """
    tc_schema_link_dict = {}
    for table_idx, (table, table_label) in enumerate(zip(db_schema, table_labels)):
        if table_label == 0:
            continue

        used_col_names: List[str] = []
        for col_name, col_label in zip(table["column_names_original"], column_labels[table_idx]):
            if col_label == 0:
                continue

            used_col_names.append(col_name)

        tc_schema_link_dict[table["table_name_original"]] = used_col_names

    return tc_schema_link_dict


def get_tc_sequence_from_tc_dict(tc_dict: Dict[str, List[str]]) -> str:
    """
    get BASIC_REPR liked db_schema sequence from tc_dict
    """
    table_sequence_list: List[str] = []
    for table_name, cols_list in tc_dict.items():
        columns_sequence: str = ",".join(cols_list)
        table_sequence: str = f"Table {table_name}, columns = [{columns_sequence}]"
        table_sequence_list.append(table_sequence)

    return "\n".join(table_sequence_list)


def get_tc_schema_link_dot_sequence_from_schema_link_dict(tc_schema_link_dict: Dict[str, List[str]]) -> str:
    """
    dot seq like: table1.col1, table2.col2
    """
    t_dot_c_list: List[str] = []
    for table_name, cols_list in tc_schema_link_dict.items():
        for col_name in cols_list:
            t_dot_c_list.append(f"{table_name}.{col_name}")

    return ",".join(t_dot_c_list)


def get_schema_flat_list_from_db_schema(db_schema: List[Dict]) -> List[str]:
    schema_flat_list = []
    for table_idx, table in enumerate(db_schema):
        table_name = table["table_name_original"]
        schema_flat_list.append(table_name)

        for col_name in table["column_names_original"]:
            schema_flat_list.append(col_name)

    return schema_flat_list

def get_tables_list_from_db_schema(db_schema: List[Dict]) -> List[str]:
    tables_list = []
    for table_idx, table in enumerate(db_schema):
        table_name = table["table_name_original"]
        tables_list.append(table_name)

    return tables_list

def get_columns_list_from_db_schema(db_schema: List[Dict]) -> List[str]:
    columns_list = []
    for table_idx, table in enumerate(db_schema):
        for col_name in table["column_names_original"]:
            columns_list.append(col_name)

    return columns_list


def add_predicted_sqls_to_test_dataset(test_dataset: List[Dict], predicted_sqls_paths: List[str]) -> List[Dict]:
    for pre_sqls_path in predicted_sqls_paths:
        with open(pre_sqls_path, "r") as f:
            predicted_sqls_path: List[str] = f.readlines()

        if len(predicted_sqls_path) != len(test_dataset):
            raise ValueError("Length of predicted sqls and test dataset should be equal")

        for data_idx, pred_sql in enumerate(predicted_sqls_path):
            test_data: Dict = test_dataset[data_idx]
            test_data_sample_obj: PreprocessedDatasetSample = PreprocessedDatasetSample(**test_data)
            pred_difficulty_type: str = get_difficulty_type(pred_sql)
            pred_spider_difficulty_type: str = None
            pre_sql_obj: PredictedSQL = PredictedSQL(
                sql=pred_sql, difficulty_type=pred_difficulty_type, spider_difficulty=pred_spider_difficulty_type
            )
            test_data_sample_obj.predicted_sql_list.append(pre_sql_obj)
            test_data = test_data_sample_obj.model_dump()
            test_dataset[data_idx] = test_data

    return test_dataset


def load_preprocessed_dataset(path: str) -> List[PreprocessedDatasetSample]:
    with open(path, "r") as f:
        preprocessed_dataset: List[Dict] = json.load(f)

    preprocessed_dataset_obj_list: List[PreprocessedDatasetSample] = [
        PreprocessedDatasetSample(**_data) for _data in preprocessed_dataset
    ]

    return preprocessed_dataset_obj_list


def eval_spider_difficulty(sql: Dict):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
        count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium"
    elif (
        (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
        or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
        or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard"
    else:
        return "extra"


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def create_directory(path):
    if os.path.exists(path):
        print('\n')
        print_message("#> Note: Output directory", path, 'already exists\n\n')
    else:
        print('\n')
        print_message("#> Creating directory", path, '\n\n')
        os.makedirs(path)


def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result
        
        

def print_message(*s, condition=True, pad=False):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f'\n{msg}\n'
        print(msg, flush=True)


    return msg


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result

    
def torch_load_dnn(path):
    if path.startswith("http:") or path.startswith("https:"):
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        dnn = torch.load(path, map_location='cpu')
    
    return dnn

    
def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    print(f"#> Saving a checkpoint to {path} ..")

    if hasattr(model, 'module'):
        model = model.module  # extract model from a distributed/data-parallel wrapper

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['arguments'] = arguments

    torch.save(checkpoint, path)