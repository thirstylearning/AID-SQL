import json
import os

from sqlalchemy.sql.functions import random

from utils.enums import *

from typing import List, Dict, Callable, Optional

from sympy.logic.boolalg import Boolean


def load_data_factory(datasets_list: List[str],
                      db_ids: Optional[List[str]],
                      difficulty_types: Optional[List[str]],
                      domains: Optional[List[str]]) -> Callable:
    def _check_legal(sample: Dict) -> bool:
        if db_ids is not None and sample['db_id'] not in db_ids:
            return False
        if difficulty_types is not None and sample['difficulty_type'] not in difficulty_types:
            return False
        if domains is not None and sample['domain'] not in domains:
            return False
        return True

    for dataset in datasets_list:
        if dataset not in DATASETS_ARGS:
            raise ValueError(f"Dataset {dataset} is not supported")

    dataset_path_list: List[str] = []
    if 'spider' in datasets_list:
        dataset_path_list.append(PREPROCESSED_DATASET_PATH.SPIDER_DEV)
    if 'bird' in datasets_list:
        dataset_path_list.append(PREPROCESSED_DATASET_PATH.BIRD_DEV)

    def load_data() -> List[Dict]:
        data_list: List[Dict] = []
        for dataset_path in dataset_path_list:
            with open(dataset_path, 'r') as f:
                data_list.extend(json.load(f))
        legal_data_list: List[Dict] = [sample for sample in data_list if _check_legal(sample)]

        if len(legal_data_list) == 0:
            raise ValueError("No specified samples found in datasets")

        return legal_data_list

    return load_data


def get_datasets_list(datasets: str) -> List[str]:
    if datasets == 'spider':
        return DATASET_LIST.SPIDER
    elif datasets == 'bird':
        return DATASET_LIST.BIRD
    else:
        raise ValueError(f"Dataset {datasets} is not supported")


def load_test_data(datasets: Optional[str] = None,
                   db_ids: Optional[List[str]] = None,
                   difficulty_types: Optional[List[str]] = None,
                   domains: Optional[List[str]] = None,
                   mini_test: Boolean = False) -> List[Dict]:
    """"
    Load test data using the specified conditions

    Parameters:
    datasets: spider, bird, or others
    db_ids: use which db's samples, if None, use all
    difficulty_types: [easy, non-nested, nested], if None, use all
    domains: if None, use all
    mini_test: if True, use sports database's samples of spider dataset for testing
    """
    if datasets is None:
        datasets = 'spider'
    datasets_list = get_datasets_list(datasets)

    if mini_test:
        datasets_list = DATASET_LIST.SPIDER
        db_ids = ['wta_1', 'poker_player']

    load_data: Callable = load_data_factory(datasets_list, db_ids, difficulty_types, domains)

    return load_data()
