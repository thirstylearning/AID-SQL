import json
import argparse
from typing import Dict, Tuple
import ujson
import random

from tqdm import tqdm
import numpy as np

from utils.enums import *
from utils.schemas import RAGPreprocessedDatasetSample
from rag.VectorDBCollection import VectorDBCollection


default_train_set_path = "preprocessed_data/rag_preprocessed_data/rag_preprocessed_spider_train.json"
default_output_path = "preprocessed_data/contrast_triples/examples"

sql_simi_rag_collection = VectorDBCollection()
train_set_size = 0


def parse_option():
    parser = argparse.ArgumentParser("contrast_tripels_propare.py")

    parser.add_argument("--train_set_path", type=str, default=default_train_set_path, help="path to the train set")
    parser.add_argument("--collection_setting_path", type=str, default=MY_COLLECTION_SET_PATH.DEF_EUC_SQL_SPIDER)
    parser.add_argument("--output_path", type=str, default=default_output_path)
    parser.add_argument("--n_pos", type=int, default=5)
    parser.add_argument("--pos_neg_gap_count", type=int, default=0)
    parser.add_argument("--n_neg", type=int, default=200)

    parser.add_argument("--triple_type", type=str, default="single_pos", choices=["single_pos", "pos_neg"])

    opt = parser.parse_args()
    return opt


def sql_simi_pos_sample_select(example: RAGPreprocessedDatasetSample, n_samples: int) -> Tuple[List[str], List[float]]:
    query_res = sql_simi_rag_collection.collection.query(
        query_texts=[example.sql],
        n_results=n_samples * 5,
    )

    top_n_ids = query_res["ids"][0]
    assert len(top_n_ids) == n_samples * 5, (len(top_n_ids), n_samples * 5)

    top_n_distances = query_res["distances"][0]
    assert len(top_n_distances) == n_samples * 5, (len(top_n_distances), n_samples * 5)

    while top_n_distances[0] == 0.0:
        top_n_ids.pop(0)
        top_n_distances.pop(0)

    assert len(top_n_ids) == len(top_n_distances), (len(top_n_ids), len(top_n_distances))
    assert len(top_n_ids) > n_samples, (len(top_n_ids), n_samples)

    return (
        top_n_ids[:n_samples],
        [(1.0 / (distance + 0.01)) * 10.0 for distance in top_n_distances[:n_samples]],
    )


def random_neg_sample_select(
    example: RAGPreprocessedDatasetSample, n_samples: int, pos_samples_ids: List[str]
) -> Tuple[List[str], List[float]]:
    query_res = sql_simi_rag_collection.collection.query(
        query_texts=[example.sql],
        n_results=train_set_size,
    )

    ids_list = query_res["ids"][0]
    distances_list = query_res["distances"][0]

    random_sample_idx = random.sample(range(n_samples + 1, train_set_size - 1), n_samples)

    neg_samples_ids = [ids_list[idx] for idx in random_sample_idx]
    neg_samples_distances = [distances_list[idx] for idx in random_sample_idx]

    return (
        neg_samples_ids,
        [(1.0 / distance) * 10.0 for distance in neg_samples_distances],
    )


def prepare_pos_neg_triples(opt):
    pos_sample_func = sql_simi_pos_sample_select
    neg_sample_func = random_neg_sample_select
    n_samples: int = 5

    with open(opt.train_set_path, "r") as f:
        train_set = json.load(f)
    global train_set_size
    train_set_size = len(train_set)

    triples_list = []

    # mini test
    # train_set = train_set[:500]

    example_dict: Dict
    for example_dict in tqdm(train_set, desc="Processing triples dataset"):
        example = RAGPreprocessedDatasetSample(**example_dict)
        pos_samples_ids, pos_scores = pos_sample_func(example, n_samples)
        neg_samples_ids, neg_scores = neg_sample_func(example, n_samples, pos_samples_ids)

        triples_list.append(
            [
                example.id,
                *[[p_id, p_s] for p_id, p_s in zip(pos_samples_ids, pos_scores)],
                *[[n_id, n_s] for n_id, n_s in zip(neg_samples_ids, neg_scores)],
            ]
        )

    with open(opt.output_path, "w") as f:
        for list_elem in triples_list:
            list_line_txt: str = json.dumps(list_elem)
            f.write(list_line_txt + "\n")

    print(f"contrast triples saved to {opt.output_path}")


def prepare_single_pos_triples(opt):
    n_pos: int = opt.n_pos

    with open(opt.train_set_path, "r") as f:
        train_set = json.load(f)

    global train_set_size
    train_set_size = len(train_set)
    
    # mini test
    # train_set = train_set[:500]

    triple_list = []
    neg_id_dict = {}

    example_dict: Dict
    for example_dict in tqdm(train_set, desc="Processing triples dataset"):
        example = RAGPreprocessedDatasetSample(**example_dict)
        all_query_res = sql_simi_rag_collection.collection.query(
            query_texts=[example.sql],
            n_results=train_set_size,
        )
        all_ids = all_query_res["ids"][0]
        all_distances = all_query_res["distances"][0]
        all_documents: List[str] = all_query_res["documents"][0]

        while all_documents[0] == example.sql:
            all_ids.pop(0)
            all_distances.pop(0)
            all_documents.pop(0)

        assert len(all_ids) == len(all_distances), (len(all_ids), len(all_distances))

        top_n_pos_ids = all_ids[:n_pos]
        top_n_pos_distances = all_distances[:n_pos]
        
        neg_start_idx = n_pos + opt.pos_neg_gap_count
        neg_end_idx = n_pos + opt.pos_neg_gap_count + opt.n_neg
        easy_neg_start_idx = neg_end_idx
        
        neg_ids = all_ids[neg_start_idx : neg_end_idx]
        neg_distances = all_distances[neg_start_idx : neg_end_idx]
        easy_neg_ids = all_ids[easy_neg_start_idx:] 
        easy_neg_distance = all_distances[easy_neg_start_idx:]
        
        assert len(neg_ids) == len(neg_distances), (len(neg_ids), len(neg_distances))

        # use numpy to calculate score from distance: score = 1 / (distance + 0.01) * 10
        top_n_pos_scores = (10.0 / (np.array(top_n_pos_distances) + 0.01)).tolist()
        neg_scores = (10.0 / (np.array(neg_distances) + 0.01)).tolist()
        easy_neg_scores = (10.0 / (np.array(easy_neg_distance) + 0.01)).tolist()

        triple_list.extend([[example.id, [p_id, p_s]] for p_id, p_s in zip(top_n_pos_ids, top_n_pos_scores)])
        
        neg_id_dict[example.id] = {
            "difficulty_ids": neg_ids,
            "difficulty_scores": neg_scores,
            "easy_ids": easy_neg_ids,
            "easy_scores": easy_neg_scores,
        }
    
    with open(opt.output_path, "w") as f:
        for list_elem in triple_list:
            list_line_txt: str = json.dumps(list_elem)
            f.write(list_line_txt + "\n")
            
    print(f"contrast triples saved to {opt.output_path}")
            
    with open(opt.output_path + ".neg", "w") as f:
        f.write(json.dumps(neg_id_dict, indent=4))

    print(f"contrast triples neg ids saved to {opt.output_path}.neg")



if __name__ == "__main__":
    opt = parse_option()
    
    spider_def_euc_sql_collection_setting_path = MY_COLLECTION_SET_PATH.DEF_EUC_SQL_SPIDER
    
    opt.collection_setting_path = spider_def_euc_sql_collection_setting_path
    opt.train_set_path = default_train_set_path
    opt.output_path = default_output_path

    sql_simi_rag_collection.load_setting(setting_path=opt.collection_setting_path)

    if opt.triple_type == "pos_neg":
        prepare_pos_neg_triples(opt)
    elif opt.triple_type == "single_pos":
        prepare_single_pos_triples(opt)
    else:
        raise NotImplementedError()
