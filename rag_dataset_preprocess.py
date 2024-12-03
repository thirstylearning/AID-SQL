import json
from typing import Dict, List

from tqdm import tqdm
import argparse

from utils.schemas import RAGPreprocessedDatasetSample
from utils.utils import get_tc_dict_from_db_schema, get_tc_schema_link_dict_from_db_schema, get_tc_sequence_from_tc_dict


def parse_options():
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--input_dataset_path",
        type=str,
        help="preprocessed dataset path, preprocessed_data/preprocessed_train_spider.json for example",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="./preprocessed_data/rag_preprocessed_data/rag_preprocessed_spider_train.json",
        help="output path of the preprocessed train dataset, will be loaded by RAG DataLoader",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.1,
        help="rate of noise, if noise_rate > 0, the order of tables and columns will be shuffled",
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    with open(opt.input_dataset_path, "r", encoding="utf-8") as f:
        dataset: list = json.load(f)

    id_prefix: str = get_id_prefix(opt.input_dataset_path)

    output_dataset: list = []
    for data_id, data_sample in tqdm(enumerate(dataset)):
        if data_sample["id"] is None or data_sample["id"] == "":
            rag_data_sample_id: str = id_prefix + str(data_id)
        else:
            rag_data_sample_id: str = id_prefix + str(data_sample["id"])
        del data_sample["id"]

        scheme_db_dict: Dict[str, List[str]] = get_tc_dict_from_db_schema(data_sample['db_schema'])
        rag_data_sample_db_schema_sequence: str = get_tc_sequence_from_tc_dict(scheme_db_dict)

        used_schema_db_dict: Dict[str, List[str]] = get_tc_schema_link_dict_from_db_schema(
            data_sample['db_schema'], data_sample['table_labels'], data_sample['column_labels']
        )
        rag_data_sample_golden_schema_link_sequence: str = get_tc_sequence_from_tc_dict(used_schema_db_dict)

        rag_preprocessed_data_sample: RAGPreprocessedDatasetSample = RAGPreprocessedDatasetSample(
            id=rag_data_sample_id,
            db_schema_sequence=rag_data_sample_db_schema_sequence,
            golden_schema_link_sequence=rag_data_sample_golden_schema_link_sequence,
            **data_sample)

        output_dataset.append(rag_preprocessed_data_sample.model_dump())

    with open(opt.output_dataset_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(output_dataset, indent=2, ensure_ascii=False))


def get_id_prefix(dataset_path: str) -> str:
    if "spider" in dataset_path:
        return "spider_train_id_"
    elif "bird" in dataset_path:
        return "bird_train_id_"
    else:
        raise ValueError("Unknown dataset name")


if __name__ == "__main__":
    opt = parse_options()
    main(opt)
