import argparse
import os

from utils.enums import RAG_PREPROCESSED_DATA_PATH
from rag.VectorDBCollection import VectorDBCollection


def parse_option():
    parser = argparse.ArgumentParser(description="command line for constructing vector database")

    parser.add_argument('--dataset', nargs='+', type=str,
                        help="datasets used to construct the rag vector collection")

    parser.add_argument('--embedding_method', type=str, default='default',
                        help="method to embed the rag dataset")
    parser.add_argument('--similarity_method', type=str, default='EUCLIDEAN',
                        help='method to calculate vector similarity')
    parser.add_argument('--used_embedding_fields', type=str, default='FULL',
                        help="fields of rag dataset used to construct the collection")

    opt = parser.parse_args()
    return opt


rag_preprocessed_data_path_prefix = 'preprocessed_data/rag_preprocessed_data/'


def get_rag_pre_dataset_path(dataset_name: str) -> str:
    if dataset_name == "spider":
        return RAG_PREPROCESSED_DATA_PATH.SPIDER
    elif dataset_name == "bird":
        return RAG_PREPROCESSED_DATA_PATH.BIRD
    else:
        raise ValueError(f"dataset: {dataset_name} is not supported")


def main(opt: argparse.Namespace):
    collection_name: str = "_".join(
        [opt.embedding_method, opt.similarity_method, opt.used_embedding_fields] + opt.dataset)

    vectorDB_collection = VectorDBCollection()
    vectorDB_collection.create_collection(
        collection_name=collection_name,
        embedding_method=opt.embedding_method,
        similarity_method=opt.similarity_method,
        used_embedding_fields=opt.used_embedding_fields
    )

    # input name in opt.dataset should be formatted like ['spider_train', 'bird_train']
    for dataset_name in opt.dataset:
        dataset_path: str = get_rag_pre_dataset_path(dataset_name)
        assert os.path.exists(dataset_path), f"dataset: {dataset_path} does not exist"
        vectorDB_collection.insert_rag_dataset(dataset_path, dataset_name)

    vectorDB_collection.save_setting()


if __name__ == '__main__':
    opt: argparse.Namespace = parse_option()
    main(opt)
