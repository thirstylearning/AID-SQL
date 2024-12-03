from functools import partial
from typing import List
import random
import json
import os

from rag.VectorDBCollection import VectorDBCollection
from rag.rag_dataloader import RAGDataloader

from infra.config.config import ColBERTConfig
from modeling.tokenization import (
    QueryTokenizer,
    DocTokenizer,
    tensorize_triples,
)
from third_party.colbert_v1.ColBERT_v1.colbert.data.examples import Examples


class MyBatcher:
    def __init__(self, config: ColBERTConfig, triples, rag_collection_path, rag_dataset_path, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway
        self.easy_nway = config.easy_nway
        assert self.easy_nway < self.nway - 1, (self.easy_nway, self.nway)
        self.difficult_nway = self.nway - 1 - self.easy_nway
        assert self.nway == self.easy_nway + self.difficult_nway + 1, (self.nway, self.easy_nway, self.difficult_nway)

        self.query_str_format_type: str = config.query_str_formatter_type
        self.doc_str_format_type: str = config.doc_str_formatter_type

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.rag_dataloader = RAGDataloader(paths=rag_dataset_path)

        self._get_query = self._default_get_query
        self._get_doc = self._defautl_get_doc

        self.rag_collection = VectorDBCollection()
        self.rag_collection.load_setting(setting_path=rag_collection_path)

        self._neg_ids_dict = self._load_neg_ids_dict(triples)

        assert len(triples) > 0, "No triples for training"

    def _default_get_query(self, query_id) -> str:
        if self.query_str_format_type == "question":
            return self.rag_dataloader[query_id].question
        elif self.query_str_format_type == "question_sql":  # for training, use true sql
            return self.rag_dataloader[query_id].question + "\n" + self.rag_dataloader[query_id].sql
        elif self.query_str_format_type == "question_evidence":  # bird dataset only
            return self.rag_dataloader[query_id].question + " " + self.rag_dataloader[query_id].evidence
        elif self.query_str_format_type == "question_evidence_sql":  # bird dataset only
            return f"{self.rag_dataloader[query_id].question} {self.rag_dataloader[query_id].evidence}\n{self.rag_dataloader[query_id].sql}"
        else:
            raise ValueError(f"Unknown query_str_format_type: {self.query_str_format_type}")

    def _defautl_get_doc(self, doc_id) -> str:
        if self.doc_str_format_type == "sql":
            return self.rag_dataloader[doc_id].sql
        elif self.doc_str_format_type == "question_sql":
            return self.rag_dataloader[doc_id].question + "\n" + self.rag_dataloader[doc_id].sql
        else:
            raise ValueError(f"Unknown doc_str_format_type: {self.doc_str_format_type}")

    @classmethod
    def _load_neg_ids_dict(cls, triples):
        assert isinstance(triples, str), triples
        _triples_neg_path = triples + ".neg"
        assert os.path.exists(_triples_neg_path), _triples_neg_path

        with open(_triples_neg_path, "r") as f:
            neg_ids_dict = json.load(f)

        assert isinstance(neg_ids_dict, dict), neg_ids_dict
        return neg_ids_dict

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, pos_sample = self.triples[position]

            neg_difficulty_ids_list = self._neg_ids_dict[query]["difficulty_ids"]
            neg_difficulty_scores_list = self._neg_ids_dict[query]["difficulty_scores"]

            # random sample difficulty nway negative sample
            difficulty_random_idx = random.sample(range(len(neg_difficulty_ids_list)), self.difficult_nway)
            difficulty_neg_samples_list = []
            for _idx in difficulty_random_idx:
                difficulty_neg_samples_list.append([neg_difficulty_ids_list[_idx], neg_difficulty_scores_list[_idx]])

            neg_easy_ids_list = self._neg_ids_dict[query]["easy_ids"]
            neg_easy_scores_list = self._neg_ids_dict[query]["easy_scores"]

            # random sample easy nway negative sample
            easy_random_idx = random.sample(range(len(neg_easy_ids_list)), self.easy_nway)
            easy_neg_samples_list = []
            for _idx in easy_random_idx:
                easy_neg_samples_list.append([neg_easy_ids_list[_idx], neg_easy_scores_list[_idx]])

            pos_neg_samples_list = [pos_sample] + difficulty_neg_samples_list + easy_neg_samples_list

            assert len(pos_neg_samples_list) == self.nway, len(pos_neg_samples_list)

            # query = self.queries[query]
            query: str = self._get_query(query)

            doc_ids, scores = zip(*pos_neg_samples_list)

            # passages = [self.collection[pid] for pid in doc_ids]
            docs: List[str] = [self._get_doc(_d_id) for _d_id in doc_ids]

            all_queries.append(query)
            all_passages.extend(docs)
            all_scores.extend(scores)

        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, docs, scores):
        assert len(queries) == self.bsize
        assert len(docs) == self.bsize * self.nway

        return self.tensorize_triples(queries, docs, scores, self.bsize // self.accumsteps, self.nway)

    def shuffle(self):
        random.shuffle(self.triples)
        self.position = 0

    def reset(self):
        self.position = 0
