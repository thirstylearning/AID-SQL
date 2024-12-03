import os, json
from typing import Dict, Union, List, Optional
import chromadb

from dotenv import load_dotenv
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    Embeddable,
    EmbeddingFunction,
    Documents,
)
from nltk.app.wordnet_app import SIMILAR

from typing import Callable, Type

from sympy.abc import lamda
from tqdm import tqdm

from prompt.DBSchemaReprTemplate import BasicRepr
from prompt.PromptTemplate import PromptBase
from prompt.ExampleOrganizeTemplate import FullExampleOrg, QuesSQLExampleOrg, ExampleOrgBase, SQLExampleOrg, \
    QuesExampleOrg
from rag.rag_dataloader import RAGDataloader
from utils.schemas import MyCollectionSetting, RAGPreprocessedDatasetSample
from utils.enums import *
from utils.utils import get_tc_dict_from_db_schema, get_tc_schema_link_dict_from_db_schema, \
    get_tc_sequence_from_tc_dict, get_tc_schema_link_dot_sequence_from_schema_link_dict


def get_embedding_method(embedding_method_type: str) -> EmbeddingFunction[Documents]:
    if embedding_method_type == EMBEDDING_TYPE.DEFAULT:
        return DefaultEmbeddingFunction()
    else:
        raise ValueError(f"embedding method {embedding_method_type} not supported.")


def get_similarity_method(simi_method_type: str) -> Dict[str, str]:
    if simi_method_type == SIMILARITY_TYPE.COSINE:
        return {"hnsw:space": "cosine"}
    elif simi_method_type == SIMILARITY_TYPE.EUCLIDEAN:
        return {"hnsw:space": "l2"}
    elif simi_method_type == SIMILARITY_TYPE.IP:
        return {"hnsw:space": "ip"}
    else:
        raise ValueError(f"similarity method {simi_method_type} not supported.")


def get_doc_seq_format_func(used_emb_fields: str) -> Callable[[RAGPreprocessedDatasetSample], str]:
    org_cls: Type[ExampleOrgBase]

    if used_emb_fields == EMBEDDING_FIELD.QUESTION:
        org_cls = QuesExampleOrg
    elif used_emb_fields == EMBEDDING_FIELD.SQL:
        org_cls = SQLExampleOrg
    elif used_emb_fields == EMBEDDING_FIELD.SQL_SKELETON:
        raise NotImplementedError()
    #         return lambda rag_sample: rag_sample.sql_skeleton
    elif used_emb_fields == EMBEDDING_FIELD.DB_SCHEMA:
        raise NotImplementedError()
    #         return lambda rag_sample: get_tc_sequence_from_tc_dict(
    #             get_tc_dict_from_db_schema(**rag_sample.model_dump())
    #         )
    elif used_emb_fields == EMBEDDING_FIELD.SCHEMA_LINK:
        raise NotImplementedError()
    #         return lambda rag_sample: get_tc_schema_link_dot_sequence_from_schema_link_dict(
    #             get_tc_schema_link_dict_from_db_schema(**rag_sample.model_dump())
    #         )
    elif used_emb_fields == EMBEDDING_FIELD.QUESTION_SQL:
        org_cls = QuesSQLExampleOrg
    elif used_emb_fields == EMBEDDING_FIELD.QUESTION_SQL_SKELETON:
        raise NotImplementedError()
    #         return lambda rag_sample: "\n".join([rag_sample.question, rag_sample.sql_skeleton])
    elif used_emb_fields == EMBEDDING_FIELD.FULL:
        org_cls = FullExampleOrg
    else:
        raise ValueError(f"used embedding fields: {used_emb_fields} not supported.")

    class _TmpPromptCls(BasicRepr, org_cls, PromptBase):
        def __init__(self):
            super().__init__()

    return _TmpPromptCls.format_example


def sql_difficulty_type_to_chroma_meta(difficulty_type: List[str]) -> Dict[str, bool]:
    meta_dict: Dict[str, str] = {}
    if "join" in difficulty_type:
        meta_dict["join"] = True
    if "nested" in difficulty_type:
        meta_dict["nested"] = True
        
    return meta_dict

class VectorDBCollection:
    VECTOR_DB_SAVE_PATH = "./vectorDB/chroma"
    VECTOR_DB_COLLECTION_SETTING_PATH = "./vectorDB/my_collection_settings"

    def __init__(self) -> None:
        self.setting: Union[MyCollectionSetting, None] = None
        self.setting_dir: str = ""

        self.client: Union[chromadb.ClientAPI, None] = None
        self.collection: Optional[Collection] = None

        self.used_emb_fields_to_doc: Optional[Callable[[RAGPreprocessedDatasetSample], str]] = None

        self.rag_dataloader: Union[RAGDataloader, None] = None

        self._load_persisted_client()

    def load_setting(self, setting_path: str) -> None:
        if not os.path.exists(setting_path):
            raise FileNotFoundError(f"File {setting_path} not found.")

        with open(setting_path, "r", encoding="utf-8") as f:
            self.setting = MyCollectionSetting(**json.load(f))

        self.collection = self.client.get_collection(self.setting.collection_name)
        self.rag_dataloader = RAGDataloader(paths=self.setting.rag_dataset_paths)

        print(f"Loaded setting for collection {self.setting.collection_name}, which contains rag datasets:")
        for rag_dataset_path in self.setting.rag_dataset_paths:
            print(f" - {rag_dataset_path}")

    def _load_persisted_client(self) -> None:
        persisted_path = self.VECTOR_DB_SAVE_PATH
        assert isinstance(persisted_path, str)
        self.client = chromadb.PersistentClient(path=persisted_path)

    def create_collection(self, collection_name: str, embedding_method: str, similarity_method: str,
                          used_embedding_fields: str) -> None:
        """
        if the collection name already exists in local db, will overwrite it.
        """
        collection_list = self.client.list_collections()
        collection_names = [_collection.name for _collection in collection_list]
        if collection_name in collection_names:
            self.client.delete_collection(collection_name)
            print(f"{collection_name} already existed in local db has been deleted.")

        self.setting = MyCollectionSetting()

        emb_method: EmbeddingFunction[Documents] = get_embedding_method(embedding_method)
        meta_dict: Dict[str, str] = get_similarity_method(similarity_method)

        self.collection = self.client.create_collection(name=collection_name,
                                                        embedding_function=emb_method,
                                                        metadata=meta_dict)

        self.setting.collection_name = collection_name
        self.setting.embedding_method = embedding_method
        self.setting.similarity_method = similarity_method
        self.setting.used_embedding_fields = used_embedding_fields

        self.used_emb_fields_to_doc = get_doc_seq_format_func(used_embedding_fields)

        self.setting_dir = os.path.join(self.VECTOR_DB_COLLECTION_SETTING_PATH, self.setting.collection_name)
        if not os.path.exists(self.setting_dir):
            os.mkdir(self.setting_dir)
        self.setting = MyCollectionSetting(collection_name=self.setting.collection_name,
                                           embedding_method=self.setting.embedding_method,
                                           similarity_method=self.setting.similarity_method,
                                           used_embedding_fields=self.setting.used_embedding_fields)

        print(f"collection {collection_name} has been created with embedding method {embedding_method}.")

    def save_setting(self):
        if self.setting is None:
            raise ValueError("setting not exists, create or load collection first")

        setting_path: str = os.path.join(self.setting_dir, "setting.json")
        with open(setting_path, "w", encoding="utf-8") as f:
            json.dump(self.setting.model_dump(), f, ensure_ascii=False, indent=4)
        print(f"collection setting has been saved to {setting_path}.")

    def insert_rag_dataset(self, rag_dataset_path: str, dataset_name: str) -> None:
        if self.collection is None:
            raise ValueError("Collection not created or loaded.")

        if rag_dataset_path in self.setting.rag_dataset_paths:
            print(f"RAG dataset {rag_dataset_path} already exists in collection setting.")
            return

        if self.rag_dataloader is None:
            self.rag_dataloader = RAGDataloader()
        self.rag_dataloader.load_rag_file(rag_dataset_path)

        documents: List[str] = []
        ids: List[str] = []
        filter_metadatas: List[Dict] = []
        with open(rag_dataset_path, "r", encoding="utf-8") as f:
            _rag_dataset_dict_list: List[Dict] = json.load(f)
            rag_dataset: List[RAGPreprocessedDatasetSample] = [
                RAGPreprocessedDatasetSample(**rag_sample_dict) for rag_sample_dict in _rag_dataset_dict_list
            ]

        for rag_sample in tqdm(rag_dataset, desc=f"reading {rag_dataset_path}"):
            ids.append(rag_sample.id)
            documents.append(self.used_emb_fields_to_doc(rag_sample))
            filter_metadatas.append({
                                     "domain": rag_sample.domain if rag_sample.domain else "",
                                     **sql_difficulty_type_to_chroma_meta(rag_sample.difficulty_type)
                                     })

        with tqdm(total=1, desc=f"inserting to {self.setting.collection_name}", unit="step") as _bar:
            self.collection.add(documents=documents, ids=ids, metadatas=filter_metadatas)
            _bar.update(1)

        # save example embedding doc
        example_emb_doc_path: str = os.path.join(self.setting_dir, "example_emb_doc.txt")
        with open(example_emb_doc_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(doc + "\n" * 5)

        self.setting.rag_dataset_paths.append(rag_dataset_path)
        self.setting.datasets.append(dataset_name)
