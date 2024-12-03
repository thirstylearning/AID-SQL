import json
import os
from typing import Dict, List, Union
from tqdm import tqdm

from utils.schemas import RAGPreprocessedDatasetSample


class RAGDataloader:
    def __init__(self, paths=None):
        self.rag_data: Dict[str, RAGPreprocessedDatasetSample] = {}
        self._index = 0
        self._keys = []
        
        if paths is None:
            return
        
        if isinstance(paths, str):
            self.load_rag_file(paths)
        elif isinstance(paths, List):
            self.load_rag_files_list(paths)
        else:
            raise TypeError("Paths should be either a string or a list of strings.")

    def __getitem__(self, index: str) -> RAGPreprocessedDatasetSample:
        if self.rag_data is None:
            raise ValueError("RAG dataset is not loaded yet. Please load the dataset first.")
        if index not in self.rag_data:
            raise KeyError(f"Index {index} not found in the RAG dataset.")
        return self.rag_data[index]

    def __iter__(self):
        self._keys = list(self.rag_data.keys())
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.rag_data):
            key = self._keys[self._index]
            value = self.rag_data[key]
            self._index += 1
            return value
        else:
            raise StopIteration

    def keys(self):
        return self.rag_data.keys()
    
    def items(self):
        for k_v_pair in self.rag_data.items():
            yield k_v_pair

    def __len__(self):
        if self.rag_data is None:
            raise ValueError("RAG dataset is not loaded yet. Please load the dataset first.")
        return len(self.rag_data)

    def load_rag_file(self, file_path: str):
        """
        :param file_path: str: path to the RAG dataset file
        """
        # check the file path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        with open(file_path, "r", encoding="utf-8") as f:
            rag_dataset: List = json.load(f)
            for data in tqdm(rag_dataset, desc=f"Loading RAG dataset from {file_path}"):
                rag_prep_data: RAGPreprocessedDatasetSample = RAGPreprocessedDatasetSample(**data)
                self.rag_data[data["id"]] = rag_prep_data

    def load_rag_files_list(self, file_paths: List[str]):
        """
        :param file_paths: List[str]: list of paths to the RAG dataset files
        """
        for file_path in file_paths:
            self.load_rag_file(file_path)
