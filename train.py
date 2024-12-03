import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

sys.path.append("third_party/colbert_v1/ColBERT_v1")

from training.trainer import Trainer


from infra.run import Run
from infra.config import ColBERTConfig, RunConfig

n_epochs: int = 40
n_ranks = 4

raw_colbert_path = "./model/colbert/colbertv2.0"

def train():
    # use 4 gpus 
    with Run().context(RunConfig(nranks=n_ranks)):
        # triples = '/path/to/examples.64.json'  # `wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true` (26GB)
        spider_triples = "preprocessed_data/contrast_triples/examples_merged_difficulty"

        bird_triples = "preprocessed_data/contrast_triples/bird/examples_merged_difficulty"
        # queries = '/path/to/MSMARCO/queries.train.tsv'
        spider_rag_collection_path = 'vectorDB/my_collection_settings/default_EUCLIDEAN_SQL_spider/setting.json'
        spider_rag_dataset_path = 'preprocessed_data/rag_preprocessed_data/rag_preprocessed_spider_train.json'

        bird_rag_collection_path = "vectorDB/my_collection_settings/default_EUCLIDEAN_SQL_bird/setting.json"
        bird_rag_dataset_path = "preprocessed_data/rag_preprocessed_data/rag_preprocessed_bird_train.json"

        config = ColBERTConfig(
            bsize=16,
            lr=1e-05, # start from 0 epoch training
            # lr=0.5e-05, # for continue training
            warmup=1_000,
            doc_maxlen=180,
            dim=128,
            attend_to_mask_tokens=False,
            nway=32,
            easy_nway=5,
            accumsteps=1,
            similarity="cosine",
            use_ib_negatives=False,
            use_self_negatives=False,
            use_self_easy_negatives=False,
            use_scores_distillation=True,
            maxsteps=150_000, # 50_000 step = 20 epochs (with 16 batch size, 32 nway, for spider train),
            nranks=n_ranks,
            save_each_n_epoch=1,
            # start_epoch=20,
            query_str_formatter_type="question",
            doc_str_formatter_type="sql",
        )
        trainer = Trainer(
            triples=bird_triples, rag_collection_path=bird_rag_collection_path, rag_dataset_path=bird_rag_dataset_path, config=config,
        )

        trainer.train(checkpoint=raw_colbert_path, n_epochs=n_epochs)  # or start from scratch, like `bert-base-uncased`


if __name__ == "__main__":
    train()
