# AID-SQL

Recent research in Text-to-SQL translation has primarily adopted in-context learning methods leveraging large language models (LLMs), achieving significant progress. However, these methods face challenges in their adaptability to natural language questions of varying difficulty and the relevance of the examples provided. Consequently, we propose an adaptive in-context learning approach with difficulty-aware instruction and retrieval-augmented generation to enhance the performance of Text-to-SQL translation, which is called AID-SQL.

## Installation

Set up the python environment:
```
conda create -n AID-SQL python=3.8
conda activate AID-SQL
python -m pip install --upgrade pip
pip install -r requirements.txt
python nltk_downloader.py
```

## Dataset Preparation

You need to download the [Spider](https://yale-lily.github.io/spider). Once download, unzip it in the following directory `./dataset/spider-dataset/spider`. If the path does not exist, please create it.

The dataset path looks like
```
dataset/
└── spider-dataset
    └── spider
        ├── README.txt
        ├── database
        ├── dev.json
        ├── dev_gold.sql
        ├── tables.json
        ├── train_gold.sql
        ├── train_others.json
        └── train_spider.json
```

## Data Preprocess

### Dataset Preprocess

Use the following .sh script to preprocess all data.

```
sh scripts/preprocess_dataset.sh
```

### Vector Database Construct

Use the following .sh script to organize the preprocessed dataset into vector indexed database, which will facilitate subsequent retrieval and training tasks.

```
sh scripts/preprocess_rag_dataset.sh

sh scripts/rag/rag_construct/default_euc_sql_spider.sh
```

## Run

Execute the following code to run our main program.
```
python main.py --dataset <dataset> --output predict_output.txt --gold gold_sql.txt
```
Please replace the above `<dataset>` with your database path, and make sure `tables.json` exists in the current path.

## Model Train

Or you can train the category and ranking models from scratch here.

### Train Dataset Preparation

First, reorganize the dataset for ranking model training.

```
python contrast_triples_prepare.py
```

### Ranking Model Train

Use the reorganized dataset to train the ranking model.

```
python train.py
```

