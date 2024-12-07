# AID-SQL

AID-SQL is a [description]


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

