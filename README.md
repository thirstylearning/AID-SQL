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

Use the category model and ranking model to get category predictions and ranked few-shot examples previously.

For category prediction, run the following script.

```
sh scripts/predict/category_predict.sh
```

For few-shot example ranking, run the following script.

```
sh script/predict/few_shot_rank.sh
```

Execute the following command to run the main generation program.
```
python main.py 
    --dataset <dataset> 
    --output predict_output.txt 
    --gold gold_sql.txt
    --category_prediction pre_sqls/pred_sql_type/dev_spider_pred_sql_type.json
    --few_shot_ranked pre_sqls/few_shot_ranked/filter/few_shot_ranked.json
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

