set -e

# proprecess spider train dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./dataset/spider-dataset/spider/tables.json" \
    --input_dataset_path "./dataset/spider-dataset/spider/train_spider.json" \
    --output_dataset_path "./preprocessed_data/preprocessed_train_spider.json" \
    --db_path "./dataset/spider-dataset/spider/database" \
    --target_type "sql" \
    --dataset_name "spider"

# preprocess spider dev dataset
python preprocessing.py \
    --mode "train" \
    --table_path "./dataset/spider-dataset/spider/tables.json" \
    --input_dataset_path "./dataset/spider-dataset/spider/dev.json" \
    --output_dataset_path "./preprocessed_data/preprocessed_dev_spider.json" \
    --db_path "./dataset/spider-dataset/spider/database" \
    --target_type "sql" \
    --dataset_name "spider"


# preprocess spider test dataset
python preprocessing.py \
    --mode "train" \
    --table_path "dataset/spider-dataset/spider_test/test_tables.json" \
    --input_dataset_path "dataset/spider-dataset/spider_test/test.json" \
    --output_dataset_path "./preprocessed_data/spider_test/preprocessed_test_spider.json" \
    --db_path "dataset/spider-dataset/spider_test/test_database" \
    --target_type "sql" \
    --dataset_name "spider_test"






