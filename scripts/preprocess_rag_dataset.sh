set -e

# preprocess spider train dataset to RAG format
# input needs to be preprocessed type
python rag_dataset_preprocess.py \
    --input_dataset_path "./preprocessed_data/preprocessed_train_spider.json" \
    --output_dataset_path "./preprocessed_data/rag_preprocessed_data/rag_preprocessed_spider_train.json" \




