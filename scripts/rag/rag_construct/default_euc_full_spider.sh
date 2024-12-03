set -e

python rag_db_construct.py \
  --dataset spider \
  --embedding_method default \
  --similarity_method EUCLIDEAN \
  --used_embedding_fields FULL