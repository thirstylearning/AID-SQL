# encoding=utf8
import json
import os
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric


class EvaluateTool(object):
    def __init__(self):
        # self.args = args
        self.schema_cache = dict()
        self.golds = []

    def register_golds(self, dataset_filepath, db_path):
        with open(dataset_filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for idx, sample in enumerate(dataset):
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                    )
                schema = self.schema_cache[db_id]

                self.golds.append({
                    "query": sample["sql"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": {
                        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                        "column_name": [column_name for _, column_name in schema["column_names_original"]]
                    },
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": {
                        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
                    },
                })

    def evaluate(self, preds):
        exact_match = compute_exact_match_metric(preds, self.golds)
        test_suite = compute_test_suite_metric(preds, self.golds, db_dir=None)

        return {**exact_match, **test_suite}
