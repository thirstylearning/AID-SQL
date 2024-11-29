"""Spider Test Suite Execution Accuracy metric."""
import logging
from typing import Optional, Dict, Any, List
from collections import defaultdict
from third_party.test_suite import evaluation as test_suite_evaluation

logger = logging.getLogger(__name__)


def compute_test_suite_metric(predictions, references, db_dir: Optional[str] = None) -> Dict[str, Any]:
    if db_dir is None:
        db_dir = references[0]["db_path"]
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )

    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=foreign_key_maps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}
    tag_correct_count: Dict[str, int] = defaultdict(int)
    tag_total_count: Dict[str, int] = defaultdict(int)
    spider_crt_cnt: Dict[str, int] = defaultdict(int)
    spider_total_cnt: Dict[str, int] = defaultdict(int)
    error_ids = []
    for idx, (prediction, reference) in enumerate(zip(predictions, references)):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            _ = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )

            sql_types: List[str] = [reference["difficulty_types"]]
            # TODO: change preprocessed data structure
            spider_total_cnt[reference["spider_difficulty"]] += 1
            for tag_key in sql_types:
                tag_total_count[tag_key] += 1
            # count tag correct
            if _['exec'] == 1:  # means exec equivalent
                # TODO: change preprocessed data structure
                spider_crt_cnt[reference["spider_difficulty"]] += 1
                for tag_key in sql_types:
                    tag_correct_count[tag_key] += 1
            else:  # == 0
                error_ids.append(idx)

        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
    evaluator.finalize()
    return {
        "scores": evaluator.scores,
        "exec": evaluator.scores["all"]["exec"],
        "tag_correct_count": tag_correct_count,
        "tag_total_count": tag_total_count,
        "spider_correct_count": spider_crt_cnt,
        "spider_total_count": spider_total_cnt,
        "error_ids": error_ids,
    }
