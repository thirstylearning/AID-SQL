from typing import List, Union, Callable, Optional
import numpy as np
import os
import json

from numpy import ndarray
from tqdm import tqdm

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics.pairwise import cosine_similarity

from prompt.ExampleSelectorTemplate import (
    DefEmbSimiSQLEucCheatSelectorSPIDER,
    DefEmbSimiSQLEucCheatSelectorBIRD,
    FTColbertRerankQues2SQLAll2smallSelector,
    FTColbertRerankQuesEvid2SQLAll2smallSelector,
    FTColbertRerankQues2QuesSQLAll2smallSelector,
    FTColbertRerankQuesSQL2SQLAll2smallSelector,
    FTColbertRerankQuesSQL2QuesSQLAll2smallSelector,
    FTColbertRerankQues2SQLPredTypeFilterAll2smallSelector,
    FTColbertRerankQuesSQL2SQLPredTypeFilterAll2smallSelector,
    FTColbertRerankQuesEvidSQL2SQLAll2smallSelector,
)
from training.my_batcher import MyBatcher
from utils.schemas import RAGPreprocessedDatasetSample, PreprocessedDatasetSample
from utils.utils import load_preprocessed_dataset
from utils.enums import *


from third_party.colbert_v1.ColBERT_v1.colbert.infra import ColBERTConfig


get_query = lambda sample: sample.question


def_embedding_obj = ONNXMiniLM_L6_V2()


def embedding_call(samples: Union[str, List[str]]) -> np.ndarray:
    if isinstance(samples, str):
        samples = [samples]

    return def_embedding_obj._forward(samples)


def calculate_avg_l2_distance(query_emb: np.ndarray, tar_embs: np.ndarray) -> float:

    def l2_distance(emb1: np.ndarray, emb2: np.ndarray):
        return np.linalg.norm(emb1 - emb2)

    distances = [l2_distance(query_emb[0], _tar_emb) for _tar_emb in tar_embs]
    return np.mean(distances)


calculate_avg_distance = calculate_avg_l2_distance


def evaluate_epoch_ckpt_in_dir(
    config: ColBERTConfig,
    eval_dataset_name: str,
    epochs_dir: str,
    train_samples_path: Optional[str] = None,
    eval_on_train: bool = False,
    use_all: bool = True,
    left_formatter_type: str = "question",
    right_formatter_type: str = "sql",
    id_predsql_dict_path: str = "",
    predsql_method: str = "resdsql-3b",
    use_pred_sql_type_filter: bool = False,
    pred_sql_type_dict_path: str = "",
):

    sub_dirs_names: List[str] = [
        _name
        for _name in os.listdir(epochs_dir)
        if os.path.isdir(os.path.join(epochs_dir, _name)) and ("epoch" in _name)
    ]
    sub_dirs_names.sort(key=lambda x: int(x.split("_")[-1]))

    tb_writer = SummaryWriter(log_dir=os.path.join(epochs_dir, "tensorboard", "eval"))

    epochs_eval_res = []
    epochs_train_eval_res = []

    for epoch_sub_dir_name in sub_dirs_names:
        modeal_save_path = os.path.join(epochs_dir, epoch_sub_dir_name, "checkpoints", "colbert")

        epoch_idx: int = int(epoch_sub_dir_name.split("_")[-1])

        _g, _e, _diff, _cos_simi, _g_recall, top10_recall = evaluate(
            config=config,
            use_all=use_all,
            eval_dataset_name=eval_dataset_name,
            model_save_path=modeal_save_path,
            left_formatter_type=left_formatter_type,
            right_formatter_type=right_formatter_type,
            id_predsql_dict_path=id_predsql_dict_path,
            predsql_method=predsql_method,
            use_pred_sql_type_filter=use_pred_sql_type_filter,
            pred_sql_type_dict_path=pred_sql_type_dict_path,
        )

        tb_writer.add_scalar("eval/diff", _diff, epoch_idx)
        tb_writer.add_scalar("eval/cos_simi", _cos_simi, epoch_idx)
        tb_writer.add_scalar("eval/golden_recall", _g_recall, epoch_idx)
        tb_writer.add_scalar("eval/top10_recall", top10_recall, epoch_idx)

        epochs_eval_res.append(
            {"epoch_idx": epoch_idx, "eval_distance": _e, "golden_distance": _g, "diff": _diff, "cos_simi": _cos_simi}
        )

        if eval_on_train and train_samples_path is not None:
            _g, _e, _diff, _ = evaluate(
                config=config,
                use_all=use_all,
                eval_dataset_name=eval_dataset_name,
                model_save_path=modeal_save_path,
                eval_on_train=True,
            )

            tb_writer.add_scalar("train/diff", _diff, epoch_idx)
            epochs_train_eval_res.append(
                {"epoch_idx": epoch_idx, "eval_distance": _e, "golden_distance": _g, "diff": _diff}
            )

    tb_writer.close()
    eval_res_path = os.path.join(epochs_dir, "eval_res.json")
    train_eval_res_path = os.path.join(epochs_dir, "train_eval_res.json")

    with open(eval_res_path, "w") as f:
        f.write(json.dumps(epochs_eval_res, indent=4))

    with open(train_eval_res_path, "w") as f:
        f.write(json.dumps(epochs_train_eval_res, indent=4))


def specify_eval_samples_by_db_name(
    eval_samples: List[PreprocessedDatasetSample], db_name: Union[str, List[str]]
) -> List[PreprocessedDatasetSample]:
    if isinstance(db_name, str):
        db_name = [db_name]
    return [sample for sample in eval_samples if sample.db_id in db_name]


def recall_count(evals: List[RAGPreprocessedDatasetSample], goldens: List[RAGPreprocessedDatasetSample]) -> int:
    """
    count sql hits
    """
    assert len(evals) == len(goldens), (len(evals), len(goldens))
    eval_sqls = [eval_.sql for eval_ in evals]
    golden_sqls = [golden.sql for golden in goldens]

    count_res = 0
    for eval_sql in eval_sqls:
        if eval_sql in golden_sqls:
            count_res += 1

    return count_res


def get_golden_selector(eval_dataset_name: str):
    if eval_dataset_name == "spider" or eval_dataset_name == "spider_test":
        return DefEmbSimiSQLEucCheatSelectorSPIDER(rag_datasets=["spider"])
    elif eval_dataset_name == "bird":
        return DefEmbSimiSQLEucCheatSelectorBIRD(rag_datasets=["bird"])
    else:
        raise ValueError(f"invalid eval dataset name: {eval_dataset_name}")


def colbert_reranker_factory(
    modeal_save_path: str,
    left_formatter_type: str,
    right_formatter_type: str,
    pre_sql_path: str,
    presql_method: str,
    use_pred_sql_type_filter: bool,
    id_pred_sql_type_dict_path: str,
    eval_dataset_name: str,
):
    if left_formatter_type == "question":
        if right_formatter_type == "sql":
            if use_pred_sql_type_filter:
                return FTColbertRerankQues2SQLPredTypeFilterAll2smallSelector(
                    model_save_path=modeal_save_path,
                    rag_datasets=[eval_dataset_name],
                    pred_sql_type_path=id_pred_sql_type_dict_path,
                )
            else:
                return FTColbertRerankQues2SQLAll2smallSelector(
                    model_save_path=modeal_save_path, rag_datasets=[eval_dataset_name]
                )
        elif right_formatter_type == "question_sql":
            return FTColbertRerankQues2QuesSQLAll2smallSelector(
                model_save_path=modeal_save_path, rag_datasets=[eval_dataset_name]
            )
        else:
            raise ValueError(f"invalid right formatter type: {right_formatter_type}")
    elif left_formatter_type == "question_sql":
        if right_formatter_type == "sql":
            if use_pred_sql_type_filter:
                return FTColbertRerankQuesSQL2SQLPredTypeFilterAll2smallSelector(
                    model_save_path=modeal_save_path,
                    presql_method=presql_method,
                    pre_sql_path=pre_sql_path,
                    pred_sql_type_path=id_pred_sql_type_dict_path,
                    rag_datasets=[eval_dataset_name],
                )
            else:
                return FTColbertRerankQuesSQL2SQLAll2smallSelector(
                    model_save_path=modeal_save_path,
                    rag_datasets=[eval_dataset_name],
                    pre_sql_path=pre_sql_path,
                    presql_method=presql_method,
                )
        elif right_formatter_type == "question_sql":
            return FTColbertRerankQuesSQL2QuesSQLAll2smallSelector(
                model_save_path=modeal_save_path,
                rag_datasets=[eval_dataset_name],
                pre_sql_path=pre_sql_path,
                presql_method=presql_method,
            )
        else:
            raise ValueError(f"invalid right formatter type {right_formatter_type}")
    elif left_formatter_type == "question_evidence":
        assert eval_dataset_name == "bird", "only support bird dataset"
        if right_formatter_type == "sql":
            return FTColbertRerankQuesEvid2SQLAll2smallSelector(
                model_save_path=modeal_save_path, rag_datasets=[eval_dataset_name]
            )
        else:
            raise ValueError(f"invalid right formatter type: {right_formatter_type}")
    elif left_formatter_type == "question_evidence_sql":
        assert eval_dataset_name == "bird", "only support bird dataset"
        if right_formatter_type == "sql":
            return FTColbertRerankQuesEvidSQL2SQLAll2smallSelector(
                model_save_path=modeal_save_path, rag_datasets=[eval_dataset_name],
                pre_sql_path=pre_sql_path,
                presql_method=presql_method,
            )
        else:
            raise ValueError(f"invalid right formatter type: {right_formatter_type}")
    else:
        raise ValueError(f"invalid left formatter type: {left_formatter_type}")


def get_eval_dataset_path(eval_dataset_name: str) -> str:
    if eval_dataset_name == "spider":
        return PREPROCESSED_DATASET_PATH.SPIDER_DEV
    elif eval_dataset_name == "spider_test":
        return PREPROCESSED_DATASET_PATH.SPIDER_TEST
    elif eval_dataset_name == "bird":
        return PREPROCESSED_DATASET_PATH.BIRD_DEV
    else:
        raise ValueError(f"invalid eval dataset name: {eval_dataset_name}")


def evaluate(
    config: ColBERTConfig,
    eval_dataset_name: str,
    model_save_path: str,
    eval_on_train: bool = False,
    use_all: bool = True,
    get_query: Callable = get_query,
    embedding_call: Callable = embedding_call,
    calculate_avg_distance: Callable = calculate_avg_distance,
    top_k=5,
    left_formatter_type: str = "question",
    right_formatter_type: str = "sql",
    id_predsql_dict_path: str = "",
    predsql_method: str = "resdsql-3b",
    use_pred_sql_type_filter: bool = False,
    pred_sql_type_dict_path: str = "",
    save_predictions: bool = False,
):
    if config.rank < 1:
        config.help()

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    eval_samples_path: str = get_eval_dataset_path(eval_dataset_name)
    eval_dataset_name = "spider" if eval_dataset_name == "spider_test" else eval_dataset_name

    print(f"Using dev set from {eval_samples_path}")

    if eval_samples_path is not None:
        eval_reader = load_preprocessed_dataset(eval_samples_path)

        if not use_all and eval_dataset_name == "spider":
            # eval_reader = eval_reader[:400]
            eval_reader = specify_eval_samples_by_db_name(eval_reader, ["wta_1", "poker_player", "world_1"])
    else:
        raise NotImplementedError()

    golden_selector = get_golden_selector(eval_dataset_name=eval_dataset_name)

    eval_reranker = colbert_reranker_factory(
        model_save_path,
        left_formatter_type,
        right_formatter_type,
        id_predsql_dict_path,
        predsql_method,
        use_pred_sql_type_filter,
        pred_sql_type_dict_path,
        eval_dataset_name,
    )
    print(f"using reranker type is {eval_reranker.__class__.__name__}")

    prediction_dict = {}

    total_golden_distance_sum: float = 0.0
    total_eval_distance_sum: float = 0.0
    total_diff_sum: float = 0.0

    eval_sample: PreprocessedDatasetSample
    cos_avg_simi_list: List[float] = []
    golden_recall_count_list: List[int] = []
    golden_recall_top10_count_list: List[int] = []
    for eval_sample in tqdm(eval_reader, desc="evaluating on dev set"):
        query_top_k: int = 5
        n_goldens: List[RAGPreprocessedDatasetSample] = golden_selector.get_examples(
            eval_sample, examples_num=query_top_k
        )
        n_evals: List[RAGPreprocessedDatasetSample] = eval_reranker.get_examples(eval_sample, examples_num=query_top_k)

        assert query_top_k >= 5, "query_top_k should be at least 5"
        top5_goldens = n_goldens[:5]
        top5_evals = n_evals[:5]
        top5_recall_count = recall_count(top5_evals, top5_goldens)

        prediction_dict[eval_sample.id] = [_eval.id for _eval in top5_evals]

        query_top_k = 10
        assert query_top_k >= 10, "query_top_k should be at least 10"
        n_evals = eval_reranker.get_examples(eval_sample, examples_num=query_top_k)
        n_goldens = golden_selector.get_examples(eval_sample, examples_num=query_top_k)
        top10_goldens = n_goldens[:10]
        top10_evals = n_evals[:10]
        top10_recall_count = recall_count(top10_evals, top10_goldens)

        golden_recall_count_list.append(top5_recall_count)
        golden_recall_top10_count_list.append(top10_recall_count)

        query_sql_emb: np.ndarray = embedding_call(eval_sample.sql)
        golden_sqls_embs: np.ndarray = embedding_call([_example.sql for _example in top5_goldens])
        eval_sqls_embs: np.ndarray = embedding_call([_example.sql for _example in top5_evals])
        if eval_on_train:
            golden_sqls_embs, eval_sqls_embs = golden_sqls_embs[1:], eval_sqls_embs[1:]

        avg_golden_distance: float = calculate_avg_distance(query_sql_emb, golden_sqls_embs)
        avg_eval_distance: float = calculate_avg_distance(query_sql_emb, eval_sqls_embs)

        distance_diff: float = avg_eval_distance - avg_golden_distance

        total_golden_distance_sum += avg_golden_distance
        total_eval_distance_sum += avg_eval_distance
        total_diff_sum += distance_diff

        cos_avg_simi = np.mean(cosine_similarity(query_sql_emb, eval_sqls_embs)[0])
        # only for test
        # cos_avg_simi = np.mean(cosine_similarity(query_sql_emb, golden_sqls_embs)[0])
        cos_avg_simi_list.append(cos_avg_simi)

    total_avg_golden_distance = total_golden_distance_sum / len(eval_reader)
    total_avg_eval_distance = total_eval_distance_sum / len(eval_reader)
    total_avg_diff = total_diff_sum / len(eval_reader)

    total_avg_cos_simi = float(np.mean(cos_avg_simi_list))
    total_avg_golden_recall = float(np.mean(np.array(golden_recall_count_list)))
    total_avg_golden_top10_recall = float(np.mean(np.array(golden_recall_top10_count_list)))

    if save_predictions:
        predictions_save_path = os.path.join(os.path.dirname(model_save_path), "predictions.json")

        with open(predictions_save_path, "w", encoding="utf-8") as f:
            json.dump(prediction_dict, f, indent=4)

    print_eval_message(
        model_save_path,
        total_avg_golden_distance,
        total_avg_eval_distance,
        total_avg_diff,
        total_avg_cos_simi,
        total_avg_golden_recall,
        total_avg_golden_top10_recall,
        eval_on_train,
    )

    return (
        total_avg_golden_distance,
        total_avg_eval_distance,
        total_avg_diff,
        total_avg_cos_simi,
        total_avg_golden_recall,
        total_avg_golden_top10_recall,
    )


def print_eval_message(
    model_ckpt_path: str,
    avg_golden_distance: float,
    avg_eval_distance: float,
    avg_diff: float,
    avg_cos_simi: float,
    avg_golden_recall: float,
    avg_golden_recall_top10: float,
    eval_on_train: bool = False,
):
    print(f"evaluated model from {model_ckpt_path}")
    dev_train_msg: str = f"evaluated on {'train' if eval_on_train else 'dev'} set"
    print(dev_train_msg)

    print("\t\t" + f"average golden distance: {avg_golden_distance}")
    print("\t\t" + f"average eval distance: {avg_eval_distance}")
    print("\t\t" + f"difference between golden and eval: {avg_diff}")
    print("\t\t" + f"average cosine similarity: {avg_cos_simi}")
    print("\t\t" + f"average golden recall count: {avg_golden_recall}")
    print("\t\t" + f"average golden recall count top10: {avg_golden_recall_top10}")
