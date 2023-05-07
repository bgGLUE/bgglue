import abc
import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from datasets import load_metric

logger = logging.getLogger(__name__)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        else:
            return super(EnhancedJSONEncoder, self).default(obj)


@dataclass
class EvaluationResult:
    sample_size: int
    metrics: Dict[str, Any]


class BaseEvaluator(abc.ABC):
    def __init__(
        self,
        task_name: str,
        prediction_path: Path,
        labels_path: Path,
        metric_json_path: str,
    ) -> None:
        self.task_name = task_name
        self.metric_json_path = metric_json_path
        assert (
            self.metric_json_path
        ), "The json path to the task's main metric should not be empty."
        self._predictions = self._read_predictions(
            prediction_path / self.task_name / "predictions.jsonl"
        )
        self._labels = self._read_labels(labels_path / self.task_name / "labels.txt")
        self._post_processing()

    def _post_processing(self) -> None:
        pass

    def _read_labels(self, path: Path) -> List[Union[int, str, List[Union[int, str]]]]:
        with path.open("r") as fp:
            return [x.strip() for x in fp]

    @abc.abstractmethod
    def _read_predictions(
        self, path: Path
    ) -> List[Union[int, str, List[Union[int, str]]]]:
        pass

    @abc.abstractmethod
    def _compute_metrics(self) -> Dict[str, Any]:
        pass

    def resolve_task_metric(self, metrics) -> float:
        task_metric = metrics
        for json_component in self.metric_json_path.split("."):
            task_metric = task_metric[json_component]

        return task_metric

    def evaluate(self) -> EvaluationResult:
        assert len(self._predictions) == len(self._labels), (
            f"Error on task {self.task_name}: "
            f"Number of predictions ({len(self._predictions)}) is not equal to "
            f"the number of labels ({len(self._labels)}).",
        )

        metrics = self._compute_metrics()
        return EvaluationResult(sample_size=len(self._predictions), metrics=metrics)


class TaggerEvaluator(BaseEvaluator):
    def _post_processing(self) -> None:
        self._labels = self._parse_tags(self._labels)
        # Fill predictions  with 'O' until they reach labels' length
        for p, l in zip(self._predictions, self._labels):
            p += ["O"] * (len(l) - len(p))

    def _parse_tags(self, tags: Iterable[str]) -> Iterable[Any]:
        return [x.strip().split(" ") for x in tags]

    def _read_predictions(self, path: Path) -> Iterable[Any]:
        return self._parse_tags(pd.read_json(str(path), lines=True)["label"].tolist())

    def _compute_metrics(self) -> Dict[str, Any]:
        metric = load_metric("seqeval")

        return metric.compute(predictions=self._predictions, references=self._labels)


class ClassificationEvaluator(BaseEvaluator):
    def _post_processing(self) -> None:
        self._label_names = list(sorted(set(self._labels)))
        self._labels2id = {l: i for i, l in enumerate(self._label_names)}
        self._predictions = [self._labels2id[x] for x in self._predictions]
        self._labels = [self._labels2id[x] for x in self._labels]

    def _read_classes(self, path: Path) -> Iterable[Any]:
        return pd.read_json(str(path), lines=True)["label"].tolist()

    def _read_predictions(self, path: Path) -> Iterable[Any]:
        return self._read_classes(path)

    def _compute_metrics(self) -> Dict[str, Any]:
        return classification_report(
            self._labels,
            self._predictions,
            target_names=self._label_names,
            output_dict=True,
        )


class XNLIEvaluator(ClassificationEvaluator):
    def _compute_metrics(self) -> Dict[str, Any]:
        metric = load_metric("xnli")

        return metric.compute(predictions=self._predictions, references=self._labels)


class ExamsEvaulator(ClassificationEvaluator):
    def _compute_metrics(self) -> Dict[str, Any]:
        metric = load_metric("accuracy")

        return metric.compute(predictions=self._predictions, references=self._labels)


class CheckThatCheckworthyEvaluator(BaseEvaluator):
    def _read_labels(self, path: Path) -> List[Union[int, str, List[Union[int, str]]]]:
        return (
            pd.read_csv(
                str(path), sep="\t", names=["topic_id", "id_str", "check_worthy"]
            )["check_worthy"]
            .astype(int)
            .tolist()
        )

    def _read_predictions(self, path: Path) -> Iterable[Any]:
        return pd.read_json(path, lines=True)["label"].astype(float)

    def _compute_metrics(self) -> Dict[str, Any]:
        metric = load_metric(
            str(pathlib.Path(__file__).parent.resolve() / "metrics_clef.py")
        )

        return metric.compute(predictions=self._predictions, references=self._labels)


class CinexioEvaluator(BaseEvaluator):
    def _post_processing(self) -> None:
        self._labels = [float(x) for x in self._labels]

    def _read_predictions(self, path: Path) -> Iterable[Any]:
        return pd.read_json(str(path), lines=True)["label"].astype(float).tolist()

    def _compute_metrics(self) -> Dict[str, Any]:
        spearmanr_metric = load_metric("spearmanr")
        pearsonr_metric = load_metric("pearsonr")
        pearsonr = pearsonr_metric.compute(
            predictions=self._predictions, references=self._labels
        )
        spearmanr = spearmanr_metric.compute(
            predictions=self._predictions, references=self._labels
        )
        return {
            # We want to ignore ignore negative values in the correlation
            "sp_correlation": max(pearsonr["pearsonr"] + spearmanr["spearmanr"], 0) / 2,
            **pearsonr,
            **spearmanr,
        }


EVALUATORS = {
    "bsnlp": partial(TaggerEvaluator, metric_json_path="overall_f1"),
    "cinexio": partial(CinexioEvaluator, metric_json_path="sp_correlation"),
    "ct21t1": partial(CheckThatCheckworthyEvaluator, metric_json_path="avg_precision"),
    "crediblenews": partial(
        ClassificationEvaluator, metric_json_path="humorous.f1-score"
    ),
    "examsbg": partial(ExamsEvaulator, metric_json_path="accuracy"),
    "fakenews": partial(ClassificationEvaluator, metric_json_path="fake.f1-score"),
    "udep": partial(TaggerEvaluator, metric_json_path="overall_f1"),
    "xnlibg": partial(XNLIEvaluator, metric_json_path="accuracy"),
    "wikiannbg": partial(TaggerEvaluator, metric_json_path="overall_f1"),
}


def main(evaluation_args: argparse.Namespace) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO if evaluation_args.verbose else logging.ERROR
    logger.setLevel(log_level)

    logger.info("Evaluation parameters %s", evaluation_args)

    per_task_results = {}
    task_metric_names = {}
    results = {
        "bgglue_score": {"overall_score": 0.0, "task_scores": {}},
        "evaluation_details": {
            "model_name": evaluation_args.model_name,
            "eval_subset": evaluation_args.evaluation_subset,
            "task_metrics": task_metric_names,
        },
        "per_task_metrics": per_task_results,
    }

    for task_name in evaluation_args.evaluation_tasks:
        logger.info("Processing %s...", task_name)
        try:
            evaluator: BaseEvaluator = EVALUATORS[task_name](
                task_name=task_name,
                prediction_path=evaluation_args.prediction_path,
                labels_path=evaluation_args.labels_path,
            )

            eval_result = evaluator.evaluate()
            per_task_results[task_name] = eval_result
            task_score = evaluator.resolve_task_metric(eval_result.metrics)
            results["bgglue_score"]["task_scores"][task_name] = task_score
            task_metric_names[task_name] = evaluator.metric_json_path
        except Exception as e:
            logger.error("Error when preparing evaluator for task %s:", task_name)
            logger.exception(e)
            raise
        logger.info(
            "Task metric (%s): %.4f, samples: %d",
            evaluator.metric_json_path,
            task_score,
            eval_result.sample_size,
        )

    results["bgglue_score"]["overall_score"] = np.fromiter(
        results["bgglue_score"]["task_scores"].values(), dtype=float
    ).mean()

    metrics_json = json.dumps(
        results, ensure_ascii=False, indent=4, sort_keys=True, cls=EnhancedJSONEncoder
    )
    logger.info(metrics_json)

    if evaluation_args.output_file:
        evaluation_args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with evaluation_args.output_file.open("w") as fp:
            fp.write(metrics_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_path",
        default=None,
        type=Path,
        required=True,
        help="The predictions of one model",
    )
    parser.add_argument(
        "--labels_path",
        default=None,
        type=Path,
        required=True,
        help="The ground truth file",
    )
    parser.add_argument(
        "--evaluation_tasks",
        default="all",
        choices=[*EVALUATORS.keys(), "all"],
        help="List of tasks to evaluate",
        nargs="+",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="whether to print details"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=Path,
        required=False,
        help="The output file where the report is generated",
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        required=True,
        help="The name of the model that is being evaluated",
    )
    parser.add_argument(
        "--evaluation_subset",
        default="test",
        choices=["validation", "test"],
        type=str,
        required=False,
        help="The name of the evaluation subset",
    )
    args = parser.parse_args()
    if "all" in args.evaluation_tasks:
        args.evaluation_tasks = EVALUATORS.keys()

    main(args)
