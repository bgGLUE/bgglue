from typing import List, Tuple

import datasets


_CITATION = """\
@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""

_DESCRIPTION = """\
XNLI is a subset of a few thousand examples from MNLI which has been translated
into a 14 different languages (some low-ish resource). As with MNLI, the goal is
to predict textual entailment (does sentence A imply/contradict/neither sentence
B) and is a classification task (given two sentences, predict one of three
labels).
"""

_KWARGS_DESCRIPTION = """
Computes XNLI score which is just simple accuracy.
Args:
    predictions: Predicted labels.
    references: Ground truth labels.
Returns:
    'accuracy': accuracy
Examples:
    >>> predictions = [0, 1]
    >>> references = [0, 1]
    >>> xnli_metric = datasets.load_metric("xnli")
    >>> results = xnli_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}
"""

MAIN_THRESHOLDS = (1, 3, 5, 10, 20, 50)


def _compute_average_precision(gold_labels, ranked_lines):
    """ Computes Average Precision. """

    precisions = []
    num_correct = 0
    num_positive = sum(gold_labels)

    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            num_correct += 1
            precisions.append(num_correct / (i + 1))
    if precisions:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec


def _compute_reciprocal_rank(gold_labels, ranked_lines):
    """ Computes Reciprocal Rank. """
    rr = 0.0
    for i, line_number in enumerate(ranked_lines):
        if gold_labels[line_number] == 1:
            rr += 1.0 / (i + 1)
            break
    return rr


def _compute_precisions(gold_labels, ranked_lines, threshold):
    """ Computes Precision at each line_number in the ordered list. """
    precisions = [0.0] * threshold
    threshold = min(threshold, len(ranked_lines))

    for i, line_number in enumerate(ranked_lines[:threshold]):
        if gold_labels[line_number]:
            precisions[i] += 1.0

    for i in range(1, threshold): # accumulate
        precisions[i] += precisions[i - 1]
    for i in range(1, threshold):  # accumulate
        precisions[i] /= i+1
    return precisions


class ClefT1(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("int32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references, thresholds: Tuple[int] = MAIN_THRESHOLDS):
        ranked_lines = [t[0] for t in sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)]
        if thresholds is None or len(thresholds) == 0:
            thresholds = list(thresholds) + [len(references)]

        # Calculate Metrics
        precisions = _compute_precisions(references, ranked_lines, len(ranked_lines))
        avg_precision = _compute_average_precision(references, ranked_lines)
        reciprocal_rank = _compute_reciprocal_rank(references, ranked_lines)
        num_relevant = len({k for k, v in enumerate(references) if v == 1})

        return {"avg_precision": avg_precision,
                "reciprocal_rank": reciprocal_rank,
                "num_relevant": num_relevant,
                "r_precision": precisions[num_relevant - 1],
                **{f"P@{th}": precisions[th - 1] for th in thresholds}}