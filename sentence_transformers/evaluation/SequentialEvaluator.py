from . import SentenceEvaluator
from typing import Iterable, Union, List, Tuple


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    If `return_all_scores` is passed, a list of the list of all scores per evaluator is returned.
    Otherwise, each evaluator returns one main score. Those are passed to 'main_score_function', which derives
    one final score value.
    """
    def __init__(self, evaluators: Iterable[SentenceEvaluator], main_score_function=lambda scores: scores[-1]):
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1,
                 return_all_scores: bool = False) -> Union[Tuple[float, dict], float]:
        main_scores = []
        all_scores = {}
        if return_all_scores:
            for i, evaluator in enumerate(self.evaluators):
                main_score, all_score = evaluator(model, output_path, epoch, steps, return_all_scores)
                main_scores.append(main_score)
                all_scores[i] = all_score
        else:
            for i, evaluator in enumerate(self.evaluators):
                main_scores.append(evaluator(model, output_path, epoch, steps, return_all_scores))

        if return_all_scores:
            return self.main_score_function(main_scores), all_scores
        else:
            return self.main_score_function(main_scores)
