import json
import os

import evaluate
from uptrain import EvalLLM, Evals

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def bleu(predictions, references):
    return (predictions[0], references[0])


def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions,
                           references=references)["bleu"]


def bleurt(predictions, references):
    bleurt_fn = evaluate.load("bleurt")
    return bleurt_fn.compute(predictions=predictions,
                             references=references)["scores"][0]


def llm_eval(predictions, references):
    data = [{"response": predictions[0]}]

    eval_llm = EvalLLM(openai_api_key=OPENAI_API_KEY)
    response = eval_llm.evaluate(data, checks=[Evals.CRITIQUE_LANGUAGE])
    return response[0]["score_fluency"]
