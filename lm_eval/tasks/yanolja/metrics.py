import json
import os

import evaluate
from uptrain import CritiqueTone, EvalLLM, Evals, Settings

AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")


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

    if not AZURE_API_KEY or not AZURE_API_VERSION or not AZURE_ENDPOINT:
        raise ValueError(
            "Please set the environment variables AZURE_API_KEY, AZURE_API_VERSION and AZURE_ENDPOINT"
        )

    eval_llm = EvalLLM(
        Settings(model='azure/gpt-4-turbo',
                 azure_api_key=AZURE_API_KEY,
                 api_version=AZURE_API_VERSION,
                 azure_api_base=AZURE_ENDPOINT))
    response = eval_llm.evaluate(data, checks=[Evals.CRITIQUE_LANGUAGE])
    return response[0]["score_critique_language"]
