import json
import os

import evaluate
from comet import download_model, load_from_checkpoint
from uptrain import CritiqueTone, EvalLLM, Evals, Settings

AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")


def bleu(predictions, references):
    prediction, reference = predictions[0], json.loads(
        references[0])["reference"]

    bleu_fn = evaluate.load("bleu")
    return bleu_fn.compute(predictions=[prediction],
                           references=[reference])["bleu"]


def xcomet(predictions, references):
    prediction, source = predictions[0], json.loads(references[0])["source"]

    data = [{"src": source, "mt": prediction}]

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data, batch_size=8, gpus=0)

    # Example output:
    # Prediction([('scores', [0.8676194548606873]), ('system_score', 0.8676194548606873)])
    return model_output[0][0]


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
