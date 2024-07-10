import json
import os
import torch
import evaluate
import torch
from comet import download_model, load_from_checkpoint
from uptrain import CritiqueTone, EvalLLM, Evals, Settings
from BARTScore.bart_score import BARTScorer
from datetime import datetime

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH")

# Get available gpus from environment variable (e.g. AVAILABLE_GPUS=0,1,2,3,4,5)
AVAILABLE_GPUS = os.environ.get("AVAILABLE_GPUS")
AVAILABLE_GPUS = AVAILABLE_GPUS.split(",")
CURRENT_GPU_INDEX = 0

if len(AVAILABLE_GPUS) != 1:
    # First GPU is always allocated to bleurt (maybe because of Tensorflow)
    AVAILABLE_GPUS = AVAILABLE_GPUS[1:]


def bleurt(predictions, references):
    prediction, reference = predictions[0], json.loads(
        references[0])["reference"]
    return (prediction, reference)

def agg_bleurt(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
    bleurt_fn = evaluate.load('bleurt', 'bleurt-large-512')

    predictions, references = zip(*items)
    output = bleurt_fn.compute(predictions=predictions,
                         references=references)["scores"]
    return sum(output) / len(output)


def bleu(predictions, references):
    try:
        prediction, reference = predictions[0], json.loads(
            references[0])["reference"]
    except:
        print(references[0])
        raise ValueError()
    return (prediction, reference)

def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions,
                           references=references)["bleu"]


def rouge(predictions, references):
    prediction, reference = predictions[0], json.loads(
        references[0])["reference"]
    return (prediction, reference)

def agg_rouge(items):
    rouge_fn = evaluate.load("rouge")
    predictions, references = zip(*items)
    return rouge_fn.compute(predictions=predictions,
                            references=references,
                            use_aggregator=True)["rougeL"]

def bertscore(predictions, references):
    prediction, reference = predictions[0], json.loads(
        references[0])["reference"]
    lang = json.loads(references[0])["target_lang"]

    return (prediction, reference, lang)

def agg_bertscore(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
 
    bs_fn = evaluate.load("bertscore")
    predictions, references, langs = zip(*items)
    output = bs_fn.compute(predictions=predictions,
                          references=references,
                          model_type="bert-base-multilingual-cased",
                          lang=langs[0],
                          device="cuda:" + AVAILABLE_GPUS[CURRENT_GPU_INDEX])["f1"]
    CURRENT_GPU_INDEX = (CURRENT_GPU_INDEX + 1) % len(AVAILABLE_GPUS)
    return sum(output) / len(output)

def cometkiwi22(predictions, references):
    try:
        prediction, source = predictions[0], json.loads(references[0])["source"]
    except:
        print(references[0])
        raise ValueError()

    return (prediction, source)

def agg_cometkiwi22(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
 
    predictions, sources = zip(*items)

    data = []
    for prediction, source in zip(predictions, sources):
        data.append({"src": source, "mt": prediction})

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data,
                                 batch_size=4,
                                 gpus=1,
                                 devices=[int(AVAILABLE_GPUS[CURRENT_GPU_INDEX])])

    CURRENT_GPU_INDEX = (CURRENT_GPU_INDEX + 1) % len(AVAILABLE_GPUS)

    # Example output:
    # Prediction([('scores', [0.8676194548606873]), ('system_score', 0.8676194548606873)])

    return model_output[1]

def cometkiwi23(predictions, references):
    try:
        prediction, source = predictions[0], json.loads(references[0])["source"]
    except:
        print(references[0])
        raise ValueError()

    return (prediction, source)

def agg_cometkiwi23(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
 
    predictions, sources = zip(*items)

    data = []
    for prediction, source in zip(predictions, sources):
        data.append({"src": source, "mt": prediction})

    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data,
                                 batch_size=4,
                                 gpus=1,
                                 devices=[int(AVAILABLE_GPUS[CURRENT_GPU_INDEX])])

    CURRENT_GPU_INDEX = (CURRENT_GPU_INDEX + 1) % len(AVAILABLE_GPUS)

    # Example output:
    # Prediction([('scores', [0.8676194548606873]), ('system_score', 0.8676194548606873)])

    return model_output[1]


def xcomet(predictions, references):
    try:
        prediction, source = predictions[0], json.loads(references[0])["source"]
    except:
        print(references[0])
        raise ValueError()

    return (prediction, source)

def agg_xcomet(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
 
    predictions, sources = zip(*items)

    data = []
    for prediction, source in zip(predictions, sources):
        data.append({"src": source, "mt": prediction})

    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data,
                                 batch_size=4,
                                 gpus=1,
                                 devices=[int(AVAILABLE_GPUS[CURRENT_GPU_INDEX])])

    CURRENT_GPU_INDEX = (CURRENT_GPU_INDEX + 1) % len(AVAILABLE_GPUS)

    # Example output:
    # Prediction([('scores', [0.8676194548606873]), ('system_score', 0.8676194548606873)])

    return model_output[1]


def bartscore_src(predictions, references):
    try:
        prediction, source = predictions[0], json.loads(references[0])["source"]
        src_lang = json.loads(references[0])["source_lang"]
        tgt_lang = json.loads(references[0])["target_lang"]
    except:
        print(references[0])
        raise ValueError()

    return (prediction, source, src_lang, tgt_lang)

def agg_bartscore_src(items):
    global CURRENT_GPU_INDEX
    global AVAILABLE_GPUS
 
    predictions, sources, src_langs, tgt_langs = zip(*items)
    src_lang = src_langs[0]
    tgt_lang = tgt_langs[0]

    # If other language is added, then 
    lang_code_dict = {
        "en": "en_XX",
        "ko": "ko_KR",
        "ja": "ja_XX",
        "zh": "zh_CN"
    }
    device = torch.device("cuda:" + AVAILABLE_GPUS[CURRENT_GPU_INDEX] if torch.cuda.is_available() else "cpu")
    CURRENT_GPU_INDEX = (CURRENT_GPU_INDEX + 1) % len(AVAILABLE_GPUS)
    bart_scorer = BARTScorer(device=device,
                             checkpoint='facebook/mbart-large-50',
                             src_lang=lang_code_dict[src_lang],
                             tgt_lang=lang_code_dict[tgt_lang])
    
    # Calculate probability of predictions when sources are given as context
    scores = bart_scorer.score(sources, predictions)
    
    return sum(scores) / len(scores)


def llm_eval(predictions, references):
    prediction, source = predictions[0], json.loads(references[0])["source"]

    return (prediction, source)


def agg_llm_eval(items):
    predictions, sources = zip(*items)
    data = [{"response": prediction} for prediction in predictions]
    if not AZURE_OPENAI_API_KEY or not AZURE_API_VERSION or not AZURE_ENDPOINT:
        raise ValueError(
            "Please set the environment variables AZURE_OPENAI_API_KEY, AZURE_API_VERSION and AZURE_ENDPOINT"
        )

    eval_llm = EvalLLM(
        Settings(model='azure/gpt-4-turbo',
                 azure_api_key=AZURE_OPENAI_API_KEY,
                 api_version=AZURE_API_VERSION,
                 azure_api_base=AZURE_ENDPOINT))
    response = eval_llm.evaluate(data, checks=[Evals.CRITIQUE_LANGUAGE])
    
    # logging as json
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(OUTPUT_PATH, "llm-eval_input_" + current_time + ".json"), 'w', encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False)
    with open(os.path.join(OUTPUT_PATH, "llm-eval_response_" + current_time + ".json"), 'w', encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False)

    # return sum([resp["score_critique_language"] for resp in response]) /len(response)
    # uptrain 0.5.0 version has 4 type of scores, so gather them and average them
    score_fluency = sum([resp["score_fluency"] if resp["score_fluency"] is not None else 0.0 for resp in response]) /len(response)
    score_grammar = sum([resp["score_grammar"] if resp["score_grammar"] is not None else 0.0 for resp in response]) /len(response)
    score_coherence = sum([resp["score_coherence"] if resp["score_coherence"] is not None else 0.0 for resp in response]) /len(response)
    score_politeness = sum([resp["score_politeness"] if resp["score_politeness"] is not None else 0.0 for resp in response]) /len(response)
    return (score_fluency + score_grammar + score_coherence + score_politeness) / 4 
