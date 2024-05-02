import json
import os
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from datetime import datetime
from openai import AzureOpenAI
from macros import (
    TRANSLATION_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    TRANSLATION_PROMPT,
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_ASPECT_PROMPTS,
    TRANSLATION_ASPECT_PROMPTS
)

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH")

# client endpoint and model name can be changed with vllm.
client = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
)

def get_scores(aspect_prompts, system_prompt, user_prompt, source, prediction, model):
    scores = {}
    for aspect, aspect_prompt in aspect_prompts.items():
        system_prompt = system_prompt.format(aspect_prompt=aspect_prompt)
        user_prompt = user_prompt.format(source=source, prediction=prediction)

        # print(SCHEMA)
        chat_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = chat_response.choices[0].message.content
        score = re.search(r"<score>(\d+)</score>", response)
        print(response)
        if score:
            extracted_score = score.group(1)
        else:
            print("No score found")
            score = 0

        scores[aspect] = extracted_score

    return scores

def llm_eval_summarization(predictions, references):
    prediction, source = predictions[0], json.loads(references[0])["source"]

    return (prediction, source)

def agg_llm_eval_summarization(items):
    predictions, sources = zip(*items)

    data = []
    for prediction, source in zip(predictions, sources):
        data.append({"source": source, "prediction": prediction})

    avg_scores = []
    for item in data:
        scores = get_scores(
            SUMMARIZATION_ASPECT_PROMPTS,
            SUMMARIZATION_SYSTEM_PROMPT,
            SUMMARIZATION_PROMPT,
            item["source"],
            item["prediction"],
            "gpt-4-turbo"
        )
        item["scores"] = scores
        avg_scores.append(sum([int(score) for score in scores.values()]) / len(scores))

    # save input and output to file
    with open(f"{OUTPUT_PATH}/llm_eval_summarization_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
        json.dump(data, f)

    # return average score
    return sum(avg_scores) / len(avg_scores)

def llm_eval_translation(predictions, references):
    prediction, source = predictions[0], json.loads(references[0])["source"]

    return (prediction, source)

def agg_llm_eval_translation(items):
    predictions, sources = zip(*items)

    data = []
    for prediction, source in zip(predictions, sources):
        data.append({"source": source, "prediction": prediction})

    avg_scores = []
    for item in data:
        scores = get_scores(
            TRANSLATION_ASPECT_PROMPTS,
            TRANSLATION_SYSTEM_PROMPT,
            TRANSLATION_PROMPT,
            item["source"],
            item["prediction"],
            "gpt-4-turbo"
        )
        item["scores"] = scores
        avg_scores.append(sum([int(score) for score in scores.values()]) / len(scores))

    # save input and output to file
    with open(f"{OUTPUT_PATH}/llm_eval_translation_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
        json.dump(data, f)

    # return average score
    return sum(avg_scores) / len(avg_scores)
