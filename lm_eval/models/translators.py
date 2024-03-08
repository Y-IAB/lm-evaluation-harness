import json
from typing import List, Optional, Tuple, Union

import requests
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("translator")
class Translator(LM):

    def __init__(self,
                 api_key: str,
                 endpoint: str,
                 target_lang: str,
                 model: str,
                 batch_size: Optional[Union[int, str]] = 1):
        super().__init__()

        self.endpoint = endpoint
        self.headers = {
            "Authorization": api_key,
            'Content-Type': 'application/json'
        }

        self.target_lang = self._convert_lang_to_code(target_lang)
        self.model = model

        # don't know how to use it yet
        self.batch_size = batch_size

    def generate_until(self, reqs: list[Instance]) -> list[str]:
        results = []

        for request in tqdm(reqs):
            # The first argument is the formatted doc_to_text text.
            source_text = request.args[0]

            data = {
                "text": [source_text],
                "target_lang": self.target_lang,
                "model": self.model
            }

            response = requests.post(self.endpoint,
                                     headers=self.headers,
                                     json=data)

            if response.status_code != 200:
                print(f"An error occurred while translating: {response.text}")
                raise Exception(response.text)

            result = response.json()
            results.append(result['translations'][0]['text'])

        return results

    def _convert_lang_to_code(self, lang: str) -> str:
        lang_code = {
            'en': 'EN-US',
            'ko': 'KO',
            'ja': 'JA',
            'ch': 'ZH-CN',
        }
        return lang_code[lang.lower()]

    def loglikelihood(self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(
            self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass
