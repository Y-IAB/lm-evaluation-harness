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
                 batch_size: Optional[Union[int, str]] = 1,
                 **kwargs):
        super().__init__()

        self.endpoint = endpoint
        self.headers = {
            "Authorization": api_key,
            'Content-Type': 'application/json'
        }

        # Emergency hotfix: set cookie values
        """
        if 'user_agent' in kwargs:
            self.headers['User-Agent'] = kwargs['user_agent']
        if 'cookie' in kwargs:
            self.headers['Cookie'] = kwargs['cookie']
        """
        self.headers['User-Agent'] = 'Python-local'
        self.headers['Cookie'] = '__cf_bm=buqPccpdrKV0gPIyU16Ooqph8rAp8zoMAkrGHQxY8VQ-1717054278-1.0.1.1-ru9i.aWQfBUUEgXIj_dzvW3NZjTTEsKaStH6.h3rysHCFjA1aXPNbjHNaTeTlqYEtAS8z6YZvJx2sDSUkzGZYg'

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
            max_attempts = 10
            attempts = 0
            while attempts < max_attempts:
                try:
                    response = requests.post(self.endpoint,
                                             headers=self.headers,
                                             json=data)
                    response.raise_for_status()

                    result = response.json()
                    results.append(result['translations'][0]['text'])
                    break
                except Exception as e:
                    attempts += 1
                    print(e)
                    print("error occured. retrying...")
                    continue

            if attempts == max_attempts:
                print("maximum retries occured. Insert dummy result...")
                results.append("dummy text")
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
