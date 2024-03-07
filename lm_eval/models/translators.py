import json
import logging
import time
from typing import List, Literal, Optional, Tuple, Union

import deepl
import httpx
import openai
import requests
from openai import AzureOpenAI
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logging.basicConfig()
logging.getLogger('deepl').setLevel(logging.ERROR)


@register_model("deepl")
class DeepLTranslator(LM):

    def __init__(self,
                 api_key: str,
                 source_lang: str,
                 target_lang: str,
                 batch_size: Optional[Union[int, str]] = 1):
        super().__init__()

        self.api_key = api_key

        self.source_lang = self._convert_lang_to_code(source_lang)
        self.target_lang = self._convert_lang_to_code(target_lang)

        self.translator = deepl.Translator(self.api_key)

        # don't know how to use it yet
        self.batch_size = batch_size

    def generate_until(self, reqs: list[Instance]) -> list[str]:
        results = []

        for request in tqdm(reqs):
            # The first argument is the formatted doc_to_text text.
            source_text = request.args[0]

            try:
                result = self.translator.translate_text(
                    source_text,
                    target_lang=self.target_lang,
                    # Do we need these options?
                    # preserve_formatting=False,
                    # context=
                    # tag_handling='xml',
                )
                results.append(result.text)

            except deepl.exceptions.DeepLException as e:
                print(f"An error occurred while translating: {e}")
                raise e

        return results

    def _convert_lang_to_code(self, lang: str) -> str:
        # Convert the language name to the language code
        # https://www.deepl.com/docs-api/translate-text/?utm_source=github&utm_medium=github-python-readme
        lang_code = {
            'english': 'EN-US',
            'korean': 'KO',
            'japanese': 'JA',
            'chinese': 'ZH',
            'romanian': 'RO',
        }
        return lang_code[lang.lower()]

    def loglikelihood(self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(
            self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass