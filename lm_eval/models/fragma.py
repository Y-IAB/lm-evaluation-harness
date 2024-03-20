import json
from typing import List, Optional, Tuple, Union

import requests
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("fragma")
class Fragma(LM):

    def __init__(self,
                 api_key: str,
                 endpoint: str,
                 batch_size: Optional[Union[int, str]] = 1):
        super().__init__()

        self.endpoint = endpoint
        self.headers = {
            "Authorization": api_key,
            'Content-Type': 'application/json'
        }

        # don't know how to use it yet
        self.batch_size = batch_size

    def generate_until(self, reqs: list[Instance]) -> list[str]:
        results = []

        for request in tqdm(reqs):
            # The first argument is the formatted doc_to_text text.
            source_text = request.args[0]

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": source_text
                    }
                ]
            }

            response = requests.post(self.endpoint,
                                     headers=self.headers,
                                     json=data)

            if response.status_code != 200:
                print(f"An error occurred while generating: {response.text}")
                raise Exception(response.text)

            result = response.json()
            results.append(result['choices'][0]['message']["content"])

        return results

    def loglikelihood(self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(
            self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass
