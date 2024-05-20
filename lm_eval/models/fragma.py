import os
from typing import Optional, Union

import requests
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("fragma")
class Fragma(LM):
    def __init__(
        self, api_key: str, endpoint: str, batch_size: Optional[Union[int, str]] = 1
    ):
        super().__init__()

        env_api_key = os.environ.get("FRAGMA_API_KEY")
        self.endpoint = endpoint
        self.headers = {
            "api-key": api_key or env_api_key,
            "Content-Type": "application/json",
        }

        # don't know how to use it yet
        self.batch_size = batch_size

    def generate_until(self, reqs: list[Instance]) -> list[str]:
        results = []

        for request in tqdm(reqs):
            # The first argument is the formatted doc_to_text text.
            source_text = request.args[0]

            data = {"messages": [{"role": "user", "content": source_text}]}
            max_attempts = 10
            attempts = 0
            while attempts < max_attempts:
                try:
                    response = requests.post(
                        self.endpoint, headers=self.headers, json=data
                    )

                    response.raise_for_status()

                    result = response.json()
                    results.append(result["choices"][0]["message"]["content"])
                    break
                except Exception as e:
                    attempts += 1
                    print(e)
                    print("error occurred. retrying...")
                    continue

            if attempts == max_attempts:
                print("maximum retries occurred. Insert dummy result...")
                results.append("dummy text")
        return results

    def loglikelihood(self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(self, reqs: list[Instance]) -> list[tuple[float, bool]]:
        pass
