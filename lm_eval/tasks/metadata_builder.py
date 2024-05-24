import dataclasses
import os
from collections import defaultdict
from typing import Any, Dict, Set

import yaml


@dataclasses.dataclass
class Metadata:
    tasks: Set[str] = dataclasses.field(default_factory=set)
    groups: Set[str] = dataclasses.field(default_factory=set)

    def to_dict(self):
        return {"tasks": list(self.tasks), "groups": list(self.groups)}


def build_metadata(directory: str):
    yaml.add_multi_constructor(
        "!function", lambda loader, suffix, node: None, Loader=yaml.SafeLoader
    )

    result: Dict[str, Metadata] = defaultdict(Metadata)
    for root, _, files in os.walk(directory):
        key = root.removeprefix(directory).removeprefix(os.path.sep)
        for file in files:
            if not file.endswith("yaml"):
                continue

            if file == "metadata.yaml":
                continue

            with open(os.path.join(root, file), encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f)
                if not data:
                    continue

                tasks = data.get("task")
                if not isinstance(tasks, list):
                    tasks = [tasks] if tasks else []
                if all(isinstance(elem, str) for elem in tasks):
                    result[key].tasks.update(tasks)

                groups = data.get("group")
                if not isinstance(groups, list):
                    groups = [groups] if groups else []
                if all(isinstance(elem, str) for elem in tasks):
                    result[key].groups.update(groups)

    result_dict = {key: value.to_dict() for key, value in result.items()}
    result_yaml = yaml.dump(result_dict)
    with open(os.path.join(directory, "metadata.yaml"), "w", encoding="utf-8") as f:
        f.write(result_yaml)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_metadata(current_dir)
