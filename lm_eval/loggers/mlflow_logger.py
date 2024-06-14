import copy
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from lm_eval.loggers.utils import remove_none_pattern


logger = logging.getLogger(__name__)
mlflow_logger = logging.getLogger("mlflow")


class MlflowLogger:
    def __init__(self, **kwargs) -> None:
        """Attaches to wandb logger if already initialized. Otherwise, passes kwargs to wandb.init()

        Args:
            kwargs Optional[Any]: Arguments for configuration.

        Parse and log the results returned from evaluator.simple_evaluate() with:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])
        """
        import mlflow

        mlflow_logger.setLevel(logging.DEBUG)
        self.mlflow_args: Dict[str, Any] = kwargs

        self.run = mlflow.active_run()
        if self.run is not None:
            return

        # set the tracking uri if specified
        tracking_uri = self.mlflow_args.pop("tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # checks and creates an experiment if experiment_name is specified and does not exist
        experiment_name = self.mlflow_args.pop("experiment_name", None)
        experiment_id = None
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = mlflow.create_experiment(experiment_name)

        self.run = mlflow.start_run(experiment_id=experiment_id, **self.mlflow_args)

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())
        self.group_names: List[str] = list(results.get("groups", {}).keys())

    def finish(self) -> None:
        import mlflow

        mlflow.end_run()

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration parameters."""
        self.task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})
        configs = {
            "task_configs": self.task_configs,
            "cli_configs": cli_configs,
        }

        return configs

    def _sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sanitize the results dictionary."""
        _results = copy.deepcopy(self.results.get("results", dict()))

        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)

        # remove string valued keys from the results dict
        summary = {}
        for task in self.task_names:
            task_result = _results.get(task, dict())
            for metric_name, metric_value in task_result.items():
                if isinstance(metric_value, str):
                    summary[f"{task}/{metric_name}"] = metric_value

        for summary_metric, summary_value in summary.items():
            _task, _summary_metric = summary_metric.split("/")
            _results[_task].pop(_summary_metric)

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)

        return _results

    def _log_results_as_table(self) -> None:
        """Generate and log evaluation results as a table to W&B."""
        import mlflow

        columns = [
            "Version",
            "Filter",
            "num_fewshot",
            "Metric",
            "Value",
            "Stderr",
        ]

        def make_table(columns: List[str], key: str = "results"):
            results = copy.deepcopy(self.results)
            table = []

            for k, dic in results.get(key).items():
                if k in self.group_names and not key == "groups":
                    continue
                version = results.get("versions").get(k)
                if version == "N/A":
                    version = None
                n = results.get("n-shot").get(k)

                for (mf), v in dic.items():
                    m, _, f = mf.partition(",")
                    if m.endswith("_stderr"):
                        continue
                    if m == "alias":
                        continue

                    if m + "_stderr" + "," + f in dic:
                        se = dic[m + "_stderr" + "," + f]
                        if se != "N/A":
                            se = "%.4f" % se

                        values = [k, version, f, n, m, str(v), str(se)]
                        table.append(
                            {column: values[idx] for idx, column in enumerate(columns)}
                        )
                    else:
                        table.append(
                            {column: values[idx] for idx, column in enumerate(columns)}
                        )

            return pd.DataFrame.from_records(table)

        # log the complete eval result to W&B Table
        table = make_table(["Tasks"] + columns, "results")
        mlflow.log_table(table, "evaluation/eval_results.json")

        if "groups" in self.results.keys():
            table = make_table(["Groups"] + columns, "groups")
            mlflow.log_table(table, "evaluation/group_eval_results.json")

    def log_eval_result(self) -> None:
        """Log evaluation results to W&B."""
        # Log configs to wandb
        import mlflow

        logger.info("Logging evaluation result to mlfow.")
        configs = self._get_config()
        mlflow.log_params(configs)

        mlflow_results = self._sanitize_results_dict()
        # Log the evaluation metrics to wandb
        mlflow.log_metrics(mlflow_results)
        # mlflow.log_table(self.mlflow_results, "mlflow_results.json")
        # Log the evaluation metrics as W&B Table
        self._log_results_as_table()
        # Log the results dict as json to W&B Artifacts
        mlflow.log_table(self.results, "results.json")

    def _generate_dataset(
        self, data: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate a dataset from evaluation data.

        Args:
            data (List[Dict[str, Any]]): The data to generate a dataset for.
            config (Dict[str, Any]): The configuration of the task.

        Returns:
            pd.DataFrame: A dataframe that is ready to be uploaded to W&B.
        """
        ids = [x["doc_id"] for x in data]
        labels = [x["target"] for x in data]
        instance = [""] * len(ids)
        resps = [""] * len(ids)
        filtered_resps = [""] * len(ids)
        model_outputs = {}

        metrics_list = config["metric_list"]
        metrics = {}
        for metric in metrics_list:
            metric = metric.get("metric")
            if metric in ["word_perplexity", "byte_perplexity", "bits_per_byte"]:
                metrics[f"{metric}_loglikelihood"] = [x[metric][0] for x in data]
                if metric in ["byte_perplexity", "bits_per_byte"]:
                    metrics[f"{metric}_bytes"] = [x[metric][1] for x in data]
                else:
                    metrics[f"{metric}_words"] = [x[metric][1] for x in data]
            else:
                metrics[metric] = [x[metric] for x in data]

        if config["output_type"] == "loglikelihood":
            instance = [x["arguments"][0][0] for x in data]
            labels = [x["arguments"][0][1] for x in data]
            resps = [
                f'log probability of continuation is {x["resps"][0][0][0]} '
                + "\n\n"
                + "continuation will {} generated with greedy sampling".format(
                    "not be" if not x["resps"][0][0][1] else "be"
                )
                for x in data
            ]
            filtered_resps = [
                f'log probability of continuation is {x["filtered_resps"][0][0]} '
                + "\n\n"
                + "continuation will {} generated with greedy sampling".format(
                    "not be" if not x["filtered_resps"][0][1] else "be"
                )
                for x in data
            ]
        elif config["output_type"] == "multiple_choice":
            instance = [x["arguments"][0][0] for x in data]
            choices = [
                "\n".join([f"{idx}. {y[1]}" for idx, y in enumerate(x["arguments"])])
                for x in data
            ]
            resps = [np.argmax([n[0][0] for n in x["resps"]]) for x in data]
            filtered_resps = [
                np.argmax([n[0] for n in x["filtered_resps"]]) for x in data
            ]
        elif config["output_type"] == "loglikelihood_rolling":
            instance = [x["arguments"][0][0] for x in data]
            resps = [x["resps"][0][0] for x in data]
            filtered_resps = [x["filtered_resps"][0] for x in data]
        elif config["output_type"] == "generate_until":
            instance = [x["arguments"][0][0] for x in data]
            resps = [x["resps"][0][0] for x in data]
            filtered_resps = [x["filtered_resps"][0] for x in data]

        model_outputs["raw_predictions"] = resps
        model_outputs["filtered_predictions"] = filtered_resps

        df_data = {
            "id": ids,
            "data": instance,
        }
        if config["output_type"] == "multiple_choice":
            df_data["choices"] = choices

        tmp_data = {
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
        df_data.update(tmp_data)
        df_data.update(model_outputs)
        df_data.update(metrics)

        return pd.DataFrame(df_data)

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to W&B.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        import mlflow

        task_names: List[str] = [
            x for x in self.task_names if x not in self.group_names
        ]

        ungrouped_tasks = []
        tasks_by_groups = {}

        for task_name in task_names:
            group_names = self.task_configs[task_name].get("group", None)
            if group_names:
                if isinstance(group_names, str):
                    group_names = [group_names]

                for group_name in group_names:
                    if not tasks_by_groups.get(group_name):
                        tasks_by_groups[group_name] = [task_name]
                    else:
                        tasks_by_groups[group_name].append(task_name)
            else:
                ungrouped_tasks.append(task_name)

        for task_name in ungrouped_tasks:
            eval_preds = samples[task_name]

            # log the samples as a W&B Table
            df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
            mlflow.log_table(df, f"tasks/{task_name}_eval_results.json")

        for group, grouped_tasks in tasks_by_groups.items():
            grouped_df = pd.DataFrame()
            for task_name in grouped_tasks:
                eval_preds = samples[task_name]
                df = self._generate_dataset(
                    eval_preds, self.task_configs.get(task_name)
                )
                df["group"] = group
                df["task"] = task_name
                grouped_df = pd.concat([grouped_df, df], ignore_index=True)

                mlflow.log_table(df, f"groups/{group}/{task_name}_eval_samples.json")

            mlflow.log_table(grouped_df, f"groups/{group}_eval_results.json")
