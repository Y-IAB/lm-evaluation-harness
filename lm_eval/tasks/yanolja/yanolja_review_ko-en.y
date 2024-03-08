task: yanolja_review_ko-en
dataset_name: yanolja_review_ko-en
dataset_path: json
dataset_kwargs:
  data_files: /data/shared/datasets/yanolja_evaluations/yanolja_review_ko-en.jsonl

group: yanolja_translation
output_type: generate_until
test_split: train
doc_to_text: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful and concise answers to the user's questions.
Human: 주어진 한국어 문장을 영어로 번역해주세요. 번역된 문장은 영어로 시작해야 합니다.
Korean: {{source}}
Assistant: "
doc_to_target: "" # Reference-free metric

metric_list:
  - metric: xcomet
    aggregation: mean
    higher_is_better: true

generation_kwargs:
  until:
    - "\n\n"
    - "\n"
    - "<|im_end|>"
    - "</s>"
metadata:
  version: 1.0