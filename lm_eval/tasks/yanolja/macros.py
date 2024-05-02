SUMMARIZATION_SYSTEM_PROMPT = """Task Description:
You will be given a text using which a summary has been generated. Your task is to evaluate the summary based on the given metric. Evaluate to which extent does the summary follows the given metric considering the text as the input. Use the following evaluation criteria to judge the extent to which the metric is followed. Make sure you understand the task and the following evaluation metric very clearly.

Evaluation Criteria:
The task is to judge the extent to which the metric is followed by the summary. Following are the scores and the evaluation criteria according to which scores must be assigned.
<score>1</score> - The metric is not followed at all while generating the summary from the text.
<score>2</score> - The metric is followed only to a limited extent while generating the summary from the text.
<score>3</score> - The metric is followed to a good extent while generating the summary from the text.
<score>4</score> - The metric is followed mostly while generating the summary from the text.
<score>5</score> - The metric is followed completely while generating the summary from the text.

Metric:
{aspect_prompt}
"""

SUMMARIZATION_PROMPT = """
Text:
{source}

Summary:
{prediction}

Evaluation Steps:
Follow the following steps strictly while giving the response:
1.First write down the steps that are needed to evaluate the summary as per the metric. Reiterate what metric you will be using to evaluate the summary.
2.Give a step-by-step explanation if the summary adheres to the metric considering the text as the input. Stick to the metric only for evaluation.
3.Next, evaluate the extent to which the metric is followed.
4.Use the previous information to rate the summary using the evaluation criteria and assign a score within the <score></score> tags.

Note: Strictly give the score within <score></score> tags only e.g Score- <score>5</score>.

First give a detailed explanation and then finally give a single score following the format: Score- <score>5</score>

THE EVALUATION AND SCORE MUST BE ASSIGNED STRICTLY ACCORDING TO THE METRIC ONLY AND NOTHING ELSE!

Response:
"""
TRANSLATION_SYSTEM_PROMPT = """Task Description:
You will be given a set of text segments in a source language and their translations into a target language. Your task is to evaluate the quality of the translations based on a given metric. Assess to what extent the translations adhere to this metric considering the original text segments as the input. Use the following evaluation criteria to judge the extent to which the metric is followed. Ensure you understand the task and the following evaluation metric very clearly.

Evaluation Criteria:
The task is to judge the extent to which the metric is followed by the translation. The following scores and evaluation criteria should be used to assign scores:
<score>1</score> - The metric is not followed at all in the translation of the text segments.
<score>2</score> - The metric is followed only to a limited extent in the translation of the text segments.
<score>3</score> - The metric is followed to a good extent in the translation of the text segments.
<score>4</score> - The metric is followed mostly in the translation of the text segments.
<score>5</score> - The metric is followed completely in the translation of the text segments.

Metric:
{aspect_prompt}
"""

TRANSLATION_PROMPT = """
Source Text Segments:
{source}

Translated Text:
{prediction}

Evaluation Steps:
1. First, write down the steps needed to evaluate the translation according to the metric. Reiterate what metric you will be using to evaluate the translation.
2. Provide a step-by-step explanation of whether the translation adheres to the metric, considering the source text segments as the input. Focus solely on the metric for evaluation.
3. Next, evaluate the extent to which the metric is followed.
4. Use the previous information to rate the translation using the evaluation criteria and assign a score within the <score></score> tags.

Note: Strictly assign the score within <score></score> tags only, e.g., Score- <score>5</score>.

First, give a detailed explanation and then finally give a single score following the format: Score- <score>5</score>.

THE EVALUATION AND SCORE MUST BE ASSIGNED STRICTLY ACCORDING TO THE METRIC ONLY AND NOTHING ELSE!

Response:
"""

SUMMARIZATION_ASPECT_PROMPTS = {
    "fluency": "fluency - The quality of summary in terms of grammar, spelling, punctuation, capitalization, word choice, and sentence structure and should contain no errors. The summary should be easy to read, follow, comprehend and should contain no errors. Annotators received specific guidelines on how to penalize summaries based on fluency levels.",
    "coherence": "coherence - The collective quality of all sentences. The summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information.",
    "relevance": "relevance - The summary should not contain opinions that are either not consensus or important. The summary should include only important opinions from the reviews. Annotators were instructed to penalize summaries if they contained redundancies and excess/unimportant information.",
    "faithfulness": "faithfulness - Every piece of information mentioned in the summary should be verifiable/supported/inferred from the reviews only. Summaries should be penalized if any piece of information is not verifiable/supported/inferred from the reviews or if the summary overgeneralizes something.",
    "aspect coverage": "aspect coverage - The summary should cover all the aspects that are majorly being discussed in the reviews. Summaries should be penalized if they miss out on an aspect that was majorly being discussed in the reviews and awarded if it covers all.",
    "sentiment consistency": "sentiment consistency - All the aspects being discussed in the summary should accurately reflect the consensus sentiment of the corresponding aspects from the reviews. Summaries should be penalized if they do not cover accurately the sentiment regarding any aspect within the summary.",
    "specificity": "specificity - The summary should avoid containing generic opinions. All the opinions within the summary should contain detailed and specific information about the consensus opinions. Summaries should be penalized for missing out details and should be awarded if they are specific."
}

TRANSLATION_ASPECT_PROMPTS = {
    "fluency": "fluency - The quality of the translated text in terms of grammar, spelling, punctuation, capitalization, word choice, and sentence structure and should contain no errors. The translated text should be easy to read, follow, comprehend and should contain no errors. Annotators received specific guidelines on how to penalize translated text based on fluency levels.",
    "adequacy": "adequacy - This assesses whether the translation accurately conveys the meaning of the source text without omissions or distortions. The translation should be complete and include all relevant information from the original text.",
    "style matching": "style matching - This evaluates how well the translation captures the style and tone of the original text. The translated text should maintain the author's voice and the text's original mood, which may involve cultural adaptation to fit the target audience.",
    "terminology consistency" : "terminology consistency - This ensures that specialized terms are translated consistently throughout the document. It's crucial for texts in specific fields like legal, medical, or technical documents."
}
