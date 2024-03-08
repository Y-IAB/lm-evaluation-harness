def process_results(docs, results):
    source_text = docs["source"]
    reference = docs["target"]
    result = results[0]

    return {
        "unbabel_comet": {
            "source": source_text,
            "reference": reference,
            "result": result
        },
        "bleu": {
            "reference": reference,
            "result": result
        }
    }
