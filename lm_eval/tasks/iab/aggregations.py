import evaluate
from comet import download_model, load_from_checkpoint


def bleu(results):
    predictions, references = zip(*[(result["result"], result["reference"])
                                    for result in results])

    bleu_fn = evaluate.load("bleu")
    return bleu_fn.compute(predictions=predictions,
                           references=references)["bleu"]


def unbabel_comet(results):
    score = 0

    for result in results:
        data = [{"src": result["source"], "mt": result["result"]}]

        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)

        model_output = model.predict(data, batch_size=8, gpus=0)

        # Example output:
        # Prediction([('scores', [0.8676194548606873]), ('system_score', 0.8676194548606873)])
        score += model_output[0][0]

    return score / len(results)
