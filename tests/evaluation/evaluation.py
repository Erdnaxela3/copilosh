from dataclasses import dataclass

import pandas as pd
import yaml
from evaluate import load
from rouge_score import rouge_scorer


@dataclass
class ResponseFromModel:
    model_name: str
    error_id: int
    system_prompt_id: int
    parsed_response: str
    time: float

@dataclass
class ExpectedResponse:
    error_id: int
    expected_response: str

def main():
    with open("../response_dataset.yml", "r") as f:
        models_responses = [ResponseFromModel(**response) for response in yaml.safe_load(f)]

    with open("expected_responses.yml", "r") as f:
        expected_responses = [ExpectedResponse(**response) for response in yaml.safe_load(f)]

    evaluation_df = pd.DataFrame()
    for model_response in models_responses:
        expected_response = expected_responses[model_response.error_id]
        if expected_response is None:
            print(f"Expected response not found for error {model_response.error_id}")
            continue

        # ROUGE
        red_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        red_scores = red_scorer.score(model_response.parsed_response, expected_response.expected_response)
        
        # BLEU
        bleu_scorer = load("bleu")
        bleu_scores = bleu_scorer.compute(predictions=[model_response.parsed_response], references=[[expected_response.expected_response]])

        # BERT
        bert_scorer = load("bertscore")
        bert_scores = bert_scorer.compute(predictions=[model_response.parsed_response], references=[expected_response.expected_response], model_type="distilbert-base-uncased")

        evaluation_df = pd.concat([evaluation_df, pd.DataFrame({
            "model": [model_response.model_name],
            "error_id": [model_response.error_id],
            "system_prompt_id": [model_response.system_prompt_id],
            "time": [model_response.time],
            "rouge_1": [red_scores['rouge1'].fmeasure],
            "rouge_2": [red_scores['rouge2'].fmeasure],
            "rouge_L": [red_scores['rougeL'].fmeasure],
            "bleu": [bleu_scores['bleu']],
            "bleu_precisions": [bleu_scores['precisions']],
            "bleu_brevity_penalty": [bleu_scores['brevity_penalty']],
            "bleu_length_ratio": [bleu_scores['length_ratio']],
            "bleu_translation_length": [bleu_scores['translation_length']],
            "bleu_reference_length": [bleu_scores['reference_length']],
            "bert_precision": [bert_scores['precision'][0]],
            "bert_recall": [bert_scores['recall'][0]],
            "bert_f1": [bert_scores['f1'][0]],
        })], ignore_index=True)
    
    evaluation_df.to_csv("evaluation.csv", index=False)

if __name__ == "__main__":
    main()