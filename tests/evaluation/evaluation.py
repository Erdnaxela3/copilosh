from dataclasses import dataclass
import os
import pandas as pd
import yaml
from evaluate import load
from rouge_score import rouge_scorer

@dataclass
class ResponseFromModel:
    response_id: int
    model_name: str
    error_id: int
    system_prompt_id: int
    preprompt_id: int
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

    red_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scorer = load("bleu")
    bert_scorer = load("bertscore")

    if os.path.exists("evaluation.csv"):
        evaluation_df = pd.read_csv("evaluation.csv")
        processed_ids = set(evaluation_df["response_id"])
    else:
        evaluation_df = pd.DataFrame()
        processed_ids = set()

    for model_response in models_responses:
        # Skip already processed responses
        if model_response.response_id in processed_ids:
            continue

        expected_response = next((resp for resp in expected_responses if resp.error_id == model_response.error_id), None)
        if expected_response is None:
            print(f"Expected response not found for error {model_response.error_id}")
            continue

        # Compute ROUGE scores
        red_scores = red_scorer.score(model_response.parsed_response, expected_response.expected_response)
        
        # Compute BLEU scores
        if model_response.parsed_response:
            bleu_scores = bleu_scorer.compute(
                predictions=[model_response.parsed_response],
                references=[[expected_response.expected_response]]
            )
        else:
            bleu_scores = {
                "bleu": 0,
                "precisions": [0, 0, 0, 0],
                "brevity_penalty": 0,
                "length_ratio": 0,
                "translation_length": 0,
                "reference_length": 0,
            }

        # Compute BERT scores
        bert_scores = bert_scorer.compute(
            predictions=[model_response.parsed_response],
            references=[expected_response.expected_response],
            model_type="distilbert-base-uncased"
        )

        new_row = {
            "response_id": model_response.response_id,
            "model": model_response.model_name,
            "error_id": model_response.error_id,
            "system_prompt_id": model_response.system_prompt_id,
            "preprompt_id": model_response.preprompt_id,
            "rank": 0,
            "parsed_response": model_response.parsed_response,
            "time": model_response.time,
            "rouge_1": red_scores['rouge1'].fmeasure,
            "rouge_2": red_scores['rouge2'].fmeasure,
            "rouge_L": red_scores['rougeL'].fmeasure,
            "bleu": bleu_scores['bleu'],
            "bleu_precisions": bleu_scores['precisions'],
            "bleu_brevity_penalty": bleu_scores['brevity_penalty'],
            "bleu_length_ratio": bleu_scores['length_ratio'],
            "bleu_translation_length": bleu_scores['translation_length'],
            "bleu_reference_length": bleu_scores['reference_length'],
            "bert_precision": bert_scores['precision'][0],
            "bert_recall": bert_scores['recall'][0],
            "bert_f1": bert_scores['f1'][0],
        }
        evaluation_df = pd.concat([evaluation_df, pd.DataFrame([new_row])], ignore_index=True)

        evaluation_df.to_csv("evaluation.csv", index=False)
    
    evaluation_df.sort_values(by=["error_id", "system_prompt_id", "preprompt_id"], inplace=True)
    evaluation_df.to_csv("evaluation.csv", index=False)

if __name__ == "__main__":
    main()
