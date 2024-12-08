import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class CmdError:
    id: int
    command: str
    return_code: int
    script: str
    stderr: str
    stdout: str


with open("../test_suite_results.yml", "r") as f:
    errors = yaml.safe_load(f)


class ModelRanker:
    def __init__(self, csv_path: str):
        """
        Initialize the ModelRanker with CSV data

        :param csv_path: Path to the CSV file
        """
        self.df = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = [
            "response_id",
            "model",
            "error_id",
            "system_prompt_id",
            "preprompt_id",
            "parsed_response",
        ]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Group data by unique combination of error, system prompt, and preprompt
        self.grouped = self.df.groupby(["error_id", "system_prompt_id", "preprompt_id"])

        # Store group rankings
        self.group_rankings = {}

    def select_random_groups(self, num_groups: int) -> List[Tuple]:
        """
        Select random unique groups for analysis

        :param num_groups: Number of groups to select
        :return: List of selected group keys
        """
        group_keys = list(self.grouped.groups.keys())

        # Ensure we don't select more groups than available
        num_groups = min(num_groups, len(group_keys))

        return random.sample(group_keys, num_groups)

    def rank_group_models(self, group_key: Tuple) -> Dict[str, int]:
        """
        Rank models for a specific group through interactive comparison

        :param group_key: Tuple of (error_id, system_prompt_id, preprompt_id)
        :return: Dictionary of model rankings
        """
        # Get responses for this group
        group_data = self.grouped.get_group(group_key)

        # Get unique models in this group
        models = group_data["model"].unique()

        # If only one model, return trivial ranking
        if len(models) <= 1:
            return {models[0]: 1}

        # Store pairwise preferences
        model_scores = {model: 0 for model in models}

        # Print group context
        print("\n--- Group Comparison ---")
        print(f"Group Context: {group_key}")

        # Pairwise comparisons for all combinations of models
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]

                # Get responses
                response1 = group_data[group_data["model"] == model1][
                    "parsed_response"
                ].values[0]
                response2 = group_data[group_data["model"] == model2][
                    "parsed_response"
                ].values[0]

                # Print responses
                print(f"{'':-^80}")
                print(f"{'ERROR':-^80}")
                print(f"{'':-^80}")
                print(f"Error: {errors[group_key[0]]['stderr']}")
                print(f"{'':*^80}")
                print(f"{'RESPONSE 1':*^80}")
                print(f"{'':*^80}")
                print(response1)
                print(f"{'':*^80}")
                print(f"{'RESPONSE 2':*^80}")
                print(f"{'':*^80}")
                print(response2)

                # Get user preference
                while True:
                    preference = input(
                        f"\nWhich model do you prefer? (1, 2, or 'skip'): "
                    ).strip()

                    if preference == "1":
                        model_scores[model1] += 1
                        break
                    elif preference == "2":
                        model_scores[model2] += 1
                        break
                    elif preference.lower() == "skip":
                        break
                    else:
                        print("Invalid input. Please enter 1, 2, or 'skip'.")

        # Create ranking based on scores
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign ranks (1-based)
        group_ranking = {
            model: rank + 1 for rank, (model, _) in enumerate(sorted_models)
        }

        # Print group ranking
        print("\n--- Group Ranking ---")
        for model, rank in group_ranking.items():
            print(f"Model {model}: Rank {rank}")

        return group_ranking

    def analyze_groups(self, num_groups: int):
        """
        Analyze specified number of groups

        :param num_groups: Number of groups to analyze
        """
        # Select random groups
        selected_groups = self.select_random_groups(num_groups)

        # Rank models for each group
        for group_key in selected_groups:
            group_ranking = self.rank_group_models(group_key)
            self.group_rankings[group_key] = group_ranking

        # Save results
        self.save_results()

    def save_results(self):
        """
        Save group rankings to a file
        """

        with open("group_model_rankings.csv", "a") as f:
            for group_key, rankings in self.group_rankings.items():
                sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
                for model, rank in sorted_rankings:
                    f.write(
                        f"{group_key[0]},{group_key[1]},{group_key[2]},{model},{rank}\n"
                    )

        print("\nRankings have been saved to 'group_model_rankings.txt'")


def main(csv_path: str):
    """
    Main function to run model ranking process

    :param csv_path: Path to the CSV file
    """
    ranker = ModelRanker(csv_path)

    # Get number of groups to analyze
    num_groups = int(input("How many groups would you like to rank? "))

    # Analyze groups
    ranker.analyze_groups(num_groups)


if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual path to your CSV file
    main("evaluation.csv")
