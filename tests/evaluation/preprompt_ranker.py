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


class PrepromptRanker:
    def __init__(
        self, csv_path: str, preferred_model: str, preferred_system_prompt_id: int
    ):
        """
        Initialize the PrepromptRanker with CSV data, preferred model, and system prompt

        :param csv_path: Path to the CSV file
        :param preferred_model: The model to focus on for ranking preprompts
        :param preferred_system_prompt_id: The system prompt ID to focus on
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

        # Filter dataframe for preferred model and system prompt
        self.df = self.df[
            (self.df["model"] == preferred_model)
            & (self.df["system_prompt_id"] == preferred_system_prompt_id)
        ]

        # Group data by unique error_id
        self.grouped = self.df.groupby("error_id")

        # Store group rankings
        self.group_rankings = {}

        self.preferred_model = preferred_model
        self.preferred_system_prompt_id = preferred_system_prompt_id

    def select_random_groups(self, num_groups: int) -> List[int]:
        """
        Select random unique groups for analysis

        :param num_groups: Number of groups to select
        :return: List of selected group keys (error_ids)
        """
        group_keys = list(self.grouped.groups.keys())

        # Ensure we don't select more groups than available
        num_groups = min(num_groups, len(group_keys))

        return random.sample(group_keys, num_groups)

    def rank_group_preprompts(self, error_id: int) -> Dict[int, int]:
        """
        Rank preprompts for a specific error through interactive comparison

        :param error_id: The error ID to analyze
        :return: Dictionary of preprompt rankings
        """
        # Get responses for this error
        group_data = self.grouped.get_group(error_id)

        # Get unique preprompts in this group
        preprompts = group_data["preprompt_id"].unique()

        # If only one preprompt, return trivial ranking
        if len(preprompts) <= 1:
            return {preprompts[0]: 1}

        # Store pairwise preferences
        preprompt_scores = {preprompt: 0 for preprompt in preprompts}

        # Print error context
        print("\n--- Error Comparison ---")
        print(f"Error ID: {error_id}")

        # Pairwise comparisons for all combinations of preprompts
        for i in range(len(preprompts)):
            for j in range(i + 1, len(preprompts)):
                preprompt1, preprompt2 = preprompts[i], preprompts[j]

                # Get responses
                response1 = group_data[group_data["preprompt_id"] == preprompt1][
                    "parsed_response"
                ].values[0]
                response2 = group_data[group_data["preprompt_id"] == preprompt2][
                    "parsed_response"
                ].values[0]

                # Print responses
                print(f"{'':-^80}")
                print(f"{'ERROR DETAILS':-^80}")
                print(f"{'':-^80}")
                print(f"Error: {errors[error_id]['stderr']}")
                print(f"{'':*^80}")
                print(f"{'RESPONSE 1 (Preprompt ID: ' + str(preprompt1) + ')':*^80}")
                print(f"{'':*^80}")
                print(response1)
                print(f"{'':*^80}")
                print(f"{'RESPONSE 2 (Preprompt ID: ' + str(preprompt2) + ')':*^80}")
                print(f"{'':*^80}")
                print(response2)

                # Get user preference
                while True:
                    preference = input(
                        f"\nWhich preprompt response do you prefer? (1, 2, or 'skip'): "
                    ).strip()

                    if preference == "1":
                        preprompt_scores[preprompt1] += 1
                        break
                    elif preference == "2":
                        preprompt_scores[preprompt2] += 1
                        break
                    elif preference.lower() == "skip":
                        break
                    else:
                        print("Invalid input. Please enter 1, 2, or 'skip'.")

        # Create ranking based on scores
        sorted_preprompts = sorted(
            preprompt_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Assign ranks (1-based)
        group_ranking = {
            preprompt: rank + 1 for rank, (preprompt, _) in enumerate(sorted_preprompts)
        }

        # Print group ranking
        print("\n--- Preprompt Ranking ---")
        for preprompt, rank in group_ranking.items():
            print(f"Preprompt ID {preprompt}: Rank {rank}")

        return group_ranking

    def analyze_groups(self, num_groups: int):
        """
        Analyze specified number of groups

        :param num_groups: Number of groups to analyze
        """
        # Select random groups
        selected_groups = self.select_random_groups(num_groups)

        # Rank preprompts for each group
        for error_id in selected_groups:
            group_ranking = self.rank_group_preprompts(error_id)
            self.group_rankings[error_id] = group_ranking

        # Save results
        self.save_results()

    def save_results(self):
        """
        Save group rankings to a file
        """
        with open("group_preprompt_rankings.csv", "a") as f:
            for error_id, rankings in self.group_rankings.items():
                sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
                for preprompt, rank in sorted_rankings:
                    f.write(
                        f"{self.preferred_model},{error_id},{self.preferred_system_prompt_id},{preprompt},{rank}\n"
                    )

        print("\nRankings have been saved to 'group_preprompt_rankings.csv'")


def main(csv_path: str):
    """
    Main function to run preprompt ranking process

    :param csv_path: Path to the CSV file
    """
    # Ask user for preferred model
    preferred_model = input("Enter the model you want to rank preprompts for: ")

    # Ask user for preferred system prompt ID
    preferred_system_prompt_id = int(input("Enter the system prompt ID to focus on: "))

    # Get number of groups to analyze
    num_groups = int(input("How many error groups would you like to rank? "))

    # Create ranker with preferred model and system prompt
    ranker = PrepromptRanker(csv_path, preferred_model, preferred_system_prompt_id)

    # Analyze groups
    ranker.analyze_groups(num_groups)


if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual path to your CSV file
    main("evaluation.csv")
