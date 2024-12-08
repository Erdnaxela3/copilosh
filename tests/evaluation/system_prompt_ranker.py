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


class SystemPromptRanker:
    def __init__(self, csv_path: str, preferred_model: str):
        """
        Initialize the SystemPromptRanker with CSV data and preferred model

        :param csv_path: Path to the CSV file
        :param preferred_model: The model to focus on for ranking system prompts
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

        # Filter dataframe for preferred model
        self.df = self.df[self.df["model"] == preferred_model]

        # Group data by unique combination of error, and preprompt
        self.grouped = self.df.groupby(["error_id", "preprompt_id"])

        # Store group rankings
        self.group_rankings = {}

        self.preferred_model = preferred_model

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

    def rank_group_system_prompts(self, group_key: Tuple) -> Dict[int, int]:
        """
        Rank system prompts for a specific group through interactive comparison

        :param group_key: Tuple of (error_id, preprompt_id)
        :return: Dictionary of system prompt rankings
        """
        # Get responses for this group
        group_data = self.grouped.get_group(group_key)

        # Get unique system prompts in this group
        system_prompts = group_data["system_prompt_id"].unique()

        # If only one system prompt, return trivial ranking
        if len(system_prompts) <= 1:
            return {system_prompts[0]: 1}

        # Store pairwise preferences
        system_prompt_scores = {prompt: 0 for prompt in system_prompts}

        # Print group context
        print("\n--- Group Comparison ---")
        print(f"Group Context: {group_key}")

        # Pairwise comparisons for all combinations of system prompts
        for i in range(len(system_prompts)):
            for j in range(i + 1, len(system_prompts)):
                system_prompt1, system_prompt2 = system_prompts[i], system_prompts[j]

                # Get responses
                response1 = group_data[
                    group_data["system_prompt_id"] == system_prompt1
                ]["parsed_response"].values[0]
                response2 = group_data[
                    group_data["system_prompt_id"] == system_prompt2
                ]["parsed_response"].values[0]

                # Print responses
                print(f"{'':-^80}")
                print(f"{'ERROR':-^80}")
                print(f"{'':-^80}")
                print(f"Error: {errors[group_key[0]]['stderr']}")
                print(f"{'':*^80}")
                print(
                    f"{'RESPONSE 1 (System Prompt ID: ' + str(system_prompt1) + ')':*^80}"
                )
                print(f"{'':*^80}")
                print(response1)
                print(f"{'':*^80}")
                print(
                    f"{'RESPONSE 2 (System Prompt ID: ' + str(system_prompt2) + ')':*^80}"
                )
                print(f"{'':*^80}")
                print(response2)

                # Get user preference
                while True:
                    preference = input(
                        f"\nWhich system prompt do you prefer? (1, 2, or 'skip'): "
                    ).strip()

                    if preference == "1":
                        system_prompt_scores[system_prompt1] += 1
                        break
                    elif preference == "2":
                        system_prompt_scores[system_prompt2] += 1
                        break
                    elif preference.lower() == "skip":
                        break
                    else:
                        print("Invalid input. Please enter 1, 2, or 'skip'.")

        # Create ranking based on scores
        sorted_system_prompts = sorted(
            system_prompt_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Assign ranks (1-based)
        group_ranking = {
            system_prompt: rank + 1
            for rank, (system_prompt, _) in enumerate(sorted_system_prompts)
        }

        # Print group ranking
        print("\n--- Group Ranking ---")
        for system_prompt, rank in group_ranking.items():
            print(f"System Prompt ID {system_prompt}: Rank {rank}")

        return group_ranking

    def analyze_groups(self, num_groups: int):
        """
        Analyze specified number of groups

        :param num_groups: Number of groups to analyze
        """
        # Select random groups
        selected_groups = self.select_random_groups(num_groups)

        # Rank system prompts for each group
        for group_key in selected_groups:
            group_ranking = self.rank_group_system_prompts(group_key)
            self.group_rankings[group_key] = group_ranking

        # Save results
        self.save_results()

    def save_results(self):
        """
        Save group rankings to a file
        """
        with open("group_system_prompt_rankings.csv", "a") as f:
            for group_key, rankings in self.group_rankings.items():
                sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
                for system_prompt, rank in sorted_rankings:
                    f.write(
                        f"{self.preferred_model},{group_key[0]},{group_key[1]},{system_prompt},{rank}\n"
                    )

        print("\nRankings have been saved to 'group_system_prompt_rankings.csv'")


def main(csv_path: str):
    """
    Main function to run system prompt ranking process

    :param csv_path: Path to the CSV file
    """
    # Ask user for preferred model
    preferred_model = input("Enter the model you want to rank system prompts for: ")

    # Create ranker with preferred model
    ranker = SystemPromptRanker(csv_path, preferred_model)

    # Get number of groups to analyze
    num_groups = int(input("How many groups would you like to rank? "))

    # Analyze groups
    ranker.analyze_groups(num_groups)


if __name__ == "__main__":
    # Replace 'your_data.csv' with the actual path to your CSV file
    main("evaluation.csv")
