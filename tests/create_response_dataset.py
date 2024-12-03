import yaml
from models.slm import model_to_class


def main():
    with open("test_suite_results.yml", "r") as file:
        test_suite_results = yaml.safe_load(file)

    for model_name, cls in model_to_class.items():
        model = cls()
        model.load_model()
        for test in test_suite_results:
            response_dataset = {
                "command": test["command"],
                "exit_code": test["return_code"],
                "stderr": test["stderr"],
            }
            user_prompt = f"""
When running {response_dataset["command"]},
it failed with exit code {response_dataset["exit_code"]}.
The error message was: {response_dataset["stderr"]}.
Help me fix it.
"""

            response = model.get_response(system_prompt="", user_prompt=user_prompt)
            parsed_response = model.parse_response(response)
            print(parsed_response)


if __name__ == "__main__":
    main()
