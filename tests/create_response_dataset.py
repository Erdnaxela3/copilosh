import time
from dataclasses import asdict, dataclass

import yaml
from models.slm import model_to_class


@dataclass
class SystemPrompt:
    id: int
    type: str
    prompt: str


@dataclass
class CmdError:
    id: int
    command: str
    return_code: int
    script: str
    stderr: str
    stdout: str


@dataclass
class ResponseFromModel:
    model_name: str
    error_id: int
    system_prompt_id: int
    parsed_response: str
    time: float


def user_prompt_formatter(cmd_error: CmdError) -> str:
    """
    Format the user prompt from the command error, that will be used to generate a response.

    :param cmd_error: The command error.
    :return: The formatted user prompt.
    """
    return f"""
When running {cmd_error.command},
it failed with exit code {cmd_error.return_code}.
The error message was: {cmd_error.stderr}.
Help me fix it.
"""


def main():
    with open("test_suite_results.yml", "r") as f:
        cmd_errors = [CmdError(**err) for err in yaml.safe_load(f)]

    with open("system_prompts.yml", "r") as f:
        system_prompts = [SystemPrompt(**sp) for sp in yaml.safe_load(f)]

    responses = []
    for model_name, model_cls in model_to_class.items():
        model = model_cls()
        model.load_model()
        for cmd_err in cmd_errors:
            for sp in system_prompts:
                user_prompt = user_prompt_formatter(cmd_err)
                start_time = time.time()
                response = model.get_response(system_prompt=sp, user_prompt=user_prompt)
                parsed_response = model.parse_response(response)
                response_time = time.time() - start_time

                responses.append(
                    ResponseFromModel(
                        model_name=model_name,
                        error_id=cmd_err.id,
                        system_prompt_id=sp.id,
                        parsed_response=parsed_response,
                        time=response_time,
                    )
                )
    dict_responses = [asdict(r) for r in responses]
    with open("response_dataset.yml", "w") as f:
        yaml.dump(dict_responses, f)


if __name__ == "__main__":
    main()
