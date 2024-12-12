import logging
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
class PrePrompt:
    id: int
    user_prompt: str
    assistant_response: str


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
    response_id: int
    model_name: str
    error_id: int
    system_prompt_id: int
    preprompt_id: int
    parsed_response: str
    time: float


logger = logging.getLogger(__name__)


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
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("copilosh").setLevel(logging.INFO)

    with open("test_suite_results.yml", "r") as f:
        cmd_errors = [CmdError(**err) for err in yaml.safe_load(f)]

    with open("system_prompts.yml", "r") as f:
        system_prompts = [SystemPrompt(**sp) for sp in yaml.safe_load(f)]

    # None is for zero shot prompt
    with open("pre_prompts.yml", "r") as f:
        pre_prompts = [None] + [PrePrompt(**pp) for pp in yaml.safe_load(f)]

    n_sp, n_errors, n_models, n_pp = (
        len(system_prompts),
        len(cmd_errors),
        len(model_to_class),
        len(pre_prompts),
    )
    n_responses = n_sp * n_errors * n_models * n_pp
    logger.info(f"Creating dataset with {n_responses} generated responses")

    output_file = "response_dataset.yml"

    # Check if the file already exists and how many responses are already generated
    try:
        with open(output_file, "r") as f:
            already_generated = yaml.safe_load(f)
            if already_generated:
                logger.info(
                    f"Already generated {len(already_generated)} responses. Appending to the file."
                )
            already_done = len(already_generated)
    except:
        already_done = 0

    # Append mode if the program was interrupted
    file = open(output_file, "a")

    response_i = 0
    for model_no, (model_name, model_cls) in enumerate(model_to_class.items(), 1):
        model = model_cls()
        model.load_model()
        for err_no, cmd_err in enumerate(cmd_errors, 1):
            for sp_no, sp in enumerate(system_prompts, 1):
                for pp_no, pre_prompt in enumerate(pre_prompts, 1):
                    # Skip already done responses
                    if response_i < already_done:
                        response_i += 1
                        continue

                    if pre_prompt:
                        messages = [
                            {"role": "user", "content": pre_prompt.user_prompt},
                            {
                                "role": "assistant",
                                "content": pre_prompt.assistant_response,
                            },
                        ]
                        pp_id = pre_prompt.id
                    else:
                        messages = None
                        pp_id = 0
                    response_i += 1

                    logger.info(
                        f"Response {response_i}/{n_responses} - Model {model_no}/{n_models} - Error {err_no}/{n_errors} - System Prompt {sp_no}/{n_sp} - Pre Prompt {pp_no}/{n_pp} ({model_name}, {cmd_err.id}, {sp.id}, {pp_id})"
                    )

                    user_prompt = user_prompt_formatter(cmd_err)
                    start_time = time.time()
                    response = model.get_response(
                        system_prompt=str(sp),
                        user_prompt=user_prompt,
                        messages=messages,
                    )

                    if response_i % 20 == 0:
                        logger.info(f"Response: {response}")

                    parsed_response = model.parse_response(response)
                    response_time = time.time() - start_time

                    response = ResponseFromModel(
                        response_id=response_i,
                        model_name=model_name,
                        error_id=cmd_err.id,
                        system_prompt_id=sp.id,
                        preprompt_id=pp_id,
                        parsed_response=parsed_response,
                        time=response_time,
                    )
                    yaml.dump([asdict(response)], file)


if __name__ == "__main__":
    main()
