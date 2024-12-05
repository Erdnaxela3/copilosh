import os
import subprocess
from dataclasses import asdict, dataclass

import yaml


@dataclass
class CmdError:
    id: int
    command: str
    return_code: int
    script: str
    stderr: str
    stdout: str


def run_shell_scripts_and_collect_data(directory: str):
    """
    Execute all `.sh` files in the specified directory and collect error information.

    :param directory: The directory containing the `.sh` files
    """
    results = []
    id = 0
    for filename in os.listdir(directory):
        if filename.endswith(".sh"):
            script_path_from_cwd = os.path.join(directory, filename)
            with open(script_path_from_cwd, "r") as script_file:
                script_content = script_file.read()
            try:
                # Run the shell script and capture output and error
                process = subprocess.run(
                    script_content,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=directory,
                )
                results.append(
                    CmdError(
                        id=id,
                        command=script_content,
                        return_code=process.returncode,
                        script=filename,
                        stderr=process.stderr.strip(),
                        stdout=process.stdout.strip(),
                    )
                )
            except Exception as e:
                results.append(
                    CmdError(
                        id=id,
                        command=script_content,
                        return_code=-1,
                        script=filename,
                        stderr=f"Failed to execute script: {str(e)}",
                        stdout="",
                    )
                )
            id += 1
    results = [asdict(result) for result in results]
    return results


def save_results_to_yaml(results, output_file):
    """
    Save the collected error information to a YAML file.
    """
    with open(output_file, "w") as yaml_file:
        yaml.dump(results, yaml_file, default_flow_style=False)


def main():
    # Specify the directory containing the `.sh` files
    scripts_directory = "./test_scripts"  # Change this if `.sh` files are in another directory
    output_yaml = "test_suite_results.yml"

    # Run the shell scripts and collect data
    results = run_shell_scripts_and_collect_data(scripts_directory)

    # Save the results to a YAML file
    save_results_to_yaml(results, output_yaml)
    print(f"Test suite results saved to {output_yaml}")


if __name__ == "__main__":
    main()
