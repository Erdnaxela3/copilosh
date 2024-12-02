import os
import subprocess
import yaml

def run_shell_scripts_and_collect_data(directory):
    """
    Execute all `.sh` files in the specified directory and collect error information.
    """
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".sh"):
            script_path = os.path.join(directory, filename)
            command = f"bash {script_path}"
            try:
                # Run the shell script and capture output and error
                process = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                results.append({
                    "script": filename,
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": process.stdout.strip(),
                    "stderr": process.stderr.strip(),
                })
            except Exception as e:
                results.append({
                    "script": filename,
                    "command": command,
                    "error": f"Failed to execute script: {str(e)}",
                })
    return results

def save_results_to_yaml(results, output_file):
    """
    Save the collected error information to a YAML file.
    """
    with open(output_file, "w") as yaml_file:
        yaml.dump(results, yaml_file, default_flow_style=False)

def main():
    # Specify the directory containing the `.sh` files
    scripts_directory = "./"  # Change this if `.sh` files are in another directory
    output_yaml = "test_suite_results.yml"
    
    # Run the shell scripts and collect data
    results = run_shell_scripts_and_collect_data(scripts_directory)
    
    # Save the results to a YAML file
    save_results_to_yaml(results, output_yaml)
    print(f"Test suite results saved to {output_yaml}")

if __name__ == "__main__":
    main()
