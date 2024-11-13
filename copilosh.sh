function command_wrapper() {
    local status=$? # Capture the exit code
    local stderr_output

    if [ $status -ne 0 ]; then
        stderr_output=$( { eval "$LAST_COMMAND"; } 2>&1 )

        payload=$(jq -n \
            --arg command "$LAST_COMMAND" \
            --argjson exit_code "$status" \
            --arg stderr "$stderr_output" \
            '{command: $command, exit_code: $exit_code, stderr: $stderr}')

        response=$(curl -s -X POST http://localhost:8082/error \
             -H "Content-Type: application/json" \
             -d "$payload")

        explanation=$(echo "$response" | jq -r '.explanation')
        echo "$explanation"
    fi
}

# Trap DEBUG to capture the command before it executes and set PROMPT_COMMAND to run the wrapper after each command
trap 'LAST_COMMAND=$(history 1 | { read -r _ cmd; echo "$cmd"; })' DEBUG
PROMPT_COMMAND='command_wrapper'