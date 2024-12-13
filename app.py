from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

N_CTX = 2048
N_THREADS = 8
MAX_TOKENS = 512

app = FastAPI(debug=True)


cache_dir = "./cache"


def load_model() -> tuple:
    checkpoint = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    model = Llama.from_pretrained(
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            repo_id=checkpoint,
            filename="tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",  # 3.8GB RAM required
        )
    return model


# device = "cuda"
device = "cpu"

model = load_model()

class WrappedMessage(BaseModel):
    command: str
    exit_code: int
    stderr: str


class ErrorResponse(BaseModel):
    explanation: str


@app.post("/error")
def error(wrapped_message: WrappedMessage) -> ErrorResponse:
    super_prompt = f"""
When running {wrapped_message.command},
it failed with exit code {wrapped_message.exit_code}.
The error message was: {wrapped_message.stderr}.
Help me fix it.
"""
    system_prompt = "You are a pragmatic and efficiency-focused coding assistant who specializes in finding elegant solutions to problems. Beyond just fixing errors, you suggest optimizations, explain trade-offs, and recommend best practices for long-term maintainability. Your advice is actionable, with a keen eye for performance and code clarity."
    preprompt_user_prompt = """
    When running ls notfolder.
    It failed with exit code 1.
    The error message was: 'ls: cannot access ''inexistant_folder'': No such file or directory'.
    Help me fix it."""

    preprompt_assistant_prompt = """
    The ls command is used to list files. The given argument was 'notfolder', the command tried to list the files in the notfolder directory. 
    You should verify if inexistant_folder exist and is a folder
    If it isn't you can fix this by using mkdir inexistant_folder and execute ls notfolder
    """

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": preprompt_user_prompt},
            {"role": "assistant", "content": preprompt_assistant_prompt},
            {"role": "user", "content": super_prompt},
        ]
    llama_response = model.create_chat_completion(
        messages,
        max_tokens=MAX_TOKENS,
        stop=["</s>"],
    )
    llama_response = llama_response["choices"][0]["message"]["content"]
    return ErrorResponse(explanation=llama_response)
