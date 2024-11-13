from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(debug=True)


cache_dir = "./cache"


def load_model() -> tuple:
    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)
    return tokenizer, model


# device = "cuda"
device = "cpu"

tokenizer, model = load_model()
model.to(device)


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
    messages = [{"role": "user", "content": super_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        min_new_tokens=100,
        max_new_tokens=250,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        early_stopping=True,
    )

    # only take the assistant's response
    response = (
        tokenizer.decode(outputs[0])
        .split("<|im_start|>assistant")[-1]
        .strip()
        .split("<|im_end|>")[0]
        .strip()
    )
    return ErrorResponse(explanation=response)
