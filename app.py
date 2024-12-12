import json
import logging
import yaml

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import List

from src.slm import model_to_class

# Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)

cache_dir = "./cache"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST and other methods
    allow_headers=["*"],  # Allow all headers
)

# Global variable
SYSTEM_PROMPTS_PATH = "./tests/system_prompts.yml"
PRE_PROMPTS_PATH = "./tests/pre_prompts.yml"
model = None
checkpoint = None
system_prompt = ""
messages = None


def default_load_model():
    """
    Load the default model and return its instance.

    This function sets the global variable `checkpoint` to a predefined model name,
    retrieves the corresponding model class from the `model_to_class` mapping,
    and initializes the model by loading it. The initialized model instance is
    then returned.

    Returns:
        tuple: A tuple containing the initialized model instance.
    """
    global checkpoint

    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_class = model_to_class.get("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = model_class()
    model.load_model()
    return model


model = default_load_model()


class Message(BaseModel):
    role: str
    content: str


class ModelMessage(BaseModel):
    model: str
    messages: List[Message]
    system_prompt: str


class WrappedMessage(BaseModel):
    command: str
    exit_code: int
    stderr: str


class ErrorResponse(BaseModel):
    explanation: str


# Store connected WebSocket clients
active_connections: List[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles WebSocket connections at the `/ws` endpoint.

    This function accepts a WebSocket connection, adds it to the active connections list,
    and listens for incoming messages. When the connection is closed, it removes the
    WebSocket from the list and logs any errors.

    Args:
        websocket (WebSocket): The WebSocket connection object.
    """
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect as e:
        logger.error(f"Error connexion to WebSocket: {e}")
        active_connections.remove(websocket)


@app.get("/system_prompts")
def get_system_prompts() -> list[dict]:
    """
    Retrieve a list of predefined system prompts.

    This endpoint reads from a YAML file and returns a collection of system prompts.
    Each prompt includes an ID, a type, and a detailed description.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains:
            - "id" (int): The unique identifier of the prompt.
            - "type" (str): A short description of the prompt's personality or tone.
            - "prompt" (str): The detailed text of the system prompt.
    """
    with open(SYSTEM_PROMPTS_PATH, "r") as f:
        system_prompts = yaml.safe_load(f)
    return system_prompts


@app.get("/pre_prompts")
def get_pre_prompts() -> list[dict]:
    """
    Retrieve a list of predefined pre-prompts.

    This endpoint reads from a YAML file and returns a collection of pre-prompts.
    Each pre-prompt includes an ID, a user prompt, and the corresponding assistant's response.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains:
            - "id" (int): The unique identifier of the pre-prompt.
            - "user_prompt" (str): The user's prompt with details about the issue.
            - "assistant_response" (str): The assistant's suggested response to fix the issue.
    """
    with open(PRE_PROMPTS_PATH, "r") as f:
        pre_prompts = yaml.safe_load(f)
    return pre_prompts


@app.get("/model_info")
def get_model_info() -> dict:
    """
    Retrieve and return the current model's information.

    This function accesses the global variables `checkpoint`,
    `system_prompt`, and `messages` to provide details about the
    currently loaded model, its system prompt, and related messages.

    Returns:
        dict: A dictionary containing:
            - "checkpoint": The name of the current model's checkpoint.
            - "system_prompt": The system prompt associated with the model.
            - "messages": The messages relevant to the current model session.
    """
    global checkpoint, system_prompt, messages
    return {
        "checkpoint": checkpoint,
        "system_prompt": system_prompt,
        "messages": messages,
    }


@app.get("/models")
def get_models() -> list[str]:
    """
    Retrieve a list of available model names.

    This endpoint returns a list of model names from the `model_to_class` mapping,
    which represents the available models in the system.

    Returns:
        list[str]: A list of strings, where each string is a model name.
    """
    return list(model_to_class.keys())


@app.post("/error")
async def error(wrapped_message: WrappedMessage) -> ErrorResponse:
    """
    Handle error reporting and send model response to connected clients.

    This endpoint receives a wrapped message containing information about an error,
    constructs a super prompt for the model, retrieves a response from the model,
    and sends the response to all active WebSocket connections. It also returns an
    `ErrorResponse` containing the model's explanation for the error.

    Args:
        wrapped_message (WrappedMessage): A wrapped message containing the error details,
                                          including the command, exit code, and stderr.

    Returns:
        ErrorResponse: The response containing the model's explanation for the error.
    """
    global checkpoint, active_connections, system_prompt, messages, model

    super_prompt = f"""
        When running {wrapped_message.command},
        it failed with exit code {wrapped_message.exit_code}.
        The error message was: {wrapped_message.stderr}.
        Help me fix it.
    """

    logger.info(model)
    response = model.get_response(system_prompt=system_prompt, user_prompt=super_prompt, messages=messages)
    response = model.parse_response(response=response)

    message = {
        "type": "model_response",
        "super_prompt": super_prompt.strip(),
        "response": response.strip(),
    }

    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending model info to client: {e}")
            active_connections.remove(connection)

    return ErrorResponse(explanation=response)


# Endpoint to load a new model
@app.post("/load_model")
async def load_model_web(model_message: ModelMessage):
    """
    Load a new model based on the provided model message.

    This endpoint receives a model message with details about the model to load,
    including the model checkpoint, system prompt, and optional messages. It then
    loads the specified model, updates the global state, and sends model information
    to all active WebSocket connections. A success or error response is returned.

    Args:
        model_message (ModelMessage): A message containing the model checkpoint,
                                      system prompt, and optional messages.

    Returns:
        JSONResponse: A response indicating whether the model was loaded successfully
                      or if an error occurred.
    """
    global model, checkpoint, messages, system_prompt

    checkpoint = model_message.model
    system_prompt = model_message.system_prompt

    if model_message.messages != []:
        messages = model_message.messages
    else:
        messages = None

    try:

        model_class = model_to_class.get(checkpoint)
        model = model_class()
        model.load_model()
        logger.info(f"Model {checkpoint} loaded successfully")

        model_info = {
            "type": "model_info",
            "checkpoint": checkpoint,
            "system_prompt": system_prompt,
            "messages": [m.model_dump() for m in messages] if messages != None else None,
        }
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(model_info))
            except Exception as e:
                active_connections.remove(connection)
                logger.error(f"Error sending model info to client: {e}")

        return JSONResponse(content={"message": f"Model {checkpoint} loaded successfully"})
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": f"Failed to load model {checkpoint}: {str(e)}"}, status_code=400)
