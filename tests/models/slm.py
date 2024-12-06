from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE='cpu'
MAX_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.9

class SLM:
    def __init__(self) -> None:
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        """
        Load the necessary model files. (may be tokenizer, model, etc.)
        """
        raise NotImplementedError()

    def parse_response(self, response: str) -> str:
        """
        Parse the response from the model to remove any unwanted tokens or boilerplate text.

        :param response: The raw response from the model.
        :return: The cleaned response.

        Example (to adapt according to the model's output):
        response = "<system> You are a great assistant! <system><user> Help me create a shopping list. <user><response> Sure! Here is a shopping list for you: <response>"
        _parse_response(response) -> "Sure! Here is a shopping list for you:"
        """
        raise NotImplementedError()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response for the given user prompt.

        :param system_prompt: The system's prompt.
        :param user_prompt: The user's prompt.
        :return: The model's response.

        Example (to adapt according to the model's input):
        system_prompt = "You are a great assistant!"
        user_prompt = "Help me create a shopping list."
        get_response(user_prompt) -> "<system> You are a great assistant! <system><user> Help me create a shopping list. <user><response> Sure! Here is a shopping list for you: <response>"
        """
        raise NotImplementedError()

# ============================== SMOLLM2 135M INSTRUCT ==============================

class SmolLM2135MInstruct(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

    def parse_response(self, response: str) -> str:
        return (
            response.split("<|im_start|>assistant")[-1]
            .strip()
            .split("<|im_end|>")[0]
            .strip()
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(device=DEVICE)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

class SmolLM2135MInstructGGUF(SLM):
    def __init__(self):
        super().__init__()
        self.model_path = "MaziyarPanahi/SmolLM2-135M-Instruct-GGUF"
        self.cache_dir = "./cache"
    
    def load_model(self):
        self.model = Llama.from_pretrained(self.model_path, filename="SmolLM2-135M-Instruct.Q2_K.gguf")
    
    def get_response(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        outputs = self.model.create_chat_completion(
            messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return outputs
    
    def parse_response(self, response: str) -> str:
        return response

# ============================== YI CODER 1.5B CHAT ==============================

class YiCoder15BChat(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "01-ai/Yi-Coder-1.5B-Chat"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(device=DEVICE)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

    def parse_response(self, response: str) -> str:
        return (
            response.split("<|im_start|>assistant")[-1]
            .strip()
            .split("<|im_end|>")[0]
            .strip()
        )

class YiCoder15BChatGGUF:
    def __init__(self):
        self.model_path = "bartowski/Yi-Coder-1.5B-Chat-GGUF"
        self.cache_dir = "./cache"
    
    def load_model(self):
        self.model = Llama.from_pretrained(
            repo_id=self.model_path,
            filename="Yi-Coder-1.5B-Chat-IQ2_M.gguf",
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        outputs = self.model.create_chat_completion(
            messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,            
        )
        return outputs
    
    def parse_response(self, response: str) -> str:
        print(response)
        return (
            response
        )

# ============================== ZEPHIR SMOL LLAMA 100M STF FULL ==============================

class ZephirSmolLlama100mStfFull(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "amazingvince/zephyr-smol_llama-100m-sft-full"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

    def parse_response(self, response: str) -> str:
        return (
            response.split("<|assistant|>")[-1]
            .strip()
            .split("</s>")[0]
            .strip()
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", padding=True)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=MAX_TOKENS,
            top_p=TOP_P,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response
    
class ZephirSmolLlama100mStfFullGGUF(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "afrideva/zephyr-smol_llama-100m-sft-full-GGUF"
        self.cache_dir = "./cache"
        self.model_filename = "zephyr-smol_llama-100m-sft-full.fp16.gguf"

    def load_model(self) -> None:
        self.model = Llama.from_pretrained(
            cache_dir = self.cache_dir,
            n_ctx=2048,
            n_threads=8, 
            repo_id=self.model_name,
            filename=self.model_filename
        )

    def parse_response(self, response: str) -> str:
        return response

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        llama_response = self.model(
            prompt, 
            max_tokens=MAX_TOKENS,
            stop=["</s>"],
        )
        response = llama_response["choices"][0]["text"]

        return response

# ============================== DEEPSEEK CODER 1.3B BASE ==============================

class DeepseekCoderBase(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "deepseek-ai/deepseek-coder-1.3b-base"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir, trust_remote_code=True,
        )

    def parse_response(self, response: str) -> str:
        return (
            response.split("A:")[-1]
            .strip()
            .split("<｜end▁of▁sentence｜>")[0]
            .strip()
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        inputs = self.tokenizer(user_prompt, return_tensors="pt").to(device=DEVICE)
        outputs = self.model.generate(**inputs, max_length=MAX_TOKENS, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(outputs[0])
        return response

class DeepSeekCoderBaseGGUF(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "TheBloke/deepseek-coder-1.3b-base-GGUF"
        self.cache_dir = "./cache"
        self.model_filename = "deepseek-coder-1.3b-base.Q5_K_M.gguf"

    def load_model(self) -> None:
        self.model = Llama.from_pretrained(
            cache_dir = self.cache_dir,
            n_ctx=2048,
            n_threads=8,
            repo_id=self.model_name,
            filename=self.model_filename
        )

    def parse_response(self, response: str) -> str:
        return response.split("A:")[-1].strip()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        prompt = user_prompt
        llama_response = self.model(
            prompt, 
            max_tokens=MAX_TOKENS,
            stop=["</s>"],
        )
        response = llama_response["choices"][0]["text"]
        return response

# ============================== TINYLLAMA 1.1B CHAT V1.0 ==============================

class TinyLlamaChat(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

    def parse_response(self, response: str) -> str:
        return (
            response.split("<|assistant|>")[1]
            .strip()
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role" : "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

class TinyLlamaChatGGUF(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.model = Llama.from_pretrained(
            n_ctx=2048,
            n_threads=8,
            repo_id=self.model_path,
            filename="tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", # 3.8GB RAM required
        )

    def parse_response(self, response: str) -> str:
        return response["choices"][0]["text"].split("<|assistant|>")[1].strip()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        llama_response = (
            self.model(
                prompt,
                max_tokens=MAX_TOKENS,
                stop=["</s>"],
                echo=True,
            )
        )
        return llama_response

# ============================== PHI 3.5 MINI INSTRUCT ==============================

class PhiMiniInstruct(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "microsoft/Phi-3.5-mini-instruct"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, cache_dir=self.cache_dir
        )

    def parse_response(self, response: str) -> str:
        return (
            response.split("<|im_start|>assistant")[-1]
            .strip()
            .split("<|im_end|>")[0]
            .strip()
        )

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role" : "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

class PhiMiniInstructGGUF(SLM):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "tensorblock/Phi-3.5-mini-instruct-GGUF"
        self.cache_dir = "./cache"

    def load_model(self) -> None:
        self.model = Llama.from_pretrained(
            n_ctx=2048,
            n_threads=8,
            repo_id=self.model_path,
            filename="Phi-3.5-mini-instruct-Q5_K_M.gguf", # 3.8GB RAM required
        )

    def parse_response(self, response: str) -> str:
        return response["choices"][0]["text"].split("<|assistant|>")[1].strip()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        llama_response = (
            self.model(
                prompt,
                max_tokens=MAX_TOKENS,
                stop=["</s>"],
                echo=True,
            )
        )
        return llama_response

model_to_class = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": SmolLM2135MInstruct,
    "MaziyarPanahi/SmolLM2-135M-Instruct-GGUF": SmolLM2135MInstructGGUF,
    "01-ai/Yi-Coder-1.5B-Chat": YiCoder15BChat,
    "bartowski/Yi-Coder-1.5B-Chat-GGUF": YiCoder15BChatGGUF,
    "amazingvince/zephyr-smol_llama-100m-sft-full": ZephirSmolLlama100mStfFull,
    "afrideva/zephyr-smol_llama-100m-sft-full-GGUF": ZephirSmolLlama100mStfFullGGUF,
    "deepseek-ai/deepseek-coder-1.3b-base": DeepseekCoderBase,
    "TheBloke/deepseek-coder-1.3b-base-GGUF": DeepSeekCoderBaseGGUF,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TinyLlamaChat,
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": TinyLlamaChatGGUF,
    "microsoft/Phi-3.5-mini-instruct": PhiMiniInstruct,
    "tensorblock/Phi-3.5-mini-instruct-GGUF": PhiMiniInstructGGUF,
}
