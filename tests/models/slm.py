from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

MAX_NEW_TOKENS=512

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


class SmolLM135MInstruct(SLM):
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
        messages = [{"role": "user", "content": user_prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            min_new_tokens=100,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

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
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=0.9,
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
            max_tokens=MAX_NEW_TOKENS,
            stop=["</s>"],
            echo=False,
        )
        response = llama_response["choices"][0]["text"]

        return response
    
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
        inputs = self.tokenizer(user_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=MAX_NEW_TOKENS, num_return_sequences=1, no_repeat_ngram_size=2)
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
            max_tokens=MAX_NEW_TOKENS,
            stop=["</s>"],
            echo=True,
        )
        response = llama_response["choices"][0]["text"]
        return response

class TinyLlamaInstruct(SLM):
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
            # {"role" : "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            min_new_tokens=100,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response
    
class MicrosoftPhiInstruct(SLM):
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
            # {"role" : "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(
            inputs,
            min_new_tokens=100,
            max_new_tokens=250,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0])
        return response

class TheBlokeTinyLlamaInstruct(SLM):
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
        # Default system prompt
        return response["choices"][0]["text"].split("<|assistant|>")[1].strip()

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        
        if system_prompt == "" :
            system_prompt = "You are a highly skilled and approachable debugging assistant named CodeFix. You excel at identifying and resolving coding errors while providing clear, step-by-step guidance. You have a patient, problem-solving mindset, and you strive to make debugging an empowering learning experience for the user. Your explanations are precise and beginner-friendly, with actionable advice tailored to the user's level of expertise."
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        llama_response = (
            self.model(
                prompt,
                max_tokens=512,
                stop=["</s>"],
                echo=True
            )
        )
        return llama_response

class TensorblockPhiInstruct(SLM):
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
        
        if system_prompt == "" :
            # Default system prompt
            system_prompt = "You are a highly skilled and approachable debugging assistant named CodeFix. You excel at identifying and resolving coding errors while providing clear, step-by-step guidance. You have a patient, problem-solving mindset, and you strive to make debugging an empowering learning experience for the user. Your explanations are precise and beginner-friendly, with actionable advice tailored to the user's level of expertise."
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
        llama_response = (
            self.model(
                prompt,
                max_tokens=512,
                stop=["</s>"],
                echo=True
            )
        )
        return llama_response

model_to_class = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": SmolLM135MInstruct,
    "amazingvince/zephyr-smol_llama-100m-sft-full": ZephirSmolLlama100mStfFull,
    "afrideva/zephyr-smol_llama-100m-sft-full-GGUF": ZephirSmolLlama100mStfFullGGUF,
    "deepseek-ai/deepseek-coder-1.3b-base": DeepseekCoderBase,
    "TheBloke/deepseek-coder-1.3b-base-GGUF": DeepSeekCoderBaseGGUF,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": TinyLlamaInstruct,
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": TheBlokeTinyLlamaInstruct,
    "tensorblock/Phi-3.5-mini-instruct-GGUF": TensorblockPhiInstruct,
    "microsoft/Phi-3.5-mini-instruct": MicrosoftPhiInstruct
}
