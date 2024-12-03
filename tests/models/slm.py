from transformers import AutoModelForCausalLM, AutoTokenizer


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


model_to_class = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": SmolLM135MInstruct,
}
