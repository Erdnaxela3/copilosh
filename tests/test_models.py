import pytest
from models.slm import model_to_class

system_prompt = "You are a highly skilled and approachable debugging assistant named CodeFix. You excel at identifying and resolving coding errors while providing clear, step-by-step guidance. You have a patient, problem-solving mindset, and you strive to make debugging an empowering learning experience for the user. Your explanations are precise and beginner-friendly, with actionable advice tailored to the user's level of expertise."

user_prompt = """
When running ls notfolder.
it failed with exit code 1.
The error message was: 'ls: cannot access ''inexistant_folder'': No such file or directory'.
Help me fix it.
"""


def test_model_to_class():
    for model_name, model_cls in model_to_class.items():
        try:
            model = model_cls()
            model.load_model()
            response = model.get_response(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
        except Exception as e:
            pytest.fail(f"Model {model_name} failed with error: {e}")
