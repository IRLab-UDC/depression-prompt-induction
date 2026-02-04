import dspy
from typing import Literal, Dict
import json


class SymptomClassification(dspy.Signature):
    """Determine if the sentence is SPECIFICALLY ABOUT this particular symptom dimension.

SPECIFICALLY ABOUT THIS SYMPTOM IF the sentence:
- Directly discusses whether this symptom is present or absent
- Provides evidence relevant to this specific symptom

NOT ABOUT THIS SYMPTOM IF the sentence:
- Is about a different symptom
- Is completely unrelated to this symptom dimension"""

    symptom_name: str = dspy.InputField(desc="The symptom to check for")
    symptom_definition: str = dspy.InputField(desc="Clinical definition of the symptom")
    text: str = dspy.InputField(desc="Sentence to classify")
    answer: Literal["YES", "NO"] = dspy.OutputField(desc="YES if specifically about this symptom, NO otherwise")


def create_custom_signature(symptom_key: str, si_instructions: str) -> type:
    docstring = f"""Classify if the sentence is specifically about this symptom using these guidelines:

{si_instructions}

Based on these guidelines, determine if the sentence is specifically about this symptom."""

    class CustomSymptomClassification(dspy.Signature):
        __doc__ = docstring
        symptom_name: str = dspy.InputField(desc="The symptom to check for")
        symptom_definition: str = dspy.InputField(desc="Clinical definition of the symptom")
        text: str = dspy.InputField(desc="Sentence to classify")
        answer: Literal["YES", "NO"] = dspy.OutputField(desc="YES if specifically about this symptom, NO otherwise")

    CustomSymptomClassification.__name__ = f"{symptom_key}Classification"
    return CustomSymptomClassification


def load_custom_signatures(si_json_path: str = "data/si.json") -> Dict[str, type]:
    with open(si_json_path) as f:
        si_data = json.load(f)
    return {k: create_custom_signature(k, v.get("si", "")) for k, v in si_data.items()}
