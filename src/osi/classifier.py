import dspy
from typing import Literal


class SymptomClassification(dspy.Signature):
    """You are a clinical assistant analyzing text for depression symptoms (BDI-II).

Task: Determine if the text indicates the person is CURRENTLY experiencing the specified symptom.

Answer YES if: The text explicitly expresses or clearly implies the person is experiencing this symptom now.
Answer NO if: The text is unrelated, describes past events only, or describes a different symptom."""

    symptom_name: str = dspy.InputField(desc="The name of the symptom to check for")
    symptom_definition: str = dspy.InputField(desc="The clinical definition of the symptom")
    text: str = dspy.InputField(desc="The text to analyze for symptom presence")

    answer: Literal["YES", "NO"] = dspy.OutputField(desc="YES if the person is currently experiencing the symptom, NO otherwise")
