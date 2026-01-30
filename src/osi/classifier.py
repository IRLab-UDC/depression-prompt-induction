import dspy
from typing import Literal


class SymptomClassification(dspy.Signature):
    """You are a clinical assistant analyzing text for depression symptoms (BDI-II).

Task: Determine if the text is specifically about the specified symptom dimension.

Answer YES if: The text specifically discusses or provides evidence about THIS PARTICULAR symptom (whether the symptom is present or absent). The sentence must be directly relevant to this specific symptom, not just generally depression-related.
Answer NO if: The text is about a DIFFERENT symptom, or is completely unrelated to this symptom dimension."""

    symptom_name: str = dspy.InputField(desc="The name of the symptom to check for")
    symptom_definition: str = dspy.InputField(desc="The clinical definition of the symptom")
    text: str = dspy.InputField(desc="The text to analyze for this specific symptom")

    answer: Literal["YES", "NO"] = dspy.OutputField(desc="YES if the text is specifically about this particular symptom, NO if it's about a different symptom or unrelated")
