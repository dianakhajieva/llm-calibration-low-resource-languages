"""
Model provider interfaces for the multilingual calibration project.

This module defines a common interface for all LLM providers used in the
evaluation pipeline.

Each provider must implement the generate() method, which takes a prompt
and returns the model's raw text response.

Actual API integrations will be implemented later.
"""

from abc import ABC, abstractmethod
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


class BaseProvider(ABC):
    """
    Abstract base class for all language model providers.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a response from the model.

        Parameters
        ----------
        prompt : str
            Prompt sent to the language model.

        temperature : float
            Sampling temperature.

        max_new_tokens : int
            Maximum number of generated tokens.

        Returns
        -------
        str
            Raw model response.
        """
        pass


class DummyProvider(BaseProvider):
    """
    Fake model used to test the evaluation pipeline.
    """

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:

        return """ANSWER: B
CONFIDENCE: 87"""

class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models."""    

    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        return response.choices[0].message.content

class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)

        from google import genai
        from google.genai import types

        self.types = types

        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,

                thinking_config=self.types.ThinkingConfig(
                    thinking_budget=0
                ),

                response_mime_type="application/json",

                response_schema={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"]
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100
                        }
                    },
                    "required": [
                        "answer",
                        "confidence"
                    ]
                }
            ),
        )

        return response.text
    
class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def __init__(self, model_name:str):
        super().__init__(model_name)

        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.content[0].text

class HuggingFaceProvider(BaseProvider):
    """Provider for Hugging Face Inference API."""

    def __init__(self, model_name: str):
        super().__init__(model_name)

        from huggingface_hub import InferenceClient

        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:

        response = self.client.chat_completion(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        return response.choices[0].message.content


def get_provider(provider_name: str, model_name: str) -> BaseProvider:
    """
    Factory function that returns the requested provider.

    Parameters
    ----------
    provider_name : str
        Name of the provider
        ("openai", "google", "anthropic", "huggingface")

    model_name : str
        Name of the model.

    Returns
    -------
    BaseProvider
        Provider instance.
    """

    providers = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "dummy": DummyProvider,
    }

    provider_name = provider_name.lower()

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {list(providers.keys())}"
        )

    return providers[provider_name](model_name)