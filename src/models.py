"""
Model provider interfaces for the multilingual calibration project.

This module defines a common interface for all LLM providers used in the
evaluation pipeline.

Each provider must implement the generate() method, which takes a prompt
and returns the model's raw text response.

Actual API integrations will be implemented later.
"""

from abc import ABC, abstractmethod
import os
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from google.genai import types

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

        load_dotenv()

        # 1. Initialize a Client object instead of using genai.configure()
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # We store the model_name to pass it during the generate call
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:

        # 2. Use the client.models.generate_content method
        # 3. Use types.GenerateContentConfig for the parameters
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_new_tokens,
            )
        )

        return response.text

class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        raise NotImplementedError(
            "Anthropic provider has not been implemented yet."
        )


class HuggingFaceProvider(BaseProvider):
    """Provider for Hugging Face hosted or local models."""

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> str:
        raise NotImplementedError(
            "Hugging Face provider has not been implemented yet."
        )


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