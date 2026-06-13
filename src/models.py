"""
Wrappers around each model provider.

KEY IDEA: you are TESTING these models, not building them. Each wrapper exposes
two things:

  generate(prompt)                  -> the model's text reply (used by the
                                       'verbalized' confidence method)

  answer_with_token_logprob(prompt) -> (letter, probability), where probability
                                       is the model's own confidence in that
                                       letter (used by the 'logprob' method)

Providers that cannot give token probabilities raise NotImplementedError for the
second method; for those, only run the 'verbalized' method.

API keys are read from environment variables (see .env.example). We import each
SDK lazily inside its class, so you only need to install the ones you actually use.
"""
import os
import math


class BaseModel:
    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        raise NotImplementedError

    def answer_with_token_logprob(self, prompt, option_letters=("A", "B", "C", "D")):
        raise NotImplementedError(
            "This provider does not expose token probabilities. "
            "Use the 'verbalized' method for this model instead."
        )


class OpenAIModel(BaseModel):
    def __init__(self, model_name):
        from openai import OpenAI  # pip install openai
        self.client = OpenAI()      # reads OPENAI_API_KEY from the environment
        self.model_name = model_name

    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    def answer_with_token_logprob(self, prompt, option_letters=("A", "B", "C", "D")):
        # Ask for only the letter, and request token logprobs.
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
            logprobs=True,
            top_logprobs=5,
        )
        text = resp.choices[0].message.content or ""
        # The first generated token should be the answer letter (the prompt asks for that).
        content_tokens = resp.choices[0].logprobs.content
        prob = float("nan")
        if content_tokens:
            first = content_tokens[0]
            prob = math.exp(first.logprob)  # convert log-probability to probability
        return text.strip(), prob


class GoogleModel(BaseModel):
    def __init__(self, model_name):
        import google.generativeai as genai  # pip install google-generativeai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_new_tokens},
        )
        return resp.text
    # answer_with_token_logprob: inherits the NotImplementedError (no logprobs).


class AnthropicModel(BaseModel):
    def __init__(self, model_name):
        from anthropic import Anthropic  # pip install anthropic
        self.client = Anthropic()         # reads ANTHROPIC_API_KEY from the environment
        self.model_name = model_name

    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    # answer_with_token_logprob: inherits the NotImplementedError (no logprobs).


class HuggingFaceModel(BaseModel):
    """Runs an open-weight model locally. Needs a GPU for the larger models."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def _build_inputs(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return input_ids.to(self.model.device)

    def generate(self, prompt, temperature=0.0, max_new_tokens=256):
        input_ids = self._build_inputs(prompt)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def answer_with_token_logprob(self, prompt, option_letters=("A", "B", "C", "D")):
        torch = self.torch
        input_ids = self._build_inputs(prompt)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1, :]  # scores for the next token
        # Probability the model would emit each option letter as the next token,
        # normalized over just the four letters.
        letter_ids = []
        for letter in option_letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            letter_ids.append(ids[0])  # first sub-token of the letter
        letter_logits = logits[letter_ids]
        probs = torch.softmax(letter_logits, dim=-1)
        best = int(torch.argmax(probs))
        return option_letters[best], float(probs[best])


def get_model(model_cfg):
    """Build the right wrapper from a model entry in config.yaml."""
    provider = model_cfg["provider"]
    name = model_cfg["model_name"]
    if provider == "openai":
        return OpenAIModel(name)
    if provider == "google":
        return GoogleModel(name)
    if provider == "anthropic":
        return AnthropicModel(name)
    if provider == "huggingface":
        return HuggingFaceModel(name)
    raise ValueError(f"Unknown provider: {provider}")
