import re
import json
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")


DEFAULT_SYSTEM = (
    "You're a patient AI trainer running a 3-Day GEN-AI workshop "
    "for college students. Explain concepts in plain English, "
    "be concrete, use short paragraphs and bullets, "
    "and keep answers under 200 words."
)


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class GroqSetupError(RuntimeError):
    """Raised when the API key or the groq package is missing."""


def _api_key() -> str:
    for name in ("GROQ_API_KEY", "GROQ_KEY", "GROQ_API_TOKEN"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()

    raise GroqSetupError(
        "No Groq API Key found. Add a line like\n"
        "   GROQ_API_KEY=gsk_your_key_here\n"
        f"to {PROJECT_ROOT / '.env'} and run the script again."
    )


def get_client():
    try:
        from groq import Groq
    except ImportError as exc:
        raise GroqSetupError(
            "The groq package is not installed. "
            "Run pip install -r requirements.txt"
        ) from exc

    return Groq(api_key=_api_key())


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 500,
    json_mode: bool = False,
    tools: list[dict] | None = None,
):
    client = get_client()
    model_name = model or DEFAULT_MODEL

    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if "gpt-oss" in model_name:
        kwargs["reasoning_effort"] = REASONING_EFFORT

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return client.chat.completions.create(**kwargs)


def ask(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 500,
    json_mode: bool = False,
) -> str:

    messages: list[dict] = []

    if system or json_mode:
        messages.append({
            "role": "system",
            "content": system or DEFAULT_SYSTEM
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )

    return (response.choices[0].message.content or "").strip()

def extract_json(text: str):
    """Parse JSON out of a model reply, tolerating ``` fences and small chatter."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        return json.loads(cleaned[start : end + 1])







def run(main) -> None:
    try:
        main()

    except GroqSetupError as exc:
        print(f"\n[setup error] {exc}\n")
        raise SystemExit(1)

    except KeyboardInterrupt:
        print("\nStopped.")
