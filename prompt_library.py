
import argparse
import sys
from pathlib import Path

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from groq_client import ask, extract_json,run

# Prompt Library
LIBRARY = {
    "exam_plan": {
        "category": "Study",
        "tip": "Change the number of days to adjust the plan density.",
        "prompt": """\
ROLE: You are a study coach for engineering students.

TASK: Build a {days}-day revision plan for {topic}.

CONTEXT: The student has 2 hours a day and one weak area: {weak_area}.

CONSTRAINTS:
- Create a day-by-day table.
- Include active recall tasks only.
- Divide study time into 30-minute blocks.

OUTPUT:
Markdown table with columns:
Day | Focus | Active Task | Check
"""
    },

    "summarise": {
        "category": "Research",
        "tip": "Ask for citations to source paragraphs to improve accuracy.",
        "prompt": """\
ROLE: You are a research assistant who never invents facts.

TASK: Summarise the text below in 5 bullet points and 2 open questions.

CONSTRAINTS:
- Keep all numbers exact.
- Mark anything unclear as "not stated".

TEXT:
{text}
"""
    },

    "explain_simply": {
        "category": "Study",
        "tip": "Change the age to adjust the explanation depth.",
        "prompt": """\
ROLE: You are a science communicator.

TASK: Explain {topic} to a 12-year-old.

CONSTRAINTS:
- Maximum 120 words.
- Include one everyday analogy.
- Do not use formulas.

OUTPUT:
An analogy paragraph, followed by three bullet points
and one quiz question.
"""
    },
}


# Fill the prompt template safely
def fill(template: str, values: dict[str, str]) -> str:

    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template.format_map(SafeDict(values))


# Main function
def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run a prompt from the prompt library."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Show all available prompts"
    )

    parser.add_argument(
        "--name",
        default="exam_plan",
        help="Name of the prompt to execute"
    )

    parser.add_argument(
        "--topic",
        default="Database Normalization",
        help="Topic for the prompt"
    )

    parser.add_argument(
        "--text",
        default="Paste your source text here.",
        help="Source text for summarisation"
    )

    parser.add_argument(
        "--days",
        default="7",
        help="Number of days for the study plan"
    )

    parser.add_argument(
        "--subject",
        default="Computer Science",
        help="Subject name"
    )

    parser.add_argument(
        "--weak-area",
        default="Normalization and Joins",
        help="Your weak study area"
    )

    args = parser.parse_args()

    # Display available prompts
    print("\nAVAILABLE PROMPTS")
    print("-" * 50)

    for name, entry in LIBRARY.items():
        print(
            f"- {name:<16} "
            f"[{entry['category']}] "
            f"{entry['tip']}"
        )

    # Stop if user only wants to list prompts
    if args.list:
        return

    # Check whether the requested prompt exists
    if args.name not in LIBRARY:
        print(
            f"\nUnknown prompt: '{args.name}'"
            f"\nAvailable options: {', '.join(LIBRARY.keys())}"
        )
        raise SystemExit(2)

    # Select the prompt
    entry = LIBRARY[args.name]

    # Fill the prompt with user-provided values
    prompt = fill(
        entry["prompt"],
        {
            "topic": args.topic,
            "text": args.text,
            "days": args.days,
            "subject": args.subject,
            "weak_area": args.weak_area,
        },
    )

    # Display the generated prompt
    print("\nGENERATED PROMPT")
    print("-" * 50)
    print(prompt)

    # Send prompt to Groq
    print("\nAI RESPONSE")
    print("-" * 50)

    response = ask(
        prompt,
        temperature=0.4,
        max_tokens=1000
    )

    print(response)


# Execute the program
if __name__ == "__main__":
    run(main)
