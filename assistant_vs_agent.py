import json
import sys
from pathlib import Path


# Don't write next line
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from groq_client import ask, extract_json, run


REQUEST = "I have 3 days to revise. Plan my revision and check what I already know."


AGENT_PROMPT = f"""\
You are an agent. Break the request into at most 3 steps.
Available tools: check_progress(topic), make_schedule(days), ask_user(question)


Request: {REQUEST}


Return ONLY JSON:
{{"goal": "...", "steps": [{{"step" : 1,"tool": "...", "arguments": {{}}}}]}}
"""


def main() -> None:
    print("------- 1. As an assistant: one reply, no tools -------")
    print(
        ask(
            f"Answer this request in three sentences: \n{REQUEST}",
            temperature=0.3,
            max_tokens=200,
        )
    )
   
    print("------- 2. As an agent: a plan you can inspect -------")
    plan = extract_json(
        ask(AGENT_PROMPT, json_mode=True, temperature=0.0, max_tokens=500)
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False))


    print("------- 3. Spot the difference -------")
    steps = plan.get("steps", [])
    tools = {step.get("tool") for step in steps}
    print("Assistant: 1 answer, 0 tools, nothing to check or retry.")
    print(f"Agent: {len(steps)} steps, {len(tools)} tools, every step visible.")


if __name__ == "__main__":
    run(main)




