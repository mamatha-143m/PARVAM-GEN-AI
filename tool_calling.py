
import ast
import operator
import json
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from groq_client import chat


# ==================================================
# 1. CALCULATOR
# ==================================================

def calculator(a: float, b: float, operation: str) -> dict:

    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }

    if operation not in operations:
        return {"error": "Unsupported operation"}

    if operation == "divide" and b == 0:
        return {"error": "Cannot divide by zero"}

    result = operations[operation](a, b)

    return {
        "a": a,
        "b": b,
        "operation": operation,
        "result": result,
    }


# ==================================================
# 2. TEMPERATURE CONVERTER
# ==================================================

def convert_temperature(
    value: float,
    from_unit: str,
    to_unit: str
) -> dict:

    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    if from_unit == "C":
        celsius = value

    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9

    elif from_unit == "K":
        celsius = value - 273.15

    else:
        return {"error": "Invalid source temperature unit"}

    if celsius < -273.15:
        return {"error": "Temperature is below absolute zero"}

    if to_unit == "C":
        result = celsius

    elif to_unit == "F":
        result = (celsius * 9 / 5) + 32

    elif to_unit == "K":
        result = celsius + 273.15

    else:
        return {"error": "Invalid target temperature unit"}

    return {
        "original_value": value,
        "from": from_unit,
        "to": to_unit,
        "result": round(result, 2),
    }


# ==================================================
# 3. MATHEMATICAL EXPRESSION SOLVER
# ==================================================

def solve_expression(expression: str) -> dict:

    """
    Safely evaluate mathematical expressions.
    Uses ^ as the power operator.
    Does not use eval().
    """

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def calculate(node):

        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant):

            if type(node.value) not in (int, float):
                raise ValueError("Only numbers are allowed")

            return node.value

        if isinstance(node, ast.UnaryOp):

            value = calculate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return value

            if isinstance(node.op, ast.USub):
                return -value

            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):

            if isinstance(node.op, ast.BitXor):
                op_type = ast.Pow
            else:
                op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Unsupported operator")

            left = calculate(node.left)
            right = calculate(node.right)

            if op_type == ast.Div and right == 0:
                raise ValueError("Cannot divide by zero")

            if op_type == ast.Pow and abs(right) > 10:
                raise ValueError("Exponent is too large")

            result = allowed_operators[op_type](left, right)

            if not isinstance(result, (int, float)):
                raise ValueError("Invalid result")

            if abs(result) > 1e100:
                raise ValueError("Result is too large")

            return result

        raise ValueError("Invalid mathematical expression")

    try:

        tree = ast.parse(expression, mode="eval")
        result = calculate(tree)

        return {
            "expression": expression,
            "result": result,
        }

    except (
        ValueError,
        TypeError,
        SyntaxError,
        OverflowError,
        ZeroDivisionError
    ) as error:

        return {"error": str(error)}


# ==================================================
# 4. EMAIL CHECKER
# ==================================================

EMAILS = [

    {
        "sender": "admin@college.edu",
        "subject": "Exam timetable released",
        "unread": True,
    },

    {
        "sender": "placement@company.com",
        "subject": "Interview scheduled for Monday",
        "unread": True,
    },

    {
        "sender": "library@college.edu",
        "subject": "Book return reminder",
        "unread": False,
    },

]


def check_mail(unread_only: bool = False) -> dict:

    emails = EMAILS

    if unread_only:

        emails = [
            email for email in EMAILS
            if email["unread"]
        ]

    return {

        "total_emails": len(emails),

        "unread_emails": sum(
            1 for email in EMAILS if email["unread"]
        ),

        "messages": emails,
    }


# ==================================================
# 5. WEATHER CHECKER
# ==================================================

WEATHER = {

    "bengaluru": {
        "temperature": 28,
        "condition": "Partly cloudy",
        "humidity": 65,
    },

    "chennai": {
        "temperature": 34,
        "condition": "Sunny",
        "humidity": 70,
    },

    "mumbai": {
        "temperature": 31,
        "condition": "Rainy",
        "humidity": 82,
    },

    "delhi": {
        "temperature": 24,
        "condition": "Clear",
        "humidity": 45,
    },

}


def get_weather(city: str) -> dict:

    weather = WEATHER.get(city.lower())

    if weather is None:
        return {"error": f"Weather unavailable for {city}"}

    return {
        "city": city.title(),
        **weather,
    }


# ==================================================
# 6. TOOL DEFINITIONS
# ==================================================

TOOLS = [

    {
        "type": "function",
        "function": {

            "name": "calculator",

            "description": "Perform basic arithmetic calculations.",

            "parameters": {

                "type": "object",

                "properties": {

                    "a": {"type": "number"},

                    "b": {"type": "number"},

                    "operation": {

                        "type": "string",

                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                        ],

                    },

                },

                "required": ["a", "b", "operation"],

            },

        },
    },

    {
        "type": "function",
        "function": {

            "name": "convert_temperature",

            "description": "Convert temperature between Celsius, Fahrenheit and Kelvin.",

            "parameters": {

                "type": "object",

                "properties": {

                    "value": {"type": "number"},

                    "from_unit": {
                        "type": "string",
                        "enum": ["C", "F", "K"],
                    },

                    "to_unit": {
                        "type": "string",
                        "enum": ["C", "F", "K"],
                    },

                },

                "required": [
                    "value",
                    "from_unit",
                    "to_unit",
                ],

            },

        },
    },

    {
        "type": "function",
        "function": {

            "name": "solve_expression",

            "description": "Solve mathematical expressions using +, -, *, /, %, and ^. Use ^ for exponentiation. Example: (10 + 20)^2",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {"type": "string"},

                },

                "required": ["expression"],

            },

        },
    },

    {
        "type": "function",
        "function": {

            "name": "check_mail",

            "description": "Check the mock email inbox and retrieve unread emails.",

            "parameters": {

                "type": "object",

                "properties": {

                    "unread_only": {"type": "boolean"},

                },

                "required": ["unread_only"],

            },

        },
    },

    {
        "type": "function",
        "function": {

            "name": "get_weather",

            "description": "Get mock weather information for a city.",

            "parameters": {

                "type": "object",

                "properties": {

                    "city": {"type": "string"},

                },

                "required": ["city"],

            },

        },
    },

]


# ==================================================
# 7. TOOL REGISTRY
# ==================================================

REGISTRY = {

    "calculator": calculator,

    "convert_temperature": convert_temperature,

    "solve_expression": solve_expression,

    "check_mail": check_mail,

    "get_weather": get_weather,

}


# ==================================================
# 8. MAIN FUNCTION
# ==================================================

def main():

    print("\n===================================")
    print("       PERSONAL AI ASSISTANT")
    print("===================================")

    question = input(
        "\nAsk your personal assistant: "
    )

    messages = [

        {
            "role": "system",

            "content": """
You are a helpful personal assistant.

You have access to five tools:

1. Calculator
2. Temperature converter
3. Expression solver
4. Email checker
5. Weather checker

Rules:

- Use tools whenever the user asks for information
  that requires one of these tools.

- Never invent calculation results.

- You can call multiple tools for one question.

- Use tool results to generate your final answer.

- If a tool does not support the request, explain that.

- Weather and email data are mock data.

- Use ^ as the exponentiation operator.

- Never replace ^ with Python's ** operator.
""",
        },

        {
            "role": "user",
            "content": question,
        },

    ]

    # STEP 1: MODEL DECIDES WHICH TOOLS TO USE

    print("\n1. MODEL DECIDES WHICH TOOLS TO USE")

    response = chat(
        messages,
        tools=TOOLS,
        temperature=0.0,
        max_tokens=500,
    )

    message = response.choices[0].message

    calls = message.tool_calls or []

    if not calls:

        print("\nAssistant:", message.content)

        return

    print(f"\nModel requested {len(calls)} tool(s).")

    # STEP 2: ADD TOOL CALLS TO MESSAGE HISTORY

    messages.append({

        "role": "assistant",

        "content": message.content or "",

        "tool_calls": [

            {
                "id": call.id,

                "type": "function",

                "function": {

                    "name": call.function.name,

                    "arguments": call.function.arguments,

                },

            }

            for call in calls

        ],

    })

    # STEP 3: PYTHON EXECUTES THE TOOLS

    print("\n2. PYTHON EXECUTES THE TOOLS")

    for call in calls:

        name = call.function.name

        try:

            arguments = json.loads(
                call.function.arguments or "{}"
            )

            print(f"\nTool requested: {name}")

            print(f"Arguments: {arguments}")

            if name not in REGISTRY:

                result = {"error": "Unknown tool"}

            else:

                result = REGISTRY[name](**arguments)

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
            OverflowError
        ) as error:

            result = {"error": str(error)}

        print(f"Tool result: {result}")

        messages.append({

            "role": "tool",

            "tool_call_id": call.id,

            "name": name,

            "content": json.dumps(result),

        })

    # STEP 4: MODEL GENERATES FINAL ANSWER

    print("\n3. MODEL GENERATES THE FINAL ANSWER")

    final_response = chat(

        messages,

        temperature=0.0,

        max_tokens=500,

    )

    print(
        "\nAssistant:",
        final_response.choices[0].message.content
    )


# ==================================================
# 9. PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()
