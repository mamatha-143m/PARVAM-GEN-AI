import ast
import operator
import sys
from pathlib import Path

from groq_client import ask, run

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}