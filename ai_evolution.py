import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from groq_client import ask


print("\n" + "-"*50)
print("Topic: AI Evolution => AI -> ML -> DL -> GenAI")
print("\n" + "-"*50)


prompt = (
    "Explain the evolution from Artificial Intelligence to Machine Learning to Deep Learning to Generative AI in exactly 3 short paragraphs. Each paragraph must be max 2 sentences. Use very simple language for college students."
    "Paragraph 1: Artificial Intelligence. Paragraph 2: Machine Learning. Paragraph 3: Deep Learning. Paragraph 4: Generative AI."
)


result = ask(prompt, temperature=0.3, max_tokens=400)
print(result)
