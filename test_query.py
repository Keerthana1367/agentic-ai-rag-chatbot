import json
from src.graph import run_query

print('--- 6 pillars query ---')
res = run_query('What are the six core pillars of an Agentic AI system, from perception to execution?')
print('Answer:', res['answer'])
print('--- CHUNKS ---')
for i, c in enumerate(res['retrieved_chunks']):
    print(f"Chunk {i}:\n{c['text']}\n---\n")
