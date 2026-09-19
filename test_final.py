import json
from src.graph import run_query

res = run_query("What are the six core pillars of an Agentic AI system, from perception to execution?")
print('Answer:')
print(res['answer'])
print('\nChunks:')
for i, c in enumerate(res['retrieved_chunks']):
    print(f"Chunk {i} (page {c.get('page')}): {repr(c['text'][:100])}")
