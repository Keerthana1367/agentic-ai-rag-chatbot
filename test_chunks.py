import json
from src.graph import run_query
res = run_query('What are the six core pillars of an Agentic AI system, from perception to execution?')
print('Confidence:', res['confidence'])
print('Number of chunks:', len(res['retrieved_chunks']))
for i, c in enumerate(res['retrieved_chunks']):
    print(f"Chunk {i}: {c['text'][:100].encode('ascii', 'ignore').decode('ascii')}")
