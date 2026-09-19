import json
from src.graph import run_query

res = run_query("What are the six core pillars of an Agentic AI system, from perception to execution?")
with open("test_final_output.txt", "w", encoding="utf-8") as f:
    f.write('Answer:\n')
    f.write(res['answer'])
    f.write('\n\nChunks:\n')
    for i, c in enumerate(res['retrieved_chunks']):
        f.write(f"Chunk {i} (page {c.get('page')}): {repr(c['text'][:100])}\n")
