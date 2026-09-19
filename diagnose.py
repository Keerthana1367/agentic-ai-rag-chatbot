import json
from src.graph import run_query

metadata = json.load(open('faiss_index/metadata.json', encoding='utf-8'))
print('--- Pillars Chunks (Searching for "The Core Pillars") ---')
for m in metadata:
    if 'The Core Pillars' in m['text']:
        print(f"chunk={m['chunk_id']} page={m['page']} text={repr(m['text'][:100])}")

print('\n--- Page 13 Chunks ---')
page_13_chunks = [m for m in metadata if m['page'] == 13]
for m in page_13_chunks[:3]:
    print(f"chunk={m['chunk_id']} page={m['page']} text={repr(m['text'][:100])}")

print('\n--- Retrieved Chunks for Query ---')
res = run_query("What are the six core pillars of an Agentic AI system, from perception to execution?")
for i, c in enumerate(res['retrieved_chunks'][:4]):
    print(f"Rank {i} | page={c['page']} | text={repr(c['text'][:100])}")
