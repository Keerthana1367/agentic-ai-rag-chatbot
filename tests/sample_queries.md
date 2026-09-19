# Sample Queries

> **How to populate this file:** Start the API (`uvicorn src.api:app --reload`),
> then run each `curl` command below. Paste the real responses in place of the
> placeholder blocks.

---

## 1. What is Agentic AI?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is Agentic AI?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```

---

## 2. What are the key components of an AI agent?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What are the key components of an AI agent?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```

---

## 3. How does Agentic AI differ from traditional AI?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"How does Agentic AI differ from traditional AI?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```

---

## 4. What industries can benefit from Agentic AI?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What industries can benefit from Agentic AI?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```

---

## 5. What are the challenges or risks of Agentic AI?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What are the challenges or risks of Agentic AI?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```

---

## 6. What is the role of LLMs in Agentic AI systems?

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is the role of LLMs in Agentic AI systems?\"}" | python -m json.tool
```

**Response:**
```json
<paste real output here>
```
