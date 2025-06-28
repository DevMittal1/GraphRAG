````markdown
# Qdrant-Neo4j-GraphRAG

A lightweight Python client that combines vector search (Qdrant), graph search (Neo4j), and LLM-based reasoning (LangChain) into a unified GraphRAG pipeline.

---

## ✨ Features

- Ingest structured data into Neo4j and Qdrant
- Hybrid retrieval combining semantic and graph context
- Global web search fallback 
- Compatible with LangChain LLMs and tools

---

## 📦 Installation

Install using [uv](https://github.com/astral-sh/uv) with editable mode:

```bash
uv pip install -e .
````

---

## 🧠 Requirements

Tested with:

* Python 3.12.3
* `neo4j==5.28.1`
* `qdrant-client==1.14.3`
* `langchain==0.3.10`
* `pydantic==2.11.3`

---

## 🚀 Usage

```python
from qdrant_neo4j_graphrag import Graphragclient

client = Graphragclient(
    qdrant_url="http://localhost:6333",
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    llm=your_llm,
    qdrant_client=your_qdrant_client,
    embedding_model=your_embedding_model,
    qdrant_collection="your_collection",
    global_search="ON/OFF",
    use_env_fallback="True/False",
)

response = client.search_query("Explain the GraphRAG architecture.")
print(response)

client.ingest_data(document_to_insert)

client.close()
```

---

## 🔧 Environment Variables (Optional)

Set these in a `.env` file or shell:

```env
QDRANT_URL=http://localhost:6333
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GLOBAL_SEARCH=ON
```

---

## 🛠️ Development

```bash
uv pip install -e .[dev]
```

---

## 📄 License

MIT License.
