# Estate Chat V2 - Simplified Container Setup

This is a minimal container skeleton with `docker-compose.yml` featuring ChromaDB for vector storage.

Current services:
- `chromadb` - Vector database for document embeddings
- `neo4j` - Knowledge graph database
- `api` - FastAPI backend
- `ui` - Streamlit frontend

## Quick Start

1. Copy env file (if not already present):

```bash
cp .env.example .env
```

2. Start everything:

```bash
docker compose up -d --build
```

3. Check status:

```bash
docker compose ps
```

4. Access services:
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI API**: `http://localhost:8001`
- **Neo4j Browser**: `http://localhost:7474`
- **ChromaDB**: `http://localhost:8000`

## Environment Variables

Set these in your `.env` file:
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (default: `neo4jpassword`)
- `NEO4J_URL` (default: `bolt://localhost:7687`)
- `NEO4J_URI` (optional fallback if `NEO4J_URL` is not set; use `bolt://neo4j:7687` inside Docker Compose)
- `NEO4J_DATABASE` (default: `neo4j`)
- `OPENAI_API_KEY` (required for LLM functionality)
- `OPENAI_GRAPH_MODEL` (optional, defaults to `OPENAI_CHAT_MODEL` or `gpt-4o-mini`)
- `OPENAI_GRAPH_EMBEDDING_MODEL` (optional, defaults to `text-embedding-3-small`)

## Property Graph Build

The first graph stage builds a LlamaIndex `PropertyGraphIndex` from the existing extracted JSON files in `backend/data/extracted`.

What the build does:
- loads the shared `ExtractedDocument` model so graph and vector ingestion use the same source data
- creates one LlamaIndex `Document` per extracted PDF page, preserving `document_id`, `document_type`, `filename`, and `page_number`
- uses a schema-guided estate-document extractor for people, organizations, properties, documents, dates, money amounts, roles, and legal events
- writes the graph to Neo4j by default through `Neo4jPropertyGraphStore`

Run the basic graph build from the project root:

```bash
python backend/data/preprocessing/run_property_graph.py
```

Before running it, Neo4j must be reachable on the configured Bolt URL. From the host machine that is usually `bolt://localhost:7687`; from inside Docker Compose it is usually `bolt://neo4j:7687`.

Or from the preprocessing folder:

```bash
python run_property_graph.py
```

This stage only creates the graph. The later prompt-composition stage can retrieve graph paths and normalize them into text alongside document chunks.

## Architecture

- **API** runs on port 8000 (internally) → mapped to **8001** (host)
- **UI** (Streamlit) runs on port 8501
- **ChromaDB** runs on port 8000 (internally) → mapped to **8000** (host)
- Services communicate internally via Docker network

## Dependencies

All Python dependencies are defined in `pyproject.toml` and installed via Poetry in the containers.

Key packages:
- FastAPI + Uvicorn (API)
- Streamlit (UI)
- ChromaDB (vector search)
- Neo4j (knowledge graphs)
- LangChain (chunking)
- Sentence Transformers (embeddings)
2. Go to **Dev Tools**.
3. Run a quick Elasticsearch check:

```http
GET /
GET _cluster/health
GET _cat/indices?v
```

4. Create a sample index and document:

```http
PUT demo-estate
POST demo-estate/_doc
{
  "title": "Will and Testament",
  "client": "Jane Doe"
}
GET demo-estate/_search
```

5. Optional UI view:
- Go to **Stack Management -> Data Views**.
- Create data view `demo-estate*`.
- Use **Discover** to browse documents.

## How You Will Run FastAPI Later

When you add your FastAPI project files, run inside the `api` container:

```powershell
docker compose exec api pip install fastapi uvicorn
docker compose exec api uvicorn your_module:app --host 0.0.0.0 --port 8000
```

## How You Will Run Streamlit Later

When you add your Streamlit file, run inside the `ui` container:

```powershell
docker compose exec ui pip install streamlit
docker compose exec ui streamlit run path/to/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```
