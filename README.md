# Estate Chat V2 - Simple Container Setup

This is a minimal container skeleton with one `docker-compose.yml`.

Current services:
- `elasticsearch`
- `kibana`
- `neo4j`
- `api` (empty Python container for future FastAPI code)
- `ui` (empty Python container for future Streamlit code)

No `backend/` or `frontend/` app files are created yet.
No `requirements/*.txt` files are used.

## Quick Start

If you are on Linux, set this once for Elasticsearch:

```bash
sudo sysctl -w vm.max_map_count=262144
```

1. Copy env file:

```powershell
Copy-Item .env.example .env
```

2. Start everything:

```powershell
docker compose up -d --build
```

3. Check status:

```powershell
docker compose ps
```

4. Open:
- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`
- Neo4j Browser: `http://localhost:7474`

`api` and `ui` containers are intentionally idle (`sleep infinity`) until you add app code.

## Kibana: How To Use It

1. Open `http://localhost:5601`.
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
