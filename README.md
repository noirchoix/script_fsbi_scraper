# FlavorDB / FSBI Scraper Workbench

This app wraps the existing FSBI scraping functions in a small FastAPI backend and adds a SvelteKit frontend for food-to-flavor exploration.

## What it does

Enter a food or aromatic ingredient, then the app returns:

- parsed food records
- linked molecule IDs
- compound names
- PubChem IDs where available
- flavor/taste/smell descriptors
- scrape trace and warnings
- JSON export

## Run Backend

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Run Frontend

In another terminal:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Environment

The frontend defaults to:

```text
http://127.0.0.1:8000
```

To override it, create `.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```
