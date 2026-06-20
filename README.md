# FlavorDB / FSBI Workbench

This repository is split into a deployable frontend and a separate Python API reference implementation.

## Structure

```text
apps/web   SvelteKit frontend for Netlify/Vercel
apps/api   FastAPI backend/reference FSBI scraper code
```

For the portfolio deployment, deploy only `apps/web` on Netlify and point it to the unified flagship backend on Render.

## Netlify deployment

Use these settings:

```text
Base directory: apps/web
Build command: npm ci && npm run build
Publish directory: build
```

Set this environment variable in Netlify:

```env
VITE_API_BASE_URL=https://flagship-ai-suite-api.onrender.com
```

The frontend calls:

```text
/api/scrape
```

## Local frontend

```bash
cd apps/web
npm install
cp .env.example .env
npm run dev
```

## Local API reference

```bash
cd apps/api
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

## Deployment note

Do not let Netlify install Python dependencies for this project. Netlify should build only the SvelteKit frontend. The Python backend belongs in Render or in the unified flagship backend.
