from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from functions import (
    _extract_compound_links_from_food_html,
    _fsbi_fetch,
    _parse_common_name,
    _parse_flavor_profile,
    _parse_pubchem_id,
    fsbi_discover_compound_and_food_links,
    fsbi_download_compound_json,
    fsbi_download_food_json,
)


app = FastAPI(title="FlavorDB / FSBI Scraper API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScrapeRequest(BaseModel):
    food: str = Field(min_length=2, max_length=120)
    max_foods: int = Field(default=3, ge=1, le=8)
    max_compounds: int = Field(default=25, ge=1, le=80)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _load_local_food_hint(query: str) -> dict[str, str] | None:
    path = Path("data/fragrance_food_sources.csv")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    q = query.strip().lower()
    for _, row in df.iterrows():
        common = str(row.get("common_name", "")).strip()
        if common.lower() == q:
            return {
                "common_name": common,
                "latin_name": str(row.get("latin_name", "")).strip(),
                "category": str(row.get("category", "")).strip(),
            }
    return None


def _compound_record(url: str) -> dict[str, Any] | None:
    data = fsbi_download_compound_json(url)
    if not data:
        return None
    pubchem_id = _parse_pubchem_id(data)
    name = _parse_common_name(data)
    flavor_profile = sorted(_parse_flavor_profile(data))
    return {
        "pubchem_id": pubchem_id,
        "common_name": name,
        "flavor_profile": flavor_profile,
        "source_url": url,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/scrape")
def scrape_food(payload: ScrapeRequest) -> dict[str, Any]:
    query = payload.food.strip()
    trace: list[str] = []
    warnings: list[str] = []

    trace.append(f"Searching FSBI for food term: {query}")
    compound_urls, food_urls = fsbi_discover_compound_and_food_links([query], max_per_query=payload.max_compounds)
    trace.append(f"Discovered {len(food_urls)} food links and {len(compound_urls)} compound links.")

    foods: list[dict[str, Any]] = []
    compound_pool: dict[int | str, dict[str, Any]] = {}

    for food_url in food_urls[: payload.max_foods]:
        food_data = fsbi_download_food_json(food_url, max_compounds=payload.max_compounds)
        if not food_data:
            warnings.append(f"Could not parse food page: {food_url}")
            continue

        try:
            html = _fsbi_fetch(food_url)
            linked_compounds = _extract_compound_links_from_food_html(html)[: payload.max_compounds]
        except Exception:
            linked_compounds = []

        for compound_url in linked_compounds:
            record = _compound_record(compound_url)
            if not record:
                continue
            key = record["pubchem_id"] if record["pubchem_id"] is not None else record["source_url"]
            compound_pool[key] = record

        foods.append(
            {
                "entity_id": _safe_int(food_data.get("entity_id")),
                "alias": food_data.get("alias") or query.lower(),
                "category": food_data.get("category") or "",
                "scientific_name": food_data.get("scientific_name") or "",
                "molecule_ids": sorted({_safe_int(x) for x in food_data.get("molecules", []) if _safe_int(x) is not None}),
                "source_url": food_url,
            }
        )

    for compound_url in compound_urls[: payload.max_compounds]:
        record = _compound_record(compound_url)
        if not record:
            continue
        key = record["pubchem_id"] if record["pubchem_id"] is not None else record["source_url"]
        compound_pool[key] = record

    molecules = sorted(
        compound_pool.values(),
        key=lambda item: (len(item.get("flavor_profile") or []), item.get("common_name") or ""),
        reverse=True,
    )

    flavor_counter: Counter[str] = Counter()
    for molecule in molecules:
        flavor_counter.update(molecule.get("flavor_profile") or [])

    local_hint = _load_local_food_hint(query)
    if not foods and local_hint:
        foods.append(
            {
                "entity_id": None,
                "alias": local_hint["common_name"].lower(),
                "category": local_hint["category"],
                "scientific_name": local_hint["latin_name"],
                "molecule_ids": [],
                "source_url": "local fragrance_food_sources.csv",
            }
        )
        warnings.append("FSBI did not return a parsed food page; local source hint was used.")

    return {
        "query": query,
        "status": "ok" if foods or molecules else "no_results",
        "summary": {
            "food_matches": len(foods),
            "molecule_matches": len(molecules),
            "unique_flavor_terms": len(flavor_counter),
        },
        "flavor_profile": [
            {"term": term, "count": count}
            for term, count in flavor_counter.most_common()
        ],
        "foods": foods,
        "molecules": molecules,
        "trace": trace,
        "warnings": warnings,
    }
