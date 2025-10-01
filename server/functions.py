from __future__ import annotations
import time as _time
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Set
import json
import time
import urllib.request
from urllib.parse import quote as _q
import urllib.error
import pandas as pd
import re, html as _html
from html.parser import HTMLParser

# --------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------
VERBOSE_ERRORS = True      # noisy network errors for troubleshooting
FSBI_DEBUG = False         # set False: no verbose prints for FSBI discovery/parsing

# --------------------------------------------------------------------
# === tiny copies of helpers so you can run this standalone ===
_tag_re = re.compile(r"<[^>]+>")
_space_re = re.compile(r"\s+")
def _textify(html: str) -> str:
    t = _tag_re.sub(" ", html)
    t = _space_re.sub(" ", t)
    return _html.unescape(t)

def _section_by_id(htm: str, sec_id: str, until_ids: list[str]) -> str:
    patt_start = rf'id=["\']{re.escape(sec_id)}["\'][^>]*>'
    patt_end = r'|'.join([rf'id=["\']{re.escape(x)}["\']' for x in until_ids]) or r'$\B'
    m = re.search(rf'({patt_start}).*?(?={patt_end})', htm, flags=re.I | re.S)
    return m.group(0) if m else ""

def _first_row_value_from_table(section_html: str, label_regex: str) -> str:
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", section_html, flags=re.I | re.S):
        cols = re.split(r"</t[hd]>", row, flags=re.I)
        if len(cols) < 2:
            continue
        label = _textify(cols[0]).strip()
        if re.search(label_regex, label, flags=re.I):
            return _textify(cols[1]).strip()
    return ""

# --------------------------------------------------------------------
# FlavorDB (legacy JSON endpoint)
# --------------------------------------------------------------------
def flavordb_entity_url(x: int) -> str:
    return f"https://cosylab.iiitd.edu.in/flavordb/entities_json?id={x}"

def _urlopen_with_retries(url: str, *, tries: int = 7, backoff: float = 2.0, timeout: int = 60) -> Any:
    import random
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    }

    attempt = 0
    delay = backoff
    while True:
        attempt += 1
        try:
            req = Request(url, headers=headers)
            return urlopen(req, timeout=timeout)
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 403) and attempt < tries:
                time.sleep(delay + random.uniform(0, delay * 0.25))
                delay *= 2.0
                continue
            raise
        except URLError:
            if attempt < tries:
                time.sleep(delay + random.uniform(0, delay * 0.25))
                delay *= 2.0
                continue
            raise

def get_flavordb_entity(x: int) -> Dict[str, Any]:
    with _urlopen_with_retries(flavordb_entity_url(x)) as url:
        return json.loads(url.read().decode("utf-8"))

# --------------------------------------------------------------------
# Shared CSV schemas
# --------------------------------------------------------------------
def flavordb_entity_cols() -> List[str]:
    # kept for backwards compatibility with any legacy callers
    return [
        "entity_id", "entity_alias_readable", "entity_alias_synonyms",
        "natural_source_name", "category_readable", "molecules"
    ]

def flavordb_df_cols() -> List[str]:
    return ["entity id", "alias", "synonyms", "scientific name", "category", "molecules"]

def molecules_df_cols() -> List[str]:
    return ["pubchem id", "common name", "flavor profile"]

# --------------------------------------------------------------------
# Cleaning & normalization
# --------------------------------------------------------------------
def clean_flavordb_dataframes(flavor_df: pd.DataFrame, molecules_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    strtype = str
    settype = set

    for k in ["alias", "scientific name", "category"]:
        if k in flavor_df.columns:
            flavor_df[k] = [v.strip().lower() if isinstance(v, strtype) else "" for v in flavor_df[k]]

    def _to_synset(v: Any) -> Set[str]:
        if isinstance(v, settype): return v
        if isinstance(v, strtype) and v:
            if v.startswith("{") and v.endswith("}"):
                try:
                    return set(eval(v))
                except Exception:
                    pass
            return {x.strip().lower() for x in re.split(r"[,;]\s*", v) if x.strip()}
        return set()

    if "synonyms" in flavor_df.columns:
        flavor_df["synonyms"] = [ _to_synset(x) for x in flavor_df["synonyms"] ]

    if "flavor profile" in molecules_df.columns:
        def _to_flavor_set(v: Any) -> Set[str]:
            if isinstance(v, (list, set)): return {str(x).strip().lower() for x in v if str(x).strip()}
            if isinstance(v, str) and v.startswith("{"):
                try: return set(eval(v))
                except Exception: return set()
            return set()
        molecules_df["flavor profile"] = [ _to_flavor_set(x) for x in molecules_df["flavor profile"] ]

    return (
        flavor_df.groupby("entity id").first().reset_index(),
        molecules_df.groupby("pubchem id").first().reset_index(),
    )

# --------------------------------------------------------------------
# FlavorDB batch (legacy)
# --------------------------------------------------------------------
def get_flavordb_dataframes(start: int, end: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    flavordb_data: List[List[Any]] = []
    molecules_dict: Dict[int, List[Any]] = {}
    missing: List[int] = []

    raw_cols = flavordb_entity_cols()

    for i in range(start, end):
        entity_id = i + 1
        try:
            fdbe = get_flavordb_entity(entity_id)
            series = [fdbe[k] for k in raw_cols[:-1]]
            series.append({m["pubchem_id"] for m in fdbe["molecules"]})
            flavordb_data.append(series)
            for m in fdbe["molecules"]:
                pid = m["pubchem_id"]
                if pid not in molecules_dict:
                    molecules_dict[pid] = [
                        m.get("common_name"),
                        set((m.get("flavor_profile") or "").split("@")) if m.get("flavor_profile") else set(),
                    ]
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            if VERBOSE_ERRORS:
                if isinstance(e, urllib.error.HTTPError):
                    try: body = e.read().decode("utf-8", "replace")
                    except Exception: body = "<no body>"
                    print(f"[FlavorDB HTTPError] id={entity_id} code={getattr(e,'code',None)} reason={getattr(e,'reason',None)}")
                    try: hdrs = dict(e.headers or {})
                    except Exception: hdrs = {}
                    print(f"Headers: {hdrs}\nBody (first 600):\n{body[:600]}\n" + "-"*60)
                else:
                    print(f"[FlavorDB URLError] id={entity_id} reason={getattr(e,'reason',e)}")
            missing.append(entity_id)
            continue

    flavordb_df = pd.DataFrame(flavordb_data, columns=flavordb_df_cols())
    molecules_df = pd.DataFrame(
        [[k, v[0], v[1]] for k, v in molecules_dict.items()],
        columns=molecules_df_cols(),
    )
    flavordb_df, molecules_df = clean_flavordb_dataframes(flavordb_df, molecules_df)
    return flavordb_df, molecules_df, missing

def update_flavordb_dataframes(df0: pd.DataFrame, df1: pd.DataFrame, ranges: Sequence[Tuple[int, int]],
                               out_flavordb_csv: str = "flavordb.csv", out_molecules_csv: str = "molecules.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    df0_old = df0.copy(); df1_old = df1.copy(); missing_old: List[int] = []
    start_ts = time.time()
    try:
        for a, b in ranges:
            df0_new, df1_new, missing_new = get_flavordb_dataframes(a, b)
            df0_old = pd.concat([df0_old, df0_new], ignore_index=True, sort=False)
            df1_old = pd.concat([df1_old, df1_new], ignore_index=True, sort=False)
            missing_old.extend(missing_new)
        return df0_old, df1_old, missing_old
    finally:
        df0_old.to_csv(out_flavordb_csv, index=False)
        df1_old.to_csv(out_molecules_csv, index=False)
        print("Downloading took:", (time.time() - start_ts)/60.0, "minutes")

# --------------------------------------------------------------------
# CSV utilities
# --------------------------------------------------------------------
def missing_entity_ids(flavor_df: pd.DataFrame) -> List[int]:
    if flavor_df.empty: return []
    entity_id_set = set(flavor_df["entity id"])
    return [i for i in range(1, 1 + max(entity_id_set)) if i not in entity_id_set]

def load_db(flavordb_csv: str = "flavordb.csv", molecules_csv: str = "molecules.csv") -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    settype = set
    df0 = pd.read_csv(flavordb_csv)[flavordb_df_cols()]
    df0["synonyms"] = [
        eval(x) if isinstance(x, str) and x.startswith("{") else (x if isinstance(x, settype) else set())
        for x in df0["synonyms"]
    ]
    df0["molecules"] = [
        eval(x) if isinstance(x, str) and x.startswith("{") else (x if isinstance(x, settype) else set())
        for x in df0["molecules"]
    ]
    df1 = pd.read_csv(molecules_csv)[molecules_df_cols()]
    df1["flavor profile"] = [eval(x) if isinstance(x, str) and x.startswith("{") else set()
                             for x in df1["flavor profile"]]
    df0, df1 = clean_flavordb_dataframes(df0, df1)
    return df0, df1, missing_entity_ids(df0)

# --------------------------------------------------------------------
# FSBI-DB (HTML + optional JSON download buttons)
# --------------------------------------------------------------------
_FSBI_ROOT = "https://fsbi-db.de"

class _LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._in_a = False
        self._href = None
        self._chunks = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_a = True
            self._href = dict(attrs).get("href")
            self._chunks = []
    def handle_data(self, data):
        if self._in_a:
            self._chunks.append(data)
    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = "".join(self._chunks).strip()
            if self._href:
                self.links.append((self._href, text))
            self._in_a = False; self._href=None; self._chunks=[]

def _abs(u: Optional[str]) -> str:
    if not u: return ""
    if u.startswith("http"): return u
    if u.startswith("/"): return _FSBI_ROOT + u
    return f"{_FSBI_ROOT}/{u}"

def _fsbi_fetch(url: str) -> str:
    # UPDATED: Now uses the robust _urlopen_with_retries function defined above
    try:
        with _urlopen_with_retries(url) as resp:
            raw = resp.read()
            try:
                return raw.decode("utf-8", errors="replace")
            except Exception:
                return raw.decode("iso-8859-1", errors="replace")
    except urllib.error.HTTPError as e:
        if VERBOSE_ERRORS:
            try: body = e.read().decode("utf-8", "replace")
            except Exception: body = "<no body>"
            print(f"[FSBI HTTPError] url={url} code={getattr(e,'code',None)} reason={getattr(e,'reason',None)}")
            try: hdrs = dict(e.headers or {})
            except Exception: hdrs = {}
            print(f"Headers: {hdrs}\nBody (first 600):\n{body[:600]}\n" + "-"*60)
        raise
    except urllib.error.URLError as e:
        if VERBOSE_ERRORS:
            print(f"[FSBI URLError] url={url} reason={getattr(e,'reason',e)}")
        # CRITICAL: Re-raise the URLError so the caller's catch block (e.g., in fsbi_download_food_json)
        # can log the error message you saw: "[FSBI] food error: IncompleteRead(0 bytes read)"
        raise

# ---- JSON download link finder (restored) -----------------------------------
def _find_json_download_link(html: str) -> str | None:
    p = _LinkCollector()
    p.feed(html or "")
    # Priority 1: anchor text mentions JSON
    for href, text in p.links:
        if "json" in (text or "").lower():
            return _abs(href or "")
    # Priority 2: href hints (json/download) but exclude file assets
    for href, _text in p.links:
        h = (href or "").lower()
        if any(x in h for x in ["json", "download"]) and not h.endswith((".png", ".jpg", ".jpeg", ".mol2", ".sdf")):
            return _abs(href or "")
    # Optional debug
    if FSBI_DEBUG:
        print("[FSBI] no JSON link found; sample anchors:")
        for href, text in p.links[:15]:
            print("  href=", href, "| text=", (text or "")[:60])
    return None

# --- FSBI Compound extraction -------------------------------------------------
def _extract_compound_name_from_html(htm: str) -> str:
    sec = _section_by_id(htm, "general", ["classification", "Quality", "Toxicity", "receptor"])
    m = re.search(r'<h3>\s*<i>\s*([^<]+?)\s*</i>\s*</h3>', sec or "", flags=re.I | re.S)
    if m:
        return m.group(1).strip().lower()
    
    name = _first_row_value_from_table(sec or htm, r"\bname\b").strip().lower()
    if name: return name
    
    m = re.search(r">\s*###\s*([^<\n]+?)\s*<", htm or "", flags=re.I)
    if m: return m.group(1).strip().lower()
    
    m = re.search(r"<title>\s*([^<]+?)(?:\s*[-|]\s*FSBI-DB)?\s*</title>", htm or "", flags=re.I)
    return m.group(1).strip().lower() if m else ""

def _extract_pubchem_from_compound_html(htm: str) -> int | None:
    # 0) FSBI embeds the CID in a JS var used to load the SDF:
    #    var jsvar = "6054";
    m = re.search(r'var\s+jsvar\s*=\s*["\'](\d{1,9})["\']', htm or "", flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: pass

    # 1) explicit PubChem link
    m = re.search(r'href=["\'](?:https?:)?//pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d{1,9})["\']',
                  htm or "", flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: pass

    # 2) FSBI network link carries the CID as id=
    m = re.search(r'href=["\'][^"\']*cytoscape(?:_food)?\.php\?id=(\d{1,9})["\']', htm or "", flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: pass

    # 3) FSBI search link sometimes carries the CID too
    m = re.search(r'href=["\'][^"\']*search\.php\?term=(\d{1,9})["\']', htm or "", flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: pass

    # 4) Generals table
    sec = _section_by_id(htm or "", "general", ["classification", "Quality", "Toxicity", "receptor"])
    v = _first_row_value_from_table(sec or htm, r"\b(pubchem\s*id|pubchemid|cid)\b")
    if v:
        m = re.search(r"\b(\d{1,9})\b", v)
        if m:
            try: return int(m.group(1))
            except Exception: pass

    # 5) plain text fallback
    txt = _textify(htm or "")
    m = re.search(r"\bCID\s*[:#]?\s*(\d{1,9})\b", txt, flags=re.I)
    if m:
        try: return int(m.group(1))
        except Exception: pass
    return None

def _extract_flavor_profile_from_compound_html(htm: str) -> set[str]:
    sec = _section_by_id(htm, "Quality", ["Toxicity", "receptor", "classification", "general"])
    out: set[str] = set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", sec or "", flags=re.I | re.S):
        cols = re.split(r"</t[hd]>", row, flags=re.I)
        if len(cols) < 2: continue
        category = _textify(cols[0]).strip().lower()
        if category not in {"taste", "smell"}: continue
        quality = _textify(cols[1]).strip().lower()
        for token in re.split(r"[,\;/]\s*", quality):
            tok = token.strip()
            if tok and tok not in {"na", "n/a", "none"} and len(tok) <= 40:
                out.add(tok)
    return out

def _extract_compound_fields_from_compound_html(htm: str) -> dict:
    """
    Robustly parse FSBI compound #general section (Bootstrap rows) into fields.
    """
    out = {
        "name": "",
        "flavordb_id": None,
        "pubchem_id": None,
        "molecular_weight": None,
        "molecular_formula": "",
        "smiles": "",
        "inchikey": "",
        "synonyms": [],
    }

    # 1) Isolate #general section
    m = re.search(r'<section id="general">(.*?)</section>', htm, flags=re.I | re.S)
    general = m.group(1) if m else ""

    # 2) Name (title in the section)
    m = re.search(r'<h3>\s*<i>\s*([^<]+?)\s*</i>\s*</h3>', general, flags=re.I | re.S)
    if m:
        out["name"] = m.group(1).strip()

    # 3) Extract scalar fields directly using precise patterns
    def _extract_from_row(label_pattern, value_pattern):
        m = re.search(
            rf'<h5[^>]*>\s*({label_pattern})\s*</h5>\s*</div>\s*'
            rf'<div\s+class="col-md-9"[^>]*>\s*<h5[^>]*>\s*<small>\s*({value_pattern})\s*</small>',
            general, flags=re.I | re.S
        )
        if m: return _textify(m.group(2)).strip()
        return None

    out["molecular_weight"] = float(_extract_from_row(r"Molecular\s*Weight", r"[\d\.]+") or 0.0)
    out["molecular_formula"] = _extract_from_row(r"Molecular\s*Formula", r"[A-Za-z0-9]+")
    out["smiles"] = _extract_from_row(r"OpenEye\s*CAN\s*SMILES", r"[^<]+")
    out["inchikey"] = _extract_from_row(r"(?:IUPAC\s*InChI\s*Key|InChIKey)", r"[A-Z0-9\-]+")
    
    # 4) Extract synonyms
    syn_html = ""
    ms = re.search(
        r'>\s*Synonyms\s*</h5>\s*</div>\s*<div\s+class="col-md-9"[^>]*>(.*?)</div>\s*</div>',
        general, flags=re.I | re.S
    )
    if ms:
        syn_html = ms.group(1)
    if syn_html:
        text = _textify(syn_html)
        parts = [p.strip() for p in re.split(r'[;,]', text)]
        seen = set()
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out["synonyms"].append(p)
    
    # 5) As a final safety, also try the global helpers for pubchem/name if still empty
    out["pubchem_id"] = out["pubchem_id"] or _extract_pubchem_from_compound_html(htm)
    out["name"] = out["name"] or _extract_compound_name_from_html(htm)
    
    return out

# --- FSBI Food extraction -----------------------------------------------------
def _extract_food_fields_from_html(htm: str) -> tuple[int | None, str, str]:
    """
    Food page parsing (use the Generals card only):
      - alias  := Generals -> 'Name'
      - entity := Generals -> 'FlavorDB Food ID'
      - category := Generals -> 'Food Category'
    """
    gen = _section_by_id(htm, "general", ["class", "kfo", "ref"])
    
    # alias
    alias = _first_row_value_from_table(gen, r"\bname\b").strip().lower()

    # entity id
    v_id = _first_row_value_from_table(gen, r"flavordb\s+food\s+id")
    try:
        m = re.search(r"\d+", v_id) if v_id else None
        entity_id = int(m.group()) if m else None
    except Exception:
        entity_id = None

    # category
    category = _first_row_value_from_table(gen, r"food\s+category").strip().lower()

    return (entity_id, alias, category)

def _extract_compound_links_from_food_html(htm: str) -> list[str]:
    # Prefer KFO card, fall back to whole page; return absolute links
    sec = _section_by_id(htm, "kfo", ["ref", "class", "general"]) or htm
    p = _LinkCollector(); p.feed(sec or "")
    hrefs = []
    for (h, _t) in p.links:
        h = h or ""
        # accept relative, absolute, or dotted prefixes; just look for the path fragment
        if "single.php?id=" in h:
            hrefs.append(_abs(h))
    # de-dupe, preserve order
    out, seen = [], set()
    for h in hrefs:
        if h not in seen:
            seen.add(h); out.append(h)
    return out

def fsbi_download_compound_json(compound_page_url: str) -> dict | None:
    try:
        html = _fsbi_fetch(compound_page_url)
        if FSBI_DEBUG:
            print(f"[FSBI] compound page ok: {compound_page_url} ({len(html)} bytes)")

        # 1) try JSON button (when visible)
        json_link = _find_json_download_link(html)
        if json_link:
            try:
                txt = _fsbi_fetch(json_link)
                data = json.loads(txt)
                # Ensure our standard keys exist even if JSON schema varies
                data.setdefault("pubchem_id", _extract_pubchem_from_compound_html(html))
                data.setdefault("common_name", _extract_compound_name_from_html(html))
                # add normalized flavor profile from Quality table if JSON lacks it
                if not _parse_flavor_profile(data):
                    data["flavor_profile"] = sorted(_extract_flavor_profile_from_compound_html(html))
                return data
            except Exception as e:
                if FSBI_DEBUG:
                    print(f"[FSBI] compound JSON fetch/parse failed: {json_link} ({e})")

        # 2) fallback: parse HTML comprehensively
        fields = _extract_compound_fields_from_compound_html(html)  # name, ids, MW, formula, smiles, inchikey, synonyms
        # Always add/refresh these minimal fields
        
        # ✅ Ensure pubchem_id is extracted and assigned explicitly if it wasn't found in fields
        pubchem_id = fields.get("pubchem_id") or _extract_pubchem_from_compound_html(html)
        if pubchem_id:
            fields["pubchem_id"] = pubchem_id
        fields["common_name"] = fields.get("name") or _extract_compound_name_from_html(html)
        fields["flavor_profile"] = sorted(_extract_flavor_profile_from_compound_html(html))
        return fields

    except Exception as e:
        if VERBOSE_ERRORS:
            print(f"[FSBI] compound error: {e}")
        return None

def fsbi_download_food_json(food_page_url: str, max_compounds: int = 50) -> dict | None:
    try:
        html = _fsbi_fetch(food_page_url)
        if FSBI_DEBUG:
            print(f"[FSBI] food page ok: {food_page_url} ({len(html)} bytes)")

        # JSON button attempt (if ever exposed)
        json_link = _find_json_download_link(html)
        if json_link:
            try:
                txt = _fsbi_fetch(json_link)
                return json.loads(txt)
            except Exception as e:
                if FSBI_DEBUG:
                    print(f"[FSBI] food JSON fetch/parse failed: {json_link} ({e})")

        # fallback: parse HTML and linked compounds
        entity_id, alias, category = _extract_food_fields_from_html(html)

        molecules: list[int] = []
        for u in _extract_compound_links_from_food_html(html)[:max_compounds]:
            cj = fsbi_download_compound_json(u)
            cid = cj.get("pubchem_id") if cj else None
            if isinstance(cid, int):
                molecules.append(cid)

        return {
            "entity_id": entity_id,
            "alias": alias,
            "synonyms": [],
            "scientific_name": "",
            "category": category,
            "molecules": sorted(set(molecules)),
        }
    except Exception as e:
        if VERBOSE_ERRORS:
            print(f"[FSBI] food error: {e}")
        return None

# --------------------------------------------------------------------
# FSBI discovery
# --------------------------------------------------------------------
def fsbi_discover_compound_and_food_links(queries: list[str] | None = None, max_per_query: int = 50):
    """Find item pages by hitting the type-specific search pages."""
    if not queries:
        queries = ["beer","tea","citrus","onion","garlic","cocoa","lemon","mango","apple","milk"]

    compounds, foods = set(), set()
    for q in queries:
        for typ, pat, bucket in [
            ("compounds", "single.php?id=", compounds),
            ("foods",     "food.php?id=",   foods),
        ]:
            url = f"{_FSBI_ROOT}/search.php?term={_q(q)}&type={typ}"
            try:
                html = _fsbi_fetch(url)
                if FSBI_DEBUG:
                    print(f"[FSBI] discover '{q}' typ={typ} ok: {url} -> {len(html)} bytes")
                p = _LinkCollector(); p.feed(html)
                hits = 0
                for href, _text in p.links:
                    href = href or ""
                    if pat in href:
                        bucket.add(_abs(href)); hits += 1
                if FSBI_DEBUG:
                    print(f"[FSBI]   found {hits} links matching '{pat}'")
            except Exception as e:
                if FSBI_DEBUG:
                    print(f"[FSBI] discover FAILED: {url} ({e})")
                continue

    # Trim to a sane size
    lim = len(queries) * max_per_query if queries else 500
    return sorted(list(compounds))[:lim], sorted(list(foods))[:lim]

# --------------------------------------------------------------------
# Build dataframes from FSBI pages
# --------------------------------------------------------------------
def _norm_str(x): return (x or "").strip().lower() if isinstance(x, str) else ""

def _norm_set_strs(iterable):
    out = set()
    if not iterable: return out
    for v in iterable:
        s = (v if isinstance(v, str) else str(v)).strip().lower()
        if s: out.add(s)
    return out

def _parse_pubchem_id(d: dict) -> int | None:
    for k in ["pubchem_id","pubchem","PubChem","pubChemId","pubchemId","pubchemID"]:
        if k in d:
            try: return int(str(d[k]).strip())
            except Exception: pass
    return None

def _parse_common_name(d: dict) -> str:
    for k in ["common_name","commonName","name","Name","preferred_name","title"]:
        v = d.get(k)
        if isinstance(v, str): return v
    return ""

def _parse_flavor_profile(d: dict) -> set[str]:
    out = set()
    keys = ["odor","odour","odor_qualities","odour_qualities","taste","taste_qualities","flavor_profile","flavour_profile"]
    for key in keys:
        val = d.get(key)
        if isinstance(val, list): out |= _norm_set_strs(val)
        elif isinstance(val, str):
            parts = re.split(r"[@,;/]\s*", val)
            out |= _norm_set_strs(parts)
        elif isinstance(val, dict):
            out |= _norm_set_strs(val.values())
    fp = d.get("flavor_profile") or d.get("flavour_profile")
    if isinstance(fp, dict):
        for v in fp.values():
            if isinstance(v, list):
                out |= _norm_set_strs(v)
    return out

def build_fsbi_dataframes(compound_urls: list[str] | None = None,
                          food_urls: list[str] | None = None,
                          discovery_queries: list[str] | None = None,
                          max_compounds: int | None = 1000,
                          delay: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    

    # Discover if not provided
    if not compound_urls:
        compound_urls, discovered_foods = fsbi_discover_compound_and_food_links(queries=discovery_queries)
        if not food_urls:
            food_urls = discovered_foods

    foods_map: dict[int, dict] = {}
    mols_map: dict[int, dict] = {}

    # Pass 1: compounds → (molecules, flavors)
    for idx, curl in enumerate(compound_urls):
        if max_compounds and idx >= max_compounds: break
        d = fsbi_download_compound_json(curl)
        if not d:
            if delay: _time.sleep(delay)
            continue
        pubchem = _parse_pubchem_id(d)
        cname = _parse_common_name(d)
        fprof = _parse_flavor_profile(d)
        if pubchem is not None:
            rec = mols_map.setdefault(pubchem, {"common name": cname, "flavor profile": set()})
            if cname and not rec.get("common name"):
                rec["common name"] = cname
            rec["flavor profile"] |= fprof
        if delay: _time.sleep(delay)

    # Pass 2: foods → parse alias/category + crawl linked compound pages
    for fidx, furl in enumerate(food_urls or []):
        d = fsbi_download_food_json(furl)
        if not d:
            if delay: _time.sleep(delay)
            continue

        # robust id extraction
        fid = d.get("entity_id") or d.get("id") or d.get("food_id") or d.get("fsbi_id")
        if fid is None:
            m = re.search(r"[?&]id=(\d+)", furl)
            if m:
                try: fid = int(m.group(1))
                except Exception: fid = None
        try: fid = int(str(fid)) if fid is not None else None
        except Exception: fid = None
        if fid is None:
            if delay: _time.sleep(delay)
            continue

        rec = foods_map.setdefault(fid, {
            "alias": _norm_str(d.get("alias") or d.get("name") or d.get("food_name") or d.get("label")),
            "synonyms": set(),
            "scientific name": _norm_str(d.get("scientific_name") or d.get("scientific") or d.get("latin_name")),
            "category": _norm_str(d.get("category") or d.get("group") or d.get("type")),
            "molecules": set(),
        })

        for pid in d.get("molecules") or []:
            try:
                ip = int(pid)
                rec["molecules"].add(ip)
            except Exception:
                pass

        if delay: _time.sleep(delay)

    # Materialize rows
    flavor_rows = [
        [int(fid), rec.get("alias",""), rec.get("synonyms", set()),
         rec.get("scientific name",""), rec.get("category",""),
         rec.get("molecules", set())]
        for fid, rec in foods_map.items()
    ]
    mol_rows = [
        [int(pid), rec.get("common name",""), rec.get("flavor profile", set())]
        for pid, rec in mols_map.items()
    ]

    flavordb_df = pd.DataFrame(flavor_rows, columns=flavordb_df_cols()) if flavor_rows else pd.DataFrame(columns=flavordb_df_cols())
    molecules_df = pd.DataFrame(mol_rows, columns=molecules_df_cols()) if mol_rows else pd.DataFrame(columns=molecules_df_cols())
    flavordb_df, molecules_df = clean_flavordb_dataframes(flavordb_df, molecules_df)
    return flavordb_df, molecules_df

# --------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------
def diagnose_flavordb(ids=(1,)):
    """Fetch a few FlavorDB JSONs and print detailed errors."""
    for x in ids:
        url = flavordb_entity_url(x)
        print(f"\n== Checking {url} ==")
        try:
            d = get_flavordb_entity(x)
            print(f"OK: entity {x} -> keys: {list(d)[:8]}")
        except urllib.error.HTTPError as e:
            try: body = e.read().decode("utf-8", "replace")
            except Exception: body = "<no body>"
            print(f"[HTTPError] code={getattr(e,'code',None)} reason={getattr(e,'reason',None)}")
            try: hdrs = dict(e.headers or {})
            except Exception: hdrs = {}
            print(f"Headers: {hdrs}\nBody (first 600):\n{body[:600]}")
        except urllib.error.URLError as e:
            print(f"[URLError] reason={getattr(e,'reason',e)}")
        except Exception as e:
            print(f"[OtherError]")