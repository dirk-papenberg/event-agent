import os
import sys
import threading
import time
import warnings
from urllib.parse import urlsplit, urlunsplit

import httpx
import trafilatura
from ddgs import DDGS
from strands import tool
from tavily import TavilyClient

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HTTP_TIMEOUT = 20.0
_MAX_CONTENT_BYTES = 2_000_000

_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _get_http_client() -> httpx.Client:
    """Geteilter httpx.Client mit Connection-Pooling und deaktivierter Zertifikats-Prüfung.

    SSL-Verify ist bewusst aus: auf diesem Host fehlt ein zuverlässiger CA-Store,
    wodurch sonst die meisten Seiten nicht ladbar sind. Für diesen Discovery-Use-Case
    akzeptieren wir das Risiko — wir parsen Text, laden keine sensiblen Daten hoch.
    """
    global _http_client
    if _http_client is None:
        with _http_client_lock:
            if _http_client is None:
                _http_client = httpx.Client(
                    follow_redirects=True,
                    timeout=_HTTP_TIMEOUT,
                    verify=False,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept-Language": "de-DE,de;q=0.9,en;q=0.7",
                    },
                    limits=httpx.Limits(
                        max_connections=10,
                        max_keepalive_connections=5,
                    ),
                )
    return _http_client

_tavily: TavilyClient | None = None
_tavily_lock = threading.Lock()

_fetched_urls: set[str] = set()
_call_counter = {"search_web": 0, "fetch_page": 0}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def reset_call_counter() -> None:
    _call_counter["search_web"] = 0
    _call_counter["fetch_page"] = 0


def call_counter() -> dict[str, int]:
    return dict(_call_counter)


def _client() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _tavily


def normalize_url(u: str) -> str:
    """Kanonisiert URLs für Verifikations-Matching."""
    try:
        p = urlsplit(u.strip())
        scheme = (p.scheme or "https").lower()
        host = p.netloc.lower()
        path = p.path.rstrip("/") or "/"
        return urlunsplit((scheme, host, path, p.query, ""))
    except Exception:
        return u.strip().lower()


def reset_fetch_log() -> None:
    _fetched_urls.clear()


def fetched_urls() -> set[str]:
    return set(_fetched_urls)


def ddg_search(query: str, max_results: int = 10) -> list[dict]:
    """DuckDuckGo-Suche über ddgs. Kein API-Key, keine Credits.

    Gibt eine Liste mit {title, url, content} zurück. `content` ist das DDG-Snippet (~150-250 Zeichen).
    """
    _call_counter["search_web"] += 1
    n = _call_counter["search_web"]
    _log(f"  → [{n:02d}] ddg_search(query={query!r}, n={max_results})")
    t0 = time.perf_counter()
    try:
        with _tavily_lock:
            with DDGS() as ddgs:
                raw = list(
                    ddgs.text(
                        query,
                        region="de-de",
                        safesearch="off",
                        max_results=max(1, min(max_results, 15)),
                    )
                )
    except Exception as e:
        _log(f"     ← [{n:02d}] Fehler nach {time.perf_counter() - t0:.1f}s: {e!r}")
        return []
    _log(f"     ← [{n:02d}] {len(raw)} Treffer in {time.perf_counter() - t0:.1f}s")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", "") or r.get("url", ""),
            "content": r.get("body", "") or r.get("content", ""),
        }
        for r in raw
    ]


def tavily_search(query: str, max_results: int = 10, depth: str = "advanced") -> list[dict]:
    """Low-level Tavily-Aufruf. Gibt Rohtreffer mit vollem content zurück.

    Wird vom Discoverer-Subagent genutzt; dem Hauptagenten wird diese Funktion NICHT
    als Tool angeboten, damit der rohe Suchkontext nicht im Hauptkontext landet.
    """
    _call_counter["search_web"] += 1
    n = _call_counter["search_web"]
    _log(f"  → [{n:02d}] tavily_search(query={query!r}, n={max_results}, depth={depth})")
    t0 = time.perf_counter()
    try:
        with _tavily_lock:
            resp = _client().search(
                query=query,
                max_results=max(1, min(max_results, 15)),
                search_depth=depth,
            )
    except Exception as e:
        _log(f"     ← [{n:02d}] Fehler nach {time.perf_counter() - t0:.1f}s: {e!r}")
        return []
    results = resp.get("results", [])
    _log(f"     ← [{n:02d}] {len(results)} Treffer in {time.perf_counter() - t0:.1f}s")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
    ]


@tool
def search_web(query: str, max_results: int = 8) -> list[dict]:
    """Durchsucht das Web nach einer Anfrage. Gibt eine Liste mit Titel, URL und Text-Snippet zurück.

    Nutze gezielte Suchanfragen auf Deutsch mit Ort + Zeitraum + Event-Typ,
    z.B. "Flohmarkt Kinder München April 2026" oder "Osterfest Umgebung Stuttgart 2026".
    Variiere die Suchbegriffe — rufe dieses Tool mehrfach mit unterschiedlichen Queries auf.

    Args:
        query: Die Suchanfrage. Möglichst spezifisch mit Ort, Zeitraum und Event-Typ.
        max_results: Maximale Anzahl Ergebnisse (1-10).
    """
    _call_counter["search_web"] += 1
    n = _call_counter["search_web"]
    _log(f"  → [{n:02d}] search_web(query={query!r})")
    t0 = time.perf_counter()
    resp = _client().search(
        query=query,
        max_results=max(1, min(max_results, 10)),
        search_depth="advanced",
    )
    results = resp.get("results", [])
    _log(f"     ← {len(results)} Treffer in {time.perf_counter() - t0:.1f}s")
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
    ]


@tool
def fetch_page(url: str) -> str:
    """Holt den Haupt-Textinhalt einer Webseite. Nutze dies, wenn die Snippets aus der Suche
    nicht reichen und du genaue Daten brauchst (Datum, Uhrzeit, Adresse, Preise, Altersempfehlung).

    Args:
        url: Die URL der abzurufenden Seite.
    """
    _call_counter["fetch_page"] += 1
    n = _call_counter["fetch_page"]
    _log(f"  → [{n:02d}] fetch_page(url={url!r})")
    t0 = time.perf_counter()
    try:
        resp = _get_http_client().get(url)
        resp.raise_for_status()
        html = resp.text
        if len(html.encode("utf-8", errors="ignore")) > _MAX_CONTENT_BYTES:
            html = html[: _MAX_CONTENT_BYTES // 2]
    except Exception as e:
        _log(f"     ← HTTP-Fehler: {e!r}")
        return f"Fehler beim Laden von {url}: {e}"

    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    ) or ""
    if not content:
        _log(f"     ← leer (kein Hauptinhalt extrahierbar) in {time.perf_counter() - t0:.1f}s")
        return f"Leerer Inhalt bei {url}."

    _fetched_urls.add(normalize_url(url))
    _log(f"     ← {len(content)} chars in {time.perf_counter() - t0:.1f}s")
    return content[:8000]
