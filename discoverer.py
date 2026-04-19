import sys

from strands import tool

from tools import ddg_search


_MAX_SNIPPET_CHARS = 300


@tool
def discover_candidates(query: str, zeitraum: str, max_results: int = 12) -> list[dict]:
    """Führt eine Websuche durch und gibt eine kompakte Kandidatenliste zurück.

    Intern wird Tavily mit search_depth="advanced" aufgerufen. Die Ergebnisse werden
    auf URL, Titel und ein kurzes Snippet reduziert, damit der Hauptkontext schlank
    bleibt. Volltext wird erst über `evaluate_candidate` nachgeladen.

    Nutze dies als Standard-Suchwerkzeug. Variiere Queries breit:
    Discovery-Queries, Domain-Queries, Venue-Queries, Kategorie-Queries, saisonale Queries.
    Der Zeitraum wird für Telemetrie mitgegeben, nicht für Filterung.

    Args:
        query: Suchanfrage, möglichst spezifisch (Ort + Zeitraum + Event-Typ).
        zeitraum: Zielzeitraum, z.B. "April 2026".
        max_results: 1-15, Default 12.

    Returns:
        Liste von {url, title, snippet}.
    """
    print(
        f"  → discover_candidates(query={query!r}, zeitraum={zeitraum!r})",
        file=sys.stderr,
        flush=True,
    )
    try:
        raw = ddg_search(query=query, max_results=max_results)
    except Exception as e:
        print(
            f"     ← discover_candidates Fehler: {e!r}",
            file=sys.stderr,
            flush=True,
        )
        return []
    out: list[dict] = []
    seen_urls: set[str] = set()
    for r in raw:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        content = r.get("content") or ""
        snippet = content[:_MAX_SNIPPET_CHARS].strip()
        out.append(
            {
                "url": url,
                "title": (r.get("title") or "").strip(),
                "snippet": snippet,
            }
        )
    print(
        f"     ← {len(out)} Kandidaten (getrimmt)",
        file=sys.stderr,
        flush=True,
    )
    return out
