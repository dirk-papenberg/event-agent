import os
import sys

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel

from schemas import CandidateResult
from tools import fetch_page

load_dotenv()


EVALUATOR_PROMPT = """Du bist ein Extraktions-Agent. Du bekommst eine URL und einen kurzen Hinweis, warum die Seite potenziell interessant ist.

Aufgabe:
1. Rufe `fetch_page` genau einmal für die URL auf.
2. Prüfe, ob die Seite ein konkretes, verifizierbares Familien- oder Kinder-Event im angegebenen Zeitraum beschreibt.
3. Gib strukturiert zurück:
   - `is_event=true` mit vollständig befülltem `event`, wenn Datum, Ort und Eignung klar erkennbar sind.
   - `is_event=false` mit knapper `reason`, wenn unklar, abgesagt, falsches Datum, keine Familien-/Kindereignung oder Übersichtsseite ohne konkreten Termin.

Harte Regeln:
- Niemals raten. Wenn Datum oder Ort fehlen: is_event=false.
- Keine weiteren Tool-Calls außer einem fetch_page.
- Kein Vorwort, keine Fließtext-Antwort — nur strukturierter Output.
- Sprache der Felder: Deutsch.
"""


def _build_evaluator() -> Agent:
    provider = os.environ.get("MODEL_PROVIDER", "azure").lower()
    if provider == "bedrock":
        model = BedrockModel(
            model_id=os.environ.get(
                "BEDROCK_EVALUATOR_MODEL_ID",
                os.environ.get(
                    "BEDROCK_MODEL_ID",
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                ),
            ),
            max_tokens=4000,
        )
    else:
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
        )
        model = OpenAIModel(
            client=azure_client,
            model_id=os.environ.get(
                "AZURE_OPENAI_EVALUATOR_DEPLOYMENT",
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            ),
            params={"reasoning_effort": "low"},
        )
    return Agent(
        model=model,
        tools=[fetch_page],
        system_prompt=EVALUATOR_PROMPT,
    )


@tool
def evaluate_candidate(url: str, hint: str, zeitraum: str) -> dict:
    """Bewertet eine einzelne Kandidaten-URL in Isolation und extrahiert ein Event oder verwirft sie.

    Dieses Tool läuft in einem eigenen, frischen Kontext — die Rohseite landet NICHT im
    Hauptkontext. Nutze es statt direkt Seiten zu öffnen, wenn eine Suche einen
    vielversprechenden Treffer liefert.

    Args:
        url: Die zu prüfende URL (aus search_web).
        hint: 1-2 Sätze: was auf der Seite erwartet wird und warum sie relevant ist.
        zeitraum: Zielzeitraum, z.B. "April 2026", damit die Datumsprüfung sauber läuft.

    Returns:
        {"is_event": bool, "event": {...} | None, "reason": str}
    """
    print(f"  → evaluate_candidate(url={url!r})", file=sys.stderr, flush=True)
    agent = _build_evaluator()
    prompt = (
        f"URL: {url}\n"
        f"Zeitraum: {zeitraum}\n"
        f"Hinweis: {hint}\n\n"
        "Prüfe die Seite und gib strukturiert zurück."
    )
    result = agent(prompt, structured_output_model=CandidateResult)
    parsed: CandidateResult = result.structured_output
    return parsed.model_dump(mode="json")
