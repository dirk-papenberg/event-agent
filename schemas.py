from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EventCategory = Literal[
    "flohmarkt",
    "zirkus",
    "markt",
    "jahreszeitliches_fest",
    "ritter",
    "kindertheater",
    "museum_kinder",
    "spielplatz_event",
    "stadtfest",
    "sonstiges",
]


class Event(BaseModel):
    title: str
    description: str = Field(description="2-4 Sätze, was bei der Veranstaltung passiert und warum sie für 5-7-Jährige interessant ist.")
    start_date: date | None = None
    end_date: date | None = None
    start_time: str | None = Field(default=None, description="z.B. '10:00'")
    city: str
    venue: str | None = Field(default=None, description="Veranstaltungsort-Name, z.B. 'Schlosspark Nymphenburg'")
    address: str | None = None
    category: EventCategory
    age_fit_note: str = Field(description="Kurze Einschätzung zur Eignung für 5-7-Jährige.")
    price_note: str | None = Field(default=None, description="z.B. 'Eintritt frei' oder 'Erwachsene 8€, Kinder 4€'")
    source_url: str


class EventList(BaseModel):
    events: list[Event]
    search_summary: str = Field(description="Kurze Zusammenfassung der Suche: wie viele Quellen durchsucht, welche Kategorien abgedeckt, was ausgelassen wurde.")


class CandidateResult(BaseModel):
    is_event: bool = Field(description="True, wenn die Seite ein konkretes, verifizierbares Familien-/Kinder-Event im Zielzeitraum beschreibt.")
    event: Event | None = Field(default=None, description="Extrahiertes Event, nur gesetzt wenn is_event=true. Andernfalls null.")
    reason: str = Field(description="Kurzbegründung in 1-2 Sätzen, warum geeignet/ungeeignet.")

    @field_validator("event", mode="before")
    @classmethod
    def _coerce_event(cls, v):
        if v is None or isinstance(v, (dict, Event)):
            return v
        return None
