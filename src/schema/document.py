from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_id: str
    title: str
    tier: int
    language: str
    clean_text: str




@dataclass
class SourceRecord:
    source_id: str
    title: str
    tier: int
    authority_level: str
    source_type: str
    original_url: str
    raw_path: str
    organization: str
    verification_status: str
    notes: str
