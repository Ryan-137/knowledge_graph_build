from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeMentionRecord:
    text: str
    normalized: str
    type: str
    offset_start: int
    offset_end: int


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    doc_id: str
    source_id: str
    sentence_index_in_doc: int
    text: str
    offset_start: int
    offset_end: int
    normalized_time: list[str] = field(default_factory=list)
    time_mentions: list[TimeMentionRecord] = field(default_factory=list)
