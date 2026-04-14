from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RawNerSpan:
    """NER backend 的统一输出结构，先只保留 mention 融合真正需要的字段。"""

    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    backend_name: str


class NerBackend(Protocol):
    """不同 NER 后端统一走 extract 接口，避免 pipeline 和具体模型强耦合。"""

    def extract(self, text: str) -> list[RawNerSpan]:
        ...
