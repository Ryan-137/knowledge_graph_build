"""NER backend 统一导出。"""

from src.entity.ner_backends.base import NerBackend, RawNerSpan
from src.entity.ner_backends.spacy_backend import SpacyNerBackend

__all__ = ["NerBackend", "RawNerSpan", "SpacyNerBackend"]
