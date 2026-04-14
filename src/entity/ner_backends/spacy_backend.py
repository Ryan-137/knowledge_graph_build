from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.entity.ner_backends.base import RawNerSpan
from src.preprocess.shared import read_json
from src.schema.entity import SpacyNerConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPACY_NER_CONFIG_PATH = REPO_ROOT / "configs" / "entity" / "ner_spacy.json"


def _load_spacy_ner_config(config_path: Path) -> SpacyNerConfig:
    """把标签映射和经验分集中放到配置里，便于后续替换模型或调参。"""

    return SpacyNerConfig(**read_json(config_path))


@dataclass
class SpacyNerBackend:
    """
    spaCy NER backend。

    spaCy 默认实体结果不直接暴露 span 级概率，因此这里用“标签经验分 + 后续融合规则”
    的方式作为第一版工程实现。后面如果切换到 transformers backend，再把分数替换成
    模型原生 logits / softmax 即可。
    """

    model_name: str = "en_core_web_sm"
    config_path: Path = DEFAULT_SPACY_NER_CONFIG_PATH

    def __post_init__(self) -> None:
        self._config = _load_spacy_ner_config(self.config_path)

        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - 环境缺失时给出清晰报错
            raise RuntimeError(
                "当前环境缺少 spaCy。请先在 knowgraph 环境中安装 spacy 和英文模型后再运行 NER 融合。"
            ) from exc

        try:
            self._nlp = spacy.load(self.model_name)
        except OSError as exc:  # pragma: no cover - 模型缺失时给出清晰报错
            raise RuntimeError(
                f"当前环境缺少 spaCy 模型 {self.model_name}。请先安装该模型后再运行 NER 融合。"
            ) from exc

    def extract(self, text: str) -> list[RawNerSpan]:
        doc = self._nlp(text)
        spans: list[RawNerSpan] = []

        for ent in doc.ents:
            mapped_type = self._config.label_to_entity_type.get(ent.label_)
            if mapped_type is None:
                continue

            spans.append(
                RawNerSpan(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    entity_type=mapped_type,
                    confidence=self._config.label_confidence.get(ent.label_, self._config.default_confidence),
                    backend_name=f"spacy:{self.model_name}",
                )
            )

        return spans
