from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedEntityRecord:
    """种子知识库里的规范实体。"""

    entity_id: str
    entity_type: str
    canonical_name: str
    description: str
    source: str
    status: str


@dataclass(frozen=True)
class SeedAliasRecord:
    """实体别名，供 mention 抽取和 linking 共用。"""

    alias_id: str
    entity_id: str
    alias: str
    alias_type: str
    language: str
    confidence: float


@dataclass(frozen=True)
class SeedFactRecord:
    """少量高置信种子事实，用于后续关系抽取和融合时做先验。"""

    seed_fact_id: str
    head_entity_id: str
    relation_type: str
    tail_entity_id: str
    confidence: float
    evidence_note: str


@dataclass(frozen=True)
class EntityMentionRecord:
    """mention 层输出，只表达文本里出现了什么，不直接表达 canonical 实体。"""

    mention_id: str
    sentence_id: str
    doc_id: str
    source_id: str
    text: str
    normalized_text: str
    entity_type_pred: str
    offset_start: int
    offset_end: int
    extractor: str
    confidence: float
    source_seed_entity_id: str = ""
    review_status: str = "auto"


@dataclass(frozen=True)
class MentionCandidate:
    """mention 融合前的统一候选结构，供词典流和 NER 流共用。"""

    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    extractor: str
    source_seed_entity_id: str = ""
    review_status: str = "auto"
    source_kind: str = "dictionary"


@dataclass(frozen=True)
class MentionMergeConfig:
    """mention 融合规则配置。"""

    source_priority: dict[str, int]
    multi_source_confidence_bonus: float
    max_confidence: float
    conflict_review_status: str
    dictionary_ner_extractor_name: str


@dataclass(frozen=True)
class CorefLinkRecord:
    """
    指代消解结果。

    这里的 from_mention_id 使用临时代词锚点 ID，而不是把代词塞回 mention 层，
    这样可以继续保持 mention 层只保留实体提及，不混入 pronoun 类型。
    """

    coref_id: str
    from_mention_id: str
    to_mention_id: str
    doc_id: str
    source_id: str
    sentence_id: str
    pronoun_text: str
    pronoun_offset_start: int
    pronoun_offset_end: int
    coref_type: str
    confidence: float


@dataclass(frozen=True)
class CorefRuleConfig:
    """规则型共指解析配置。"""

    person_pronouns: list[str]
    non_person_pronouns: list[str]
    max_sentence_gap: int
    confidence_by_sentence_gap: dict[str, float]
    fallback_confidence: float


@dataclass(frozen=True)
class CanonicalEntityRecord:
    """最终规范实体层。"""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    source: str = "seed_linking"
    status: str = "candidate"


@dataclass(frozen=True)
class CandidateScoreRecord:
    """linking 候选分数，便于后续人工复核。"""

    candidate_entity_id: str
    score: float


@dataclass(frozen=True)
class SpacyNerConfig:
    """spaCy NER 标签映射与经验分配置。"""

    label_to_entity_type: dict[str, str]
    label_confidence: dict[str, float]
    default_confidence: float


@dataclass(frozen=True)
class LinkingRuleConfig:
    """mention linking 阶段的裁决规则配置。"""

    seed_match_method: str
    tie_needs_review: bool
    provisional_entity_prefix: str
    provisional_source: str
    provisional_status: str
    provisional_method: str
    provisional_needs_review: bool


@dataclass(frozen=True)
class EntityLinkRecord:
    """mention 到 canonical entity 的映射结果。"""

    link_id: str
    mention_id: str
    entity_id: str
    method: str
    candidate_rank: int
    confidence: float
    is_manual_confirmed: bool
    needs_review: bool
    candidate_scores: list[CandidateScoreRecord] = field(default_factory=list)
