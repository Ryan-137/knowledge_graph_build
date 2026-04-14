from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from src.entity.ner_backends import NerBackend, SpacyNerBackend
from src.preprocess.shared import read_json, read_jsonl, write_jsonl
from src.schema.entity import (
    EntityMentionRecord,
    MentionCandidate,
    MentionMergeConfig,
    SeedAliasRecord,
    SeedEntityRecord,
)
#candidates是一个MentionCandidate类

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MENTION_RULES_PATH = REPO_ROOT / "configs" / "entity" / "mention_rules.json"
DEFAULT_SPACY_NER_CONFIG_PATH = REPO_ROOT / "configs" / "entity" / "ner_spacy.json"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _load_seed_entities(seed_entities_path: Path) -> dict[str, SeedEntityRecord]:
    return {
        item["entity_id"]: SeedEntityRecord(**item)
        for item in read_json(seed_entities_path)
    }


def _load_seed_aliases(seed_aliases_path: Path) -> list[SeedAliasRecord]:
    return [SeedAliasRecord(**item) for item in read_json(seed_aliases_path)]


def _load_mention_merge_config(config_path: Path) -> MentionMergeConfig:
    """把 mention 融合规则集中读取，避免策略散落在代码中。"""

    return MentionMergeConfig(**read_json(config_path))


def _build_alias_entries(
    entities_by_id: dict[str, SeedEntityRecord],
    aliases: list[SeedAliasRecord],
) -> list[dict[str, str | float]]:
    entries: list[dict[str, str | float]] = []

    # 规范名也要参与 mention 抽取，否则会出现 seed 有实体、文本层却抽不出来的断层。
    for entity in entities_by_id.values():
        entries.append(
            {
                "entity_id": entity.entity_id,
                "alias": entity.canonical_name,
                "entity_type": entity.entity_type,
                "confidence": 1.0,
                "source": "canonical_name",
            }
        )

    for alias in aliases:
        entity = entities_by_id.get(alias.entity_id)
        if entity is None:
            continue
        entries.append(
            {
                "entity_id": alias.entity_id,
                "alias": alias.alias,
                "entity_type": entity.entity_type,
                "confidence": alias.confidence,
                "source": f"seed_alias:{alias.alias_type}",
            }
        )

    # 长 alias 优先，避免短别名先抢占掉长别名 span。
    entries.sort(key=lambda item: len(str(item["alias"])), reverse=True)
    return entries


def _is_wordish(char: str) -> bool:
    return char.isalnum() or char == "_"


def _is_valid_boundary(text: str, start: int, end: int) -> bool:
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    if left and _is_wordish(left):
        return False
    if right and _is_wordish(right):
        return False
    return True


def _collect_dictionary_candidates(
    sentence_text: str,
    alias_entries: list[dict[str, str | float]],
) -> list[MentionCandidate]:
    occupied_spans: list[tuple[int, int]] = []
    candidates: list[MentionCandidate] = []

    for alias_entry in alias_entries:
        alias = str(alias_entry["alias"])
        if not alias:
            continue

        for match in re.finditer(re.escape(alias), sentence_text, flags=re.IGNORECASE):
            start = match.start()
            end = match.end()

            if not _is_valid_boundary(sentence_text, start, end):
                continue #边界检查

            if any(not (end <= existing_start or start >= existing_end) for existing_start, existing_end in occupied_spans):
                continue

            occupied_spans.append((start, end))
            candidates.append(
                MentionCandidate(
                    text=sentence_text[start:end],
                    start=start,
                    end=end,
                    entity_type=str(alias_entry["entity_type"]),
                    confidence=float(alias_entry["confidence"]),
                    extractor=str(alias_entry["source"]),
                    source_seed_entity_id=str(alias_entry["entity_id"]),
                    review_status="auto",
                    source_kind="dictionary",
                )
            )

    return candidates


def _collect_ner_candidates(sentence_text: str, ner_backend: NerBackend | None) -> list[MentionCandidate]:
    if ner_backend is None:
        return []

    candidates: list[MentionCandidate] = []
    for span in ner_backend.extract(sentence_text):
        if not _is_valid_boundary(sentence_text, span.start, span.end):
            continue

        candidates.append(
            MentionCandidate(
                text=span.text,
                start=span.start,
                end=span.end,
                entity_type=span.entity_type,
                confidence=span.confidence,
                extractor=span.backend_name,
                source_seed_entity_id="",
                review_status="auto",
                source_kind="ner",
            )
        )

    return candidates


def _candidate_priority(candidate: MentionCandidate, config: MentionMergeConfig) -> tuple[int, int, float]:
    """
    融合裁决优先级。

    1. 词典精确命中优先于 NER
    2. 更长 span 优先
    3. 置信度更高优先
    """

    source_priority = config.source_priority.get(candidate.source_kind, 0)
    span_length = candidate.end - candidate.start
    return (source_priority, span_length, candidate.confidence)


def _merge_same_span_candidates(
    candidates: list[MentionCandidate],
    config: MentionMergeConfig,
) -> MentionCandidate:
    best_candidate = max(candidates, key=lambda item: _candidate_priority(item, config))
    extractor_names = sorted({candidate.extractor for candidate in candidates})
    merged_extractor = "+".join(extractor_names)

    review_status = "auto"
    entity_types = {candidate.entity_type for candidate in candidates}
    if len(entity_types) > 1:
        review_status = config.conflict_review_status

    if any(candidate.source_kind == "dictionary" for candidate in candidates) and any(
        candidate.source_kind == "ner" for candidate in candidates
    ):
        merged_extractor = config.dictionary_ner_extractor_name

    return MentionCandidate(
        text=best_candidate.text,
        start=best_candidate.start,
        end=best_candidate.end,
        entity_type=best_candidate.entity_type,
        confidence=min(
            config.max_confidence,
            max(candidate.confidence for candidate in candidates)
            + (config.multi_source_confidence_bonus if len(candidates) > 1 else 0),
        ),
        extractor=merged_extractor,
        source_seed_entity_id=best_candidate.source_seed_entity_id,
        review_status=review_status if best_candidate.review_status == "auto" else best_candidate.review_status,
        source_kind=best_candidate.source_kind,
    )


def _merge_candidates(candidates: list[MentionCandidate], config: MentionMergeConfig) -> list[MentionCandidate]:
    if not candidates:
        return []

    grouped: dict[tuple[int, int], list[MentionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.start, candidate.end), []).append(candidate)

    merged_same_span = [_merge_same_span_candidates(group, config) for group in grouped.values()]
    merged_same_span.sort(
        key=lambda item: (
            item.start,
            item.end,
            -_candidate_priority(item, config)[0],
            -(item.end - item.start),
        )
    )

    final_candidates: list[MentionCandidate] = []
    for candidate in merged_same_span:
        conflicting_index: int | None = None
        for index, accepted in enumerate(final_candidates):
            if not (candidate.end <= accepted.start or candidate.start >= accepted.end):
                conflicting_index = index
                break

        if conflicting_index is None:
            final_candidates.append(candidate)
            continue

        accepted = final_candidates[conflicting_index]
        if _candidate_priority(candidate, config) > _candidate_priority(accepted, config):
            final_candidates[conflicting_index] = candidate

    final_candidates.sort(key=lambda item: (item.start, item.end))
    return final_candidates


def _build_mention_records(
    sentence: dict[str, object],
    candidates: list[MentionCandidate],
    mention_counter_start: int,
) -> tuple[list[EntityMentionRecord], int]:
    mention_counter = mention_counter_start
    mentions: list[EntityMentionRecord] = []

    for candidate in candidates:
        mention_counter += 1
        mentions.append(
            EntityMentionRecord(
                mention_id=f"men_{mention_counter:06d}",
                sentence_id=str(sentence["sentence_id"]),
                doc_id=str(sentence["doc_id"]),
                source_id=str(sentence["source_id"]),
                text=candidate.text,
                normalized_text=_normalize_text(candidate.text),
                entity_type_pred=candidate.entity_type,
                offset_start=int(sentence["offset_start"]) + candidate.start,
                offset_end=int(sentence["offset_start"]) + candidate.end,
                extractor=candidate.extractor,
                confidence=candidate.confidence,
                source_seed_entity_id=candidate.source_seed_entity_id,
                review_status=candidate.review_status,
            )
        )

    return mentions, mention_counter


def _build_ner_backend(
    ner_backend_name: str,
    ner_model_name: str,
    ner_config_path: Path,
) -> NerBackend | None:
    if ner_backend_name == "none":
        return None
    if ner_backend_name == "spacy":
        return SpacyNerBackend(model_name=ner_model_name, config_path=ner_config_path)
    raise ValueError(f"不支持的 ner_backend: {ner_backend_name}")


def build_entity_mentions(
    sentences_path: Path,
    seed_entities_path: Path,
    seed_aliases_path: Path,
    ner_backend_name: str = "spacy",
    ner_model_name: str = "en_core_web_sm",
    ner_backend: NerBackend | None = None,
    mention_rules_path: Path = DEFAULT_MENTION_RULES_PATH,
    ner_config_path: Path = DEFAULT_SPACY_NER_CONFIG_PATH,
) -> list[EntityMentionRecord]:
    entities_by_id = _load_seed_entities(seed_entities_path)
    aliases = _load_seed_aliases(seed_aliases_path)
    alias_entries = _build_alias_entries(entities_by_id, aliases)
    sentences = read_jsonl(sentences_path)
    mention_merge_config = _load_mention_merge_config(mention_rules_path)

    resolved_ner_backend = (
        ner_backend
        if ner_backend is not None
        else _build_ner_backend(
            ner_backend_name=ner_backend_name,
            ner_model_name=ner_model_name,
            ner_config_path=ner_config_path,
        )
    )

    all_mentions: list[EntityMentionRecord] = []
    mention_counter = 0
    for sentence in sentences:
        sentence_text = str(sentence["text"])
        dictionary_candidates = _collect_dictionary_candidates(sentence_text=sentence_text, alias_entries=alias_entries)
        ner_candidates = _collect_ner_candidates(sentence_text=sentence_text, ner_backend=resolved_ner_backend)
        merged_candidates = _merge_candidates(
            dictionary_candidates + ner_candidates,
            mention_merge_config,
        )

        sentence_mentions, mention_counter = _build_mention_records(
            sentence=sentence,
            candidates=merged_candidates,
            mention_counter_start=mention_counter,
        )
        all_mentions.extend(sentence_mentions)

    return all_mentions


def run_entity_extraction(
    sentences_path: Path,
    seed_entities_path: Path,
    seed_aliases_path: Path,
    output_path: Path,
    ner_backend_name: str = "spacy",
    ner_model_name: str = "en_core_web_sm",
    mention_rules_path: Path = DEFAULT_MENTION_RULES_PATH,
    ner_config_path: Path = DEFAULT_SPACY_NER_CONFIG_PATH,
) -> int:
    mentions = build_entity_mentions(
        sentences_path=sentences_path,
        seed_entities_path=seed_entities_path,
        seed_aliases_path=seed_aliases_path,
        ner_backend_name=ner_backend_name,
        ner_model_name=ner_model_name,
        mention_rules_path=mention_rules_path,
        ner_config_path=ner_config_path,
    )
    write_jsonl(output_path, [asdict(item) for item in mentions])
    return len(mentions)
