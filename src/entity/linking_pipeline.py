from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from src.preprocess.shared import read_json, read_jsonl, write_json, write_jsonl
from src.schema.entity import (
    CandidateScoreRecord,
    CanonicalEntityRecord,
    EntityLinkRecord,
    EntityMentionRecord,
    LinkingRuleConfig,
    SeedAliasRecord,
    SeedEntityRecord,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINKING_RULES_PATH = REPO_ROOT / "configs" / "entity" / "linking_rules.json"
PUNCTUATION_NORMALIZATION_TABLE = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


def _load_seed_entities(seed_entities_path: Path) -> dict[str, SeedEntityRecord]:
    return {
        item["entity_id"]: SeedEntityRecord(**item)
        for item in read_json(seed_entities_path)
    }


def _load_seed_aliases(seed_aliases_path: Path) -> list[SeedAliasRecord]:
    return [SeedAliasRecord(**item) for item in read_json(seed_aliases_path)]


def _load_mentions(mentions_path: Path) -> list[EntityMentionRecord]:
    return [EntityMentionRecord(**item) for item in read_jsonl(mentions_path)]


def _load_linking_rule_config(config_path: Path) -> LinkingRuleConfig:
    """把 linking 的裁决规则集中到配置，便于统一调整策略。"""

    return LinkingRuleConfig(**read_json(config_path))


def _normalize(text: str) -> str:
    """统一折叠大小写、空白和常见 PDF/HTML 变体标点，保证 alias 命中稳定。"""

    normalized = text.casefold().translate(PUNCTUATION_NORMALIZATION_TABLE)
    return re.sub(r"\s+", " ", normalized).strip()


def _build_alias_index(
    seed_entities_by_id: dict[str, SeedEntityRecord],
    seed_aliases: list[SeedAliasRecord],
) -> dict[str, list[tuple[str, float]]]:
    ''' 类似这样，每个名字和别名对应的实体以及概率
    {
        "alan turing": [("ent_001", 1.0)],
        "turing": [("ent_001", 0.95), ("ent_099", 0.80)],
        "turing award": [("ent_099", 1.0)]
    }
    '''
    alias_index: dict[str, list[tuple[str, float]]] = {}

    for entity in seed_entities_by_id.values():
        alias_index.setdefault(_normalize(entity.canonical_name), []).append((entity.entity_id, 1.0))

    for alias in seed_aliases:
        alias_index.setdefault(_normalize(alias.alias), []).append((alias.entity_id, alias.confidence))

    return alias_index


def build_entity_links(
    mentions_path: Path,
    seed_entities_path: Path,
    seed_aliases_path: Path,
    linking_rules_path: Path = DEFAULT_LINKING_RULES_PATH,
) -> tuple[list[CanonicalEntityRecord], list[EntityLinkRecord]]:
    seed_entities_by_id = _load_seed_entities(seed_entities_path)
    seed_aliases = _load_seed_aliases(seed_aliases_path)
    mentions = _load_mentions(mentions_path)
    alias_index = _build_alias_index(seed_entities_by_id, seed_aliases)
    linking_config = _load_linking_rule_config(linking_rules_path)

    links: list[EntityLinkRecord] = []
    linked_entity_ids: set[str] = set()  #去重
    linked_entity_order: list[str] = []
    provisional_entities: dict[tuple[str, str], CanonicalEntityRecord] = {}
    provisional_entity_counter = 0

    for index, mention in enumerate(mentions, start=1):
        mention_lookup_key = _normalize(mention.normalized_text or mention.text) #优先用正则化文本 没有就用原始文本 然后再次正则化
        candidates = alias_index.get(mention_lookup_key, [])

        if mention.source_seed_entity_id:
            # mention 抽取阶段如果已经明确命中 seed 实体，则优先直接沿用这条锚点。
            candidates = [(mention.source_seed_entity_id, 1.0)]

        if candidates:
            # 先按分数，再按 entity_id 排序，保证输出稳定可复现。
            ranked_candidates = sorted(candidates, key=lambda item: (-item[1], item[0]))
            best_entity_id, best_score = ranked_candidates[0]
            method = linking_config.seed_match_method
            has_score_tie = (
                linking_config.tie_needs_review
                and len(ranked_candidates) > 1
                and ranked_candidates[0][1] == ranked_candidates[1][1]
            )
            needs_review = mention.review_status != "auto" or has_score_tie
        else:
            # NER 抽到但 seed 里没有的 mention 不能直接丢弃，
            # 否则 mention 层和 canonical 层又会重新断开。
            provisional_key = (mention.entity_type_pred, mention_lookup_key)
            if provisional_key not in provisional_entities:
                provisional_entity_counter += 1
                provisional_entity_id = (
                    f"{linking_config.provisional_entity_prefix}{provisional_entity_counter:04d}"
                )
                provisional_entities[provisional_key] = CanonicalEntityRecord(
                    entity_id=provisional_entity_id,
                    entity_type=mention.entity_type_pred,
                    canonical_name=mention.text,
                    aliases=[mention.text],
                    description="",
                    source=linking_config.provisional_source,
                    status=linking_config.provisional_status,
                )
            provisional_entity = provisional_entities[provisional_key]
            ranked_candidates = [(provisional_entity.entity_id, mention.confidence)]
            best_entity_id, best_score = ranked_candidates[0]
            method = linking_config.provisional_method
            needs_review = linking_config.provisional_needs_review

        if best_entity_id not in linked_entity_ids:
            linked_entity_ids.add(best_entity_id)
            linked_entity_order.append(best_entity_id)

        links.append(
            EntityLinkRecord(
                link_id=f"link_{index:06d}",
                mention_id=mention.mention_id,
                entity_id=best_entity_id,
                method=method,
                candidate_rank=1,
                confidence=best_score,
                is_manual_confirmed=False,
                needs_review=needs_review,
                candidate_scores=[
                    CandidateScoreRecord(candidate_entity_id=entity_id, score=score)
                    for entity_id, score in ranked_candidates
                ],
            )
        )

    aliases_by_entity: dict[str, set[str]] = {}
    for entity in seed_entities_by_id.values():
        aliases_by_entity.setdefault(entity.entity_id, set()).add(entity.canonical_name)
    for alias in seed_aliases:
        aliases_by_entity.setdefault(alias.entity_id, set()).add(alias.alias)

    canonical_entities = []
    for entity_id in linked_entity_order:
        if entity_id in seed_entities_by_id:
            entity = seed_entities_by_id[entity_id]
            canonical_entities.append(
                CanonicalEntityRecord(
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    canonical_name=entity.canonical_name,
                    aliases=sorted(aliases_by_entity.get(entity.entity_id, set())),
                    description=entity.description,
                    source=entity.source,
                    status=entity.status,
                )
            )
            continue

        for provisional_entity in provisional_entities.values():
            if provisional_entity.entity_id == entity_id:
                canonical_entities.append(provisional_entity)
                break

    return canonical_entities, links


def run_entity_linking(
    mentions_path: Path,
    seed_entities_path: Path,
    seed_aliases_path: Path,
    canonical_output_path: Path,
    links_output_path: Path,
    linking_rules_path: Path = DEFAULT_LINKING_RULES_PATH,
) -> tuple[int, int]:
    canonical_entities, links = build_entity_links(
        mentions_path=mentions_path,
        seed_entities_path=seed_entities_path,
        seed_aliases_path=seed_aliases_path,
        linking_rules_path=linking_rules_path,
    )
    write_json(canonical_output_path, [asdict(item) for item in canonical_entities])
    write_jsonl(links_output_path, [asdict(item) for item in links])
    return len(canonical_entities), len(links)
