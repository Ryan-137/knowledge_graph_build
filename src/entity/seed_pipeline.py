from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.preprocess.shared import read_json, write_json, write_jsonl
from src.schema.entity import SeedAliasRecord, SeedEntityRecord, SeedFactRecord


def build_seed_records(config_path: Path) -> tuple[list[SeedEntityRecord], list[SeedAliasRecord], list[SeedFactRecord]]:
    """
    从手工维护的 seed 配置中构建结构化对象。

    当前项目仍处于实体层基建阶段，因此先采用“人工高置信 seed + 自动抽取”的方式，
    避免在 mention/linking 尚未稳定前就把项目耦合到复杂模型调用里。
    """

    payload = read_json(config_path)

    entities = [SeedEntityRecord(**item) for item in payload.get("entities", [])]
    aliases = [SeedAliasRecord(**item) for item in payload.get("aliases", [])]
    facts = [SeedFactRecord(**item) for item in payload.get("facts", [])]
    return entities, aliases, facts


def run_seed_build(
    config_path: Path,
    entities_output_path: Path,
    aliases_output_path: Path,
    facts_output_path: Path,
) -> tuple[int, int, int]:
    entities, aliases, facts = build_seed_records(config_path=config_path)

    write_json(entities_output_path, [asdict(item) for item in entities])
    write_json(aliases_output_path, [asdict(item) for item in aliases])
    write_jsonl(facts_output_path, [asdict(item) for item in facts])

    return len(entities), len(aliases), len(facts)
