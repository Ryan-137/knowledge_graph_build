from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.entity import (
    run_coref_resolution,
    run_entity_extraction,
    run_entity_linking,
    run_seed_build,
)
from src.preprocess import run_document_preprocess, run_sentence_preprocess


DEFAULT_SOURCES = "configs/sources.yaml"
DEFAULT_DOCUMENTS = "data/processed/documents.jsonl"
DEFAULT_DOCUMENT_REPORT = "data/processed/document_preprocess_report.json"
DEFAULT_SENTENCES = "data/processed/sentences.jsonl"
DEFAULT_SENTENCE_REPORT = "data/processed/sentence_preprocess_report.json"
DEFAULT_SEED_CONFIG = "configs/seeds/manual_seed.json"
DEFAULT_SEED_ENTITIES = "data/seed/seed_entities.json"
DEFAULT_SEED_ALIASES = "data/seed/seed_aliases.json"
DEFAULT_SEED_FACTS = "data/seed/seed_facts.jsonl"
DEFAULT_ENTITY_MENTIONS = "data/extracted/entities.jsonl"
DEFAULT_COREF_LINKS = "data/extracted/coref_links.jsonl"
DEFAULT_CANONICAL_ENTITIES = "data/fused/entity_canonical.json"
DEFAULT_ENTITY_LINKS = "data/fused/entity_links.jsonl"


def _resolve_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _build_preprocess_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="执行预处理流程。",
        description="执行文档级、句子级或全量预处理流程。",
    )
    preprocess_parser.add_argument(
        "stage",
        nargs="?",
        choices=("documents", "sentences", "all"),
        default="all",
        help="要执行的预处理阶段，默认执行全量流程。",
    )
    preprocess_parser.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help="来源登记文件路径。",
    )
    preprocess_parser.add_argument(
        "--documents",
        default=DEFAULT_DOCUMENTS,
        help="文档级预处理结果路径。",
    )
    preprocess_parser.add_argument(
        "--document-report",
        default=DEFAULT_DOCUMENT_REPORT,
        help="文档级预处理报告路径。",
    )
    preprocess_parser.add_argument(
        "--sentences",
        default=DEFAULT_SENTENCES,
        help="句子级预处理结果路径。",
    )
    preprocess_parser.add_argument(
        "--sentence-report",
        default=DEFAULT_SENTENCE_REPORT,
        help="句子级预处理报告路径。",
    )
    preprocess_parser.add_argument(
        "--strict",
        action="store_true",
        help="只要存在错误就返回失败。",
    )
    preprocess_parser.set_defaults(handler=_handle_preprocess)


def _build_seed_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    seed_parser = subparsers.add_parser(
        "seed",
        help="构建种子知识库。",
        description="根据手工维护的 seed 配置生成实体、别名和事实文件。",
    )
    seed_parser.add_argument("--config", default=DEFAULT_SEED_CONFIG, help="种子配置文件路径。")
    seed_parser.add_argument("--entities", default=DEFAULT_SEED_ENTITIES, help="种子实体输出路径。")
    seed_parser.add_argument("--aliases", default=DEFAULT_SEED_ALIASES, help="种子别名输出路径。")
    seed_parser.add_argument("--facts", default=DEFAULT_SEED_FACTS, help="种子事实输出路径。")
    seed_parser.set_defaults(handler=_handle_seed)


def _build_entities_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    entities_parser = subparsers.add_parser(
        "entities",
        help="执行实体提及抽取。",
        description="从句子层输出中抽取 mention 级实体提及。",
    )
    entities_parser.add_argument("--sentences", default=DEFAULT_SENTENCES, help="句子层输入路径。")
    entities_parser.add_argument("--seed-entities", default=DEFAULT_SEED_ENTITIES, help="种子实体输入路径。")
    entities_parser.add_argument("--seed-aliases", default=DEFAULT_SEED_ALIASES, help="种子别名输入路径。")
    entities_parser.add_argument(
        "--ner-backend",
        choices=("spacy", "none"),
        default="spacy",
        help="mention 抽取使用的 NER backend；默认启用 spaCy。",
    )
    entities_parser.add_argument(
        "--ner-model",
        default="en_core_web_sm",
        help="NER backend 使用的模型名称。",
    )
    entities_parser.add_argument("--output", default=DEFAULT_ENTITY_MENTIONS, help="mention 输出路径。")
    entities_parser.set_defaults(handler=_handle_entities)


def _build_coref_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    coref_parser = subparsers.add_parser(
        "coref",
        help="执行轻量级指代消解。",
        description="在最近上下文窗口里为 he/it 等代词建立共指线索。",
    )
    coref_parser.add_argument("--sentences", default=DEFAULT_SENTENCES, help="句子层输入路径。")
    coref_parser.add_argument("--mentions", default=DEFAULT_ENTITY_MENTIONS, help="mention 输入路径。")
    coref_parser.add_argument("--output", default=DEFAULT_COREF_LINKS, help="共指输出路径。")
    coref_parser.set_defaults(handler=_handle_coref)


def _build_link_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    link_parser = subparsers.add_parser(
        "link",
        help="执行实体链接。",
        description="将 mention 层结果映射到 canonical entity 层。",
    )
    link_parser.add_argument("--mentions", default=DEFAULT_ENTITY_MENTIONS, help="mention 输入路径。")
    link_parser.add_argument("--seed-entities", default=DEFAULT_SEED_ENTITIES, help="种子实体输入路径。")
    link_parser.add_argument("--seed-aliases", default=DEFAULT_SEED_ALIASES, help="种子别名输入路径。")
    link_parser.add_argument("--canonical-output", default=DEFAULT_CANONICAL_ENTITIES, help="规范实体输出路径。")
    link_parser.add_argument("--links-output", default=DEFAULT_ENTITY_LINKS, help="linking 输出路径。")
    link_parser.set_defaults(handler=_handle_link)


def _build_entity_pipeline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pipeline_parser = subparsers.add_parser(
        "entity-pipeline",
        help="顺序执行 seed、entities、coref、link。",
        description="统一执行实体层主流程，但仍保留各子命令可单独运行。",
    )
    pipeline_parser.add_argument("--config", default=DEFAULT_SEED_CONFIG, help="种子配置文件路径。")
    pipeline_parser.add_argument("--entities", default=DEFAULT_SEED_ENTITIES, help="种子实体输出路径。")
    pipeline_parser.add_argument("--aliases", default=DEFAULT_SEED_ALIASES, help="种子别名输出路径。")
    pipeline_parser.add_argument("--facts", default=DEFAULT_SEED_FACTS, help="种子事实输出路径。")
    pipeline_parser.add_argument("--sentences", default=DEFAULT_SENTENCES, help="句子层输入路径。")
    pipeline_parser.add_argument(
        "--ner-backend",
        choices=("spacy", "none"),
        default="spacy",
        help="mention 抽取使用的 NER backend；默认启用 spaCy。",
    )
    pipeline_parser.add_argument(
        "--ner-model",
        default="en_core_web_sm",
        help="NER backend 使用的模型名称。",
    )
    pipeline_parser.add_argument("--mentions-output", default=DEFAULT_ENTITY_MENTIONS, help="mention 输出路径。")
    pipeline_parser.add_argument("--coref-output", default=DEFAULT_COREF_LINKS, help="共指输出路径。")
    pipeline_parser.add_argument("--canonical-output", default=DEFAULT_CANONICAL_ENTITIES, help="规范实体输出路径。")
    pipeline_parser.add_argument("--links-output", default=DEFAULT_ENTITY_LINKS, help="linking 输出路径。")
    pipeline_parser.set_defaults(handler=_handle_entity_pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="knowledge_graph 项目主入口。")
    subparsers = parser.add_subparsers(dest="command")
    _build_preprocess_parser(subparsers)
    _build_seed_parser(subparsers)
    _build_entities_parser(subparsers)
    _build_coref_parser(subparsers)
    _build_link_parser(subparsers)
    _build_entity_pipeline_parser(subparsers)
    return parser


def _run_document_stage(args: argparse.Namespace, repo_root: Path) -> None:
    document_count, error_count = run_document_preprocess(
        repo_root=repo_root,
        config_path=_resolve_path(repo_root, args.sources),
        output_path=_resolve_path(repo_root, args.documents),
        report_path=_resolve_path(repo_root, args.document_report),
        strict=args.strict,
    )
    print(f"文档级预处理完成：成功 {document_count} 个，错误 {error_count} 个。")
    print(f"documents.jsonl: {_resolve_path(repo_root, args.documents).as_posix()}")
    print(f"report.json: {_resolve_path(repo_root, args.document_report).as_posix()}")


def _run_sentence_stage(args: argparse.Namespace, repo_root: Path) -> None:
    sentence_count, error_count = run_sentence_preprocess(
        documents_path=_resolve_path(repo_root, args.documents),
        output_path=_resolve_path(repo_root, args.sentences),
        report_path=_resolve_path(repo_root, args.sentence_report),
        strict=args.strict,
    )
    print(f"句子级预处理完成：成功 {sentence_count} 条，错误 {error_count} 条。")
    print(f"sentences.jsonl: {_resolve_path(repo_root, args.sentences).as_posix()}")
    print(f"report.json: {_resolve_path(repo_root, args.sentence_report).as_posix()}")


def _handle_preprocess(args: argparse.Namespace, repo_root: Path) -> int:
    if args.stage in {"documents", "all"}:
        _run_document_stage(args, repo_root)

    if args.stage in {"sentences", "all"}:
        _run_sentence_stage(args, repo_root)

    return 0


def _handle_seed(args: argparse.Namespace, repo_root: Path) -> int:
    entity_count, alias_count, fact_count = run_seed_build(
        config_path=_resolve_path(repo_root, args.config),
        entities_output_path=_resolve_path(repo_root, args.entities),
        aliases_output_path=_resolve_path(repo_root, args.aliases),
        facts_output_path=_resolve_path(repo_root, args.facts),
    )
    print(f"种子知识库构建完成：实体 {entity_count} 个，别名 {alias_count} 个，事实 {fact_count} 条。")
    return 0


def _handle_entities(args: argparse.Namespace, repo_root: Path) -> int:
    mention_count = run_entity_extraction(
        sentences_path=_resolve_path(repo_root, args.sentences),
        seed_entities_path=_resolve_path(repo_root, args.seed_entities),
        seed_aliases_path=_resolve_path(repo_root, args.seed_aliases),
        output_path=_resolve_path(repo_root, args.output),
        ner_backend_name=args.ner_backend,
        ner_model_name=args.ner_model,
    )
    print(f"实体提及抽取完成：共输出 {mention_count} 条 mention。")
    return 0


def _handle_coref(args: argparse.Namespace, repo_root: Path) -> int:
    coref_count = run_coref_resolution(
        sentences_path=_resolve_path(repo_root, args.sentences),
        mentions_path=_resolve_path(repo_root, args.mentions),
        output_path=_resolve_path(repo_root, args.output),
    )
    print(f"指代消解完成：共输出 {coref_count} 条 coref link。")
    return 0


def _handle_link(args: argparse.Namespace, repo_root: Path) -> int:
    canonical_count, link_count = run_entity_linking(
        mentions_path=_resolve_path(repo_root, args.mentions),
        seed_entities_path=_resolve_path(repo_root, args.seed_entities),
        seed_aliases_path=_resolve_path(repo_root, args.seed_aliases),
        canonical_output_path=_resolve_path(repo_root, args.canonical_output),
        links_output_path=_resolve_path(repo_root, args.links_output),
    )
    print(f"实体链接完成：规范实体 {canonical_count} 个，link 结果 {link_count} 条。")
    return 0


def _handle_entity_pipeline(args: argparse.Namespace, repo_root: Path) -> int:
    entity_count, alias_count, fact_count = run_seed_build(
        config_path=_resolve_path(repo_root, args.config),
        entities_output_path=_resolve_path(repo_root, args.entities),
        aliases_output_path=_resolve_path(repo_root, args.aliases),
        facts_output_path=_resolve_path(repo_root, args.facts),
    )
    print(f"[1/4] 种子知识库构建完成：实体 {entity_count} 个，别名 {alias_count} 个，事实 {fact_count} 条。")

    mention_count = run_entity_extraction(
        sentences_path=_resolve_path(repo_root, args.sentences),
        seed_entities_path=_resolve_path(repo_root, args.entities),
        seed_aliases_path=_resolve_path(repo_root, args.aliases),
        output_path=_resolve_path(repo_root, args.mentions_output),
        ner_backend_name=args.ner_backend,
        ner_model_name=args.ner_model,
    )
    print(f"[2/4] 实体提及抽取完成：共输出 {mention_count} 条 mention。")

    coref_count = run_coref_resolution(
        sentences_path=_resolve_path(repo_root, args.sentences),
        mentions_path=_resolve_path(repo_root, args.mentions_output),
        output_path=_resolve_path(repo_root, args.coref_output),
    )
    print(f"[3/4] 指代消解完成：共输出 {coref_count} 条 coref link。")

    canonical_count, link_count = run_entity_linking(
        mentions_path=_resolve_path(repo_root, args.mentions_output),
        seed_entities_path=_resolve_path(repo_root, args.entities),
        seed_aliases_path=_resolve_path(repo_root, args.aliases),
        canonical_output_path=_resolve_path(repo_root, args.canonical_output),
        links_output_path=_resolve_path(repo_root, args.links_output),
    )
    print(f"[4/4] 实体链接完成：规范实体 {canonical_count} 个，link 结果 {link_count} 条。")
    return 0


def main(argv: Sequence[str] | None = None, repo_root: Path | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    actual_repo_root = repo_root or Path(__file__).resolve().parent.parent

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, actual_repo_root)
