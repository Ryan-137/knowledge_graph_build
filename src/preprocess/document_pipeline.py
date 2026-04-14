from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
    raise RuntimeError(
        "缺少 PyYAML，请先安装 requirements.txt 中的依赖。"
    ) from exc

try:
    from bs4 import BeautifulSoup, Comment, Tag
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
    raise RuntimeError(
        "缺少 beautifulsoup4，请先安装 requirements.txt 中的依赖。"
    ) from exc

from configs.rules.document import (
    BLOCK_TAGS,
    CONTENT_SELECTORS,
    NOISE_KEYWORDS,
    NOISE_SELECTORS,
    NOISE_TAGS,
    PROTECTED_CONTENT_TAGS,
    TAIL_SECTION_TITLES,
)
from src.preprocess.shared import utc_now_iso, write_json, write_jsonl
from src.schema.document import SourceRecord


LEADING_REFERENCE_PATTERN = re.compile(
    r"^(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:[A-Z\"'“‘(]|\Z))"
)
TRAILING_REFERENCE_PATTERN = re.compile(
    r"([,.;:!?%)\]}\"'”’])\s*(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:\s|$))"
)
PURE_REFERENCE_BLOCK_PATTERN = re.compile(r"^(?:\[\s*\d+(?:\s*[-,–]\s*\d+)?\s*\]\s*)+$")
TOC_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_line(text: str) -> str:
    text = _normalize_whitespace(text)
    text = LEADING_REFERENCE_PATTERN.sub("", text)
    text = TRAILING_REFERENCE_PATTERN.sub(r"\1", text)
    text = re.sub(r"\s+([,.;:!?%)\]}\u201d\u2019])", r"\1", text)
    text = re.sub(r"([(\[{\u201c\u2018])\s+", r"\1", text)
    text = text.strip(" -|")
    return text


def _infer_language(text: str) -> str:
    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    alpha_count = sum(1 for char in text if char.isalpha())
    if chinese_count > 0 and chinese_count >= alpha_count * 0.2:
        return "zh"
    if alpha_count > 0:
        return "en"
    return "unknown"


def _contains_noise_keyword(value: str) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in NOISE_KEYWORDS)


def _is_visually_hidden(tag: Tag) -> bool:
    style = tag.get("style")
    if not isinstance(style, str):
        return False

    normalized_style = re.sub(r"\s+", "", style.lower())
    hidden_tokens = (
        "display:none",
        "visibility:hidden",
        "opacity:0",
    )
    return any(token in normalized_style for token in hidden_tokens)


def _looks_like_content_container(tag: Tag) -> bool:
    if tag.name in PROTECTED_CONTENT_TAGS:
        return True

    paragraph_count = len(tag.find_all("p"))
    heading_count = len(tag.find_all(("h1", "h2", "h3")))
    return paragraph_count >= 3 or (paragraph_count >= 1 and heading_count >= 1)


def _prune_html_noise(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        # BeautifulSoup 在父节点被 decompose 后，预先缓存的子节点可能仍会被遍历到，
        # 这时其 attrs 已被清空，继续调用 tag.get 会触发 NoneType 异常。
        if getattr(tag, "attrs", None) is None:
            continue

        attributes = []
        for attr_name in ("id", "class", "role", "aria-label"):
            attr_value = tag.get(attr_name)
            if isinstance(attr_value, list):
                attributes.extend(
                    value for value in attr_value if isinstance(value, str) and value.strip()
                )
            elif isinstance(attr_value, str):
                attributes.append(attr_value)

        if any(_contains_noise_keyword(value) for value in attributes):
            if _looks_like_content_container(tag):
                continue
            tag.decompose()
            continue

        if (
            tag.get("hidden") is not None
            or tag.get("aria-hidden") == "true"
            or _is_visually_hidden(tag)
        ):
            tag.decompose()


def _normalize_heading_text(text: str) -> str:
    text = _normalize_whitespace(text).lower()
    text = re.sub(r"^[^a-z\u4e00-\u9fff]+|[^a-z\u4e00-\u9fff]+$", "", text)
    return text


def _looks_like_toc_block(text: str) -> bool:
    toc_token_count = len(TOC_TOKEN_PATTERN.findall(text))
    if toc_token_count >= 2:
        punctuation_count = len(re.findall(r"[.!?;:,，。！？；：]", text))
        return punctuation_count <= 2

    if re.match(r"^\d+(?:\.\d+)*\s+\S", text):
        return len(text) <= 160 and not re.search(r"[.!?;:,，。！？；：]", text)

    return False


def _is_noise_block(text: str) -> bool:
    if not text:
        return True

    if PURE_REFERENCE_BLOCK_PATTERN.fullmatch(text):
        return True

    if _looks_like_toc_block(text):
        return True

    if text.startswith(("Jump up to:", "Archived from the original", "Retrieved ")):
        return True

    return False


def _extract_blocks(container: Tag) -> list[str]:
    blocks: list[str] = []
    for tag in container.find_all(BLOCK_TAGS):
        text = _normalize_line(tag.get_text(" ", strip=True))
        if not text:
            continue
        if _is_noise_block(text):
            continue
        if tag.name == "li" and len(text) < 24:
            continue
        blocks.append(text)

    deduped: list[str] = []
    for text in blocks:
        if not deduped or deduped[-1] != text:
            deduped.append(text)
    return deduped


def _trim_trailing_reference_sections(blocks: list[str]) -> list[str]:
    for index, block in enumerate(blocks):
        normalized_block = _normalize_heading_text(block)
        if normalized_block not in TAIL_SECTION_TITLES:
            continue

        if index < 2:
            continue

        preceding_blocks = blocks[:index]
        has_meaningful_body = any(
            len(item) >= 40 or re.search(r"[.!?;:,，。！？；：]", item)
            for item in preceding_blocks
        )
        if not has_meaningful_body:
            continue

        # 尾部出现参考文献、外链等章节时，后续内容基本不再属于正文。
        return blocks[:index]

    return blocks


def _trim_leading_blocks_for_structured_article(blocks: list[str]) -> list[str]:
    numbered_heading_indexes = [
        index
        for index, block in enumerate(blocks)
        if re.match(r"^\d+\.\s+\S", block)
    ]
    if len(numbered_heading_indexes) < 5:
        return blocks

    first_numbered_heading_index = numbered_heading_indexes[0]
    if first_numbered_heading_index < 8:
        return blocks

    # 当前导块过多且正文已经呈现出稳定的编号章节结构时，
    # 说明前面大概率混入了公告、导航或营销模块，应从正文首个编号章节开始截断。
    return blocks[first_numbered_heading_index:]


def _score_blocks(blocks: list[str]) -> int:
    if not blocks:
        return 0
    text = "\n".join(blocks)
    punctuation_bonus = len(re.findall(r"[.!?;:,，。！？；：]", text))
    long_block_bonus = sum(1 for block in blocks if len(block) >= 80) * 8
    return len(text) + punctuation_bonus * 6 + long_block_bonus


def _extract_html_text(file_path: Path) -> str:
    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")
    _prune_html_noise(soup)

    candidates: list[Tag] = []
    for selector in CONTENT_SELECTORS:
        candidates.extend(soup.select(selector))

    body = soup.body
    if body is not None:
        candidates.append(body)

    seen: set[int] = set()
    unique_candidates: list[Tag] = []
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id not in seen:
            seen.add(candidate_id)
            unique_candidates.append(candidate)

    best_text = ""
    best_score = -1
    for candidate in unique_candidates:
        blocks = _extract_blocks(candidate)
        blocks = _trim_trailing_reference_sections(blocks)
        blocks = _trim_leading_blocks_for_structured_article(blocks)
        score = _score_blocks(blocks)
        if score > best_score:
            best_score = score
            best_text = "\n\n".join(blocks)

    if not best_text:
        fallback_text = _normalize_whitespace(soup.get_text("\n"))
        return fallback_text

    return _normalize_whitespace(best_text)


def _extract_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
        raise RuntimeError(
            "检测到 PDF 来源，但当前环境缺少 pypdf。请先安装 requirements.txt 中的依赖。"
        ) from exc

    reader = PdfReader(str(file_path))
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = _normalize_whitespace(text)
        if text:
            page_texts.append(text)
    return "\n\n".join(page_texts).strip()


def _extract_text(file_path: Path, source_type: str) -> str:
    normalized_type = source_type.lower()
    if normalized_type == "html":
        return _extract_html_text(file_path)
    if normalized_type == "pdf":
        return _extract_pdf_text(file_path)
    raise ValueError(f"暂不支持的 source_type: {source_type}")


def load_sources(config_path: Path) -> list[SourceRecord]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    items = payload.get("sources", [])
    sources: list[SourceRecord] = []
    for item in items:
        sources.append(
            SourceRecord(
                source_id=item["source_id"],
                title=item["title"],
                tier=int(item["tier"]),
                authority_level=item["authority_level"],
                source_type=item["source_type"],
                original_url=item.get("original_url", ""),
                raw_path=item["raw_path"],
                organization=item.get("organization", ""),
                verification_status=item.get("verification_status", ""),
                notes=item.get("notes", ""),
            )
        )
    return sources


def build_documents(repo_root: Path, config_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = load_sources(config_path)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        raw_file = repo_root / source.raw_path
        if not raw_file.exists():
            errors.append(
                {
                    "source_id": source.source_id,
                    "raw_path": source.raw_path,
                    "error": "原始文件不存在",
                }
            )
            continue

        try:
            clean_text = _extract_text(raw_file, source.source_type)
            if not clean_text:
                raise ValueError("正文提取结果为空")

            documents.append(
                {
                    "doc_id": f"doc_{index:04d}",
                    "source_id": source.source_id,
                    "title": source.title,
                    "tier": source.tier,
                    "authority_level": source.authority_level,
                    "source_type": source.source_type,
                    "original_url": source.original_url,
                    "raw_path": source.raw_path,
                    "organization": source.organization,
                    "verification_status": source.verification_status,
                    "language": _infer_language(clean_text),
                    "clean_text": clean_text,
                    "char_count": len(clean_text),
                    "paragraph_count": len([part for part in clean_text.split("\n\n") if part.strip()]),
                    "processed_at": utc_now_iso(),
                    "notes": source.notes,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 这里需要保留原始错误信息
            errors.append(
                {
                    "source_id": source.source_id,
                    "raw_path": source.raw_path,
                    "error": str(exc),
                }
            )

    return documents, errors


def run_document_preprocess(
    repo_root: Path,
    config_path: Path,
    output_path: Path,
    report_path: Path,
    strict: bool = False,
) -> tuple[int, int]:
    documents, errors = build_documents(repo_root=repo_root, config_path=config_path)

    write_jsonl(output_path, documents)
    write_json(
        report_path,
        {
            "generated_at": utc_now_iso(),
            "document_count": len(documents),
            "error_count": len(errors),
            "errors": errors,
        },
    )

    if strict and errors:
        raise RuntimeError(
            f"文档级预处理存在 {len(errors)} 个错误，请先查看 {report_path.as_posix()}。"
        )

    return len(documents), len(errors)
