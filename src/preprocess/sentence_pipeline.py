from __future__ import annotations

import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.rules.sentence import (
    ABBREVIATIONS,
    BLOCK_PATTERN,
    LINE_PATTERN,
    MONTH_MAP,
    SECTION_PREFIX_PATTERN,
    TOC_TOKEN_PATTERN,
)
from src.preprocess.shared import read_jsonl, utc_now_iso, write_json, write_jsonl
from src.schema.document import DocumentRecord
from src.schema.sentence import SentenceRecord, TimeMentionRecord


TRAILING_CLOSERS = "\"'”’)]}"
COMMON_SINGLE_LETTER_WORDS = {"a", "A", "i", "I"}
COMMON_SINGLE_LETTER_SYMBOLS = {"A", "B", "C", "I", "Q", "X", "Y"}
SENTENCE_ENDING_PATTERN = re.compile(r"[.!?;:。！？；：][\"'”’)\]}]*$")
PURE_REFERENCE_SENTENCE_PATTERN = re.compile(r"^(?:\[\s*\d+(?:\s*[-,–]\s*\d+)?\s*\]\s*)+$")
LEADING_REFERENCE_PATTERN = re.compile(
    r"^(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:[A-Z\"'“‘(]|\Z))"
)
TRAILING_REFERENCE_PATTERN = re.compile(
    r"([,.;:!?%)\]}\"'”’])\s*(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:\s|$))"
)


@dataclass(frozen=True)
class NormalizedSegment:
    """保存归一化后的段落文本，以及每个字符对应的原始绝对偏移。"""

    text: str
    offsets: list[int]


def load_documents(documents_path: Path) -> list[DocumentRecord]:
    payload = read_jsonl(documents_path)
    documents: list[DocumentRecord] = []
    for item in payload:
        documents.append(
            DocumentRecord(
                doc_id=item["doc_id"],
                source_id=item["source_id"],
                title=item.get("title", ""),
                tier=int(item.get("tier", 0)),
                language=item.get("language", "unknown"),
                clean_text=item.get("clean_text", ""),
            )
        )
    return documents


def _iter_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    for match in BLOCK_PATTERN.finditer(text):
        blocks.append((match.group(0), match.start(), match.end()))
    return blocks


def _looks_like_toc_block(block_text: str) -> bool:
    section_token_count = len(TOC_TOKEN_PATTERN.findall(block_text))
    if section_token_count < 2:
        return False

    punctuation_count = len(re.findall(r"[.!?;:。！？；：]", block_text))
    return punctuation_count <= 2


def _split_toc_block(block_text: str, block_start: int) -> list[tuple[str, int, int]]:
    matches = list(re.finditer(r"(?:^|\s)(\d+(?:\.\d+)+|\d+)\s+[A-Z]", block_text))
    if len(matches) < 2:
        return [(block_text, block_start, block_start + len(block_text))]

    segments: list[tuple[str, int, int]] = []
    starts = [match.start(1) for match in matches]
    starts.append(len(block_text))
    for index in range(len(starts) - 1):
        segment_start = starts[index]
        segment_end = starts[index + 1]
        segment_text = block_text[segment_start:segment_end].strip()
        if not segment_text:
            continue
        absolute_start = block_start + segment_start
        segments.append((segment_text, absolute_start, absolute_start + len(segment_text)))
    return segments


def _split_block_into_segments(block_text: str, block_start: int) -> list[tuple[str, int, int]]:
    if _looks_like_toc_block(block_text):
        return _split_toc_block(block_text, block_start)

    lines = [(match.group(0), match.start(), match.end()) for match in LINE_PATTERN.finditer(block_text)]
    meaningful_lines = [line for line in lines if line[0].strip()]
    if len(meaningful_lines) <= 1:
        return [(block_text, block_start, block_start + len(block_text))]

    soft_wrap_count = 0
    for index in range(len(meaningful_lines) - 1):
        current_line = meaningful_lines[index][0].strip()
        next_line = meaningful_lines[index + 1][0].strip()
        if not current_line or not next_line:
            continue

        current_tail = current_line[-1]
        next_head = next_line[0]
        if current_tail in ",-–(":
            soft_wrap_count += 1
            continue

        if current_tail not in ".!?;:。！？；：" and next_head.islower():
            soft_wrap_count += 1

    if soft_wrap_count >= max(1, len(meaningful_lines) // 2):
        return [(block_text, block_start, block_start + len(block_text))]

    average_length = sum(len(line[0]) for line in meaningful_lines) / len(meaningful_lines)
    if average_length > 80:
        return [(block_text, block_start, block_start + len(block_text))]

    # PDF 常把“章节标题 + 正文首句”压在同一个 block 里。
    # 这里先把前导标题行剥离出来，避免后面把标题和正文误拼成一条句子。
    leading_heading_count = 0
    has_wrapped_prefix = False
    for index, (line_text, _, relative_end) in enumerate(meaningful_lines):
        if not _looks_like_heading_block(line_text):
            break
        next_line_text = meaningful_lines[index + 1][0] if index + 1 < len(meaningful_lines) else ""
        separator_text = (
            block_text[relative_end : meaningful_lines[index + 1][1]]
            if index + 1 < len(meaningful_lines)
            else ""
        )
        if _looks_like_wrapped_prefix(line_text, next_line_text, separator_text):
            has_wrapped_prefix = True
            break
        leading_heading_count += 1

    if has_wrapped_prefix:
        return [(block_text, block_start, block_start + len(block_text))]

    if 0 < leading_heading_count < len(meaningful_lines):
        segments: list[tuple[str, int, int]] = []
        for line_text, relative_start, relative_end in meaningful_lines[:leading_heading_count]:
            absolute_start = block_start + relative_start
            segments.append((line_text, absolute_start, block_start + relative_end))

        body_start = meaningful_lines[leading_heading_count][1]
        body_end = meaningful_lines[-1][2]
        segments.append((block_text[body_start:body_end], block_start + body_start, block_start + body_end))
        return segments

    segments: list[tuple[str, int, int]] = []
    for line_text, relative_start, relative_end in meaningful_lines:
        absolute_start = block_start + relative_start
        segments.append((line_text, absolute_start, block_start + relative_end))
    return segments


def _looks_like_heading_block(block_text: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", block_text).strip()
    normalized_text = SECTION_PREFIX_PATTERN.sub("", normalized_text).strip()
    if not normalized_text:
        return False

    if len(normalized_text) > 100:
        return False

    if re.search(r"[.!?;:。！？；：]", normalized_text):
        return False

    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff&+'’-]+", normalized_text)
    if not words or len(words) > 12:
        return False

    # 标题通常以短语形态出现，这里故意放宽判断，宁可少吃正文句子。
    return True


def _looks_like_wrapped_prefix(line_text: str, next_line_text: str, separator_text: str) -> bool:
    """
    识别被 PDF 强行断开的句首片段，例如 “Alan\\n Mathison ...”。

    这类文本在视觉上像标题 + 正文，但其实只是一个句子的前缀，不能被当成 heading 拆开。
    """

    if not next_line_text or "\n" not in separator_text:
        return False

    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff&+'’-]+", line_text.strip())
    # 只有换行本身说明是正常新行；换行后仍残留空白，通常是 PDF 强制折行。
    return len(words) == 1 and bool(separator_text.replace("\n", ""))


def _collapse_whitespace_with_offsets(text: str, absolute_start: int) -> NormalizedSegment:
    chars: list[str] = []
    offsets: list[int] = []
    pending_space = False
    pending_space_offset = absolute_start

    for relative_index, char in enumerate(text):
        absolute_index = absolute_start + relative_index
        if char.isspace():
            if chars and not pending_space:
                pending_space = True
                pending_space_offset = absolute_index
            continue

        if pending_space:
            previous_char = chars[-1] if chars else ""
            if previous_char not in "\"'“‘([{" and char not in ",.;:!?%)]}\"'”’":
                chars.append(" ")
                offsets.append(pending_space_offset)
            pending_space = False

        chars.append(char)
        offsets.append(absolute_index)

    return NormalizedSegment(text="".join(chars), offsets=offsets)


def _next_significant_char(text: str, start_index: int) -> str:
    cursor = start_index
    while cursor < len(text):
        char = text[cursor]
        if char.isspace() or char in TRAILING_CLOSERS:
            cursor += 1
            continue
        return char
    return ""


def _last_ascii_token(text: str, end_index: int) -> str:
    window = text[max(0, end_index - 24) : end_index + 1]
    match = re.search(r"([A-Za-z][A-Za-z.\-]{0,23})[.!?;:]?$", window)
    if not match:
        return ""
    return match.group(1).strip(".").lower()


def _consume_trailing_closers(text: str, index: int) -> int:
    cursor = index
    while cursor + 1 < len(text) and text[cursor + 1] in TRAILING_CLOSERS:
        cursor += 1
    return cursor


def _is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in "。！？；!?;":
        return True

    if char != ".":
        return False

    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""

    if previous_char.isdigit() and next_char.isdigit():
        return False

    if previous_char == "." or next_char == ".":
        return False

    token = _last_ascii_token(text, index)
    if token in ABBREVIATIONS:
        return False

    next_significant_char = _next_significant_char(text, index + 1)
    if not next_significant_char:
        return True

    if previous_char.isupper() and next_significant_char.isupper():
        return False

    if re.match(r"\s*[A-Za-z]\.", text[index + 1 :]):
        return False

    if next_significant_char.islower():
        return False

    return True


def _normalize_sentence_text(text: str) -> str:
    text = SECTION_PREFIX_PATTERN.sub("", text).strip()
    text = LEADING_REFERENCE_PATTERN.sub("", text)
    text = TRAILING_REFERENCE_PATTERN.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text)
    # 一些 PDF 提取会把整段引文前后的空格吃掉。
    # 这里按“完整引号块”补空格，避免把 opening / closing quote 分别修坏。
    text = re.sub(r"(?<=\w)(\"[^\"]+\")", r" \1", text)
    text = re.sub(r"(\"[^\"]+\")(?=\w)", r"\1 ", text)
    text = re.sub(r"([,.;:!?])(\"[^\"]+\")", r"\1 \2", text)
    text = re.sub(r"([“‘([{])\s+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?%)\]}”’])", r"\1", text)
    text = re.sub(r"\s+([\"'])(?=\s|$|[,.;:!?%)\]}])", r"\1", text)

    tokens = text.split(" ")
    merged_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        current_token = tokens[index]
        if index + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]", current_token):
            next_token = tokens[index + 1]
            if re.fullmatch(r"[a-z]+", next_token):
                if current_token.islower() and current_token not in COMMON_SINGLE_LETTER_WORDS:
                    merged_tokens.append(current_token + next_token)
                    index += 2
                    continue

                if current_token.isupper() and current_token not in COMMON_SINGLE_LETTER_SYMBOLS:
                    if len(next_token) == 1 or len(next_token) >= 4:
                        merged_tokens.append(current_token + next_token)
                        index += 2
                        continue

        merged_tokens.append(current_token)
        index += 1

    text = " ".join(token for token in merged_tokens if token)
    return text.strip()


def _complete_heading_sentence(text: str) -> str:
    if not text:
        return text

    if SENTENCE_ENDING_PATTERN.search(text):
        return text

    if re.search(r"[\u4e00-\u9fff]", text):
        return f"{text}。"
    return f"{text}."


def _extract_heading_piece(block_text: str, block_start: int, block_end: int) -> tuple[str, int, int] | None:
    left_trim = len(block_text) - len(block_text.lstrip())
    right_trim = len(block_text) - len(block_text.rstrip())
    trimmed_text = block_text.strip()
    if not trimmed_text:
        return None

    absolute_start = block_start + left_trim
    absolute_end = block_end - right_trim

    prefix_match = SECTION_PREFIX_PATTERN.match(trimmed_text)
    if prefix_match:
        prefix_length = prefix_match.end()
        trimmed_text = trimmed_text[prefix_length:]
        absolute_start += prefix_length

    normalized_text = _normalize_sentence_text(trimmed_text)
    if not normalized_text:
        return None

    return normalized_text, absolute_start, absolute_end


def _build_heading_sentence(heading_blocks: list[tuple[str, int, int]]) -> tuple[str, int, int] | None:
    heading_parts: list[str] = []
    absolute_start: int | None = None
    absolute_end: int | None = None

    for block_text, block_start, block_end in heading_blocks:
        heading_piece = _extract_heading_piece(block_text, block_start, block_end)
        if heading_piece is None:
            continue

        piece_text, piece_start, piece_end = heading_piece
        heading_parts.append(piece_text)
        if absolute_start is None:
            absolute_start = piece_start
        absolute_end = piece_end

    if not heading_parts or absolute_start is None or absolute_end is None:
        return None

    heading_text = _normalize_sentence_text(" ".join(heading_parts))
    heading_text = _complete_heading_sentence(heading_text)
    if _is_discardable_sentence(heading_text):
        return None

    return heading_text, absolute_start, absolute_end


def _is_discardable_sentence(text: str) -> bool:
    if not text or len(text) < 4:
        return True

    if PURE_REFERENCE_SENTENCE_PATTERN.fullmatch(text):
        return True

    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text):
        return True

    if re.fullmatch(r"(?:\d+(?:\.\d+)+|\d+)", text):
        return True

    if len(TOC_TOKEN_PATTERN.findall(text)) >= 3 and not re.search(r"[.!?;:。！？；：]", text):
        return True

    if text.startswith(("Retrieved ", "Archived from the original", "^")):
        return True

    if text.startswith(("Author:", "Useful Links", "Related articles", "Other on-line publications")):
        return True

    if len(text) <= 80 and any(marker in text for marker in ("ISBN", "doi", "Vol.", "pp.", "BBC News", "The Guardian")):
        return True

    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if not re.search(r"[.!?;:。！？；：]", text) and words:
        if len(text) <= 60 and len(words) <= 8 and all(word[:1].isupper() for word in words if word):
            return True

    if len(text) <= 36 and ":" in text and not re.search(r"\d{4}", text):
        return True

    if re.fullmatch(r"[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*){1,3}", text):
        return True

    return False


def _split_segment_into_sentences(segment_text: str, segment_start: int) -> list[tuple[str, int, int]]:
    if "\n" in segment_text:
        line_segments = _split_block_into_segments(segment_text, segment_start)
        should_split_lines = len(line_segments) > 1 and any(
            _looks_like_heading_block(line_text) for line_text, _, _ in line_segments[:-1]
        )
        if should_split_lines:
            sentences: list[tuple[str, int, int]] = []
            for line_text, line_start, _ in line_segments:
                sentences.extend(_split_segment_into_sentences(line_text, line_start))
            return sentences

    normalized_segment = _collapse_whitespace_with_offsets(segment_text, segment_start)
    normalized_text = normalized_segment.text
    offsets = normalized_segment.offsets
    if not normalized_text:
        return []

    sentences: list[tuple[str, int, int]] = []
    sentence_start = 0
    index = 0

    while index < len(normalized_text):
        if not _is_sentence_boundary(normalized_text, index):
            index += 1
            continue

        boundary_end = _consume_trailing_closers(normalized_text, index)
        while sentence_start <= boundary_end and normalized_text[sentence_start].isspace():
            sentence_start += 1

        if sentence_start > boundary_end:
            sentence_start = boundary_end + 1
            index = boundary_end + 1
            continue

        sentence_text = _normalize_sentence_text(normalized_text[sentence_start : boundary_end + 1])
        absolute_start = offsets[sentence_start]
        absolute_end = offsets[boundary_end] + 1
        if not _is_discardable_sentence(sentence_text):
            sentences.append((sentence_text, absolute_start, absolute_end))

        sentence_start = boundary_end + 1
        index = boundary_end + 1

    while sentence_start < len(normalized_text) and normalized_text[sentence_start].isspace():
        sentence_start += 1

    if sentence_start < len(normalized_text):
        tail_end = len(normalized_text) - 1
        while tail_end >= sentence_start and normalized_text[tail_end].isspace():
            tail_end -= 1

        if tail_end >= sentence_start:
            sentence_text = _normalize_sentence_text(normalized_text[sentence_start : tail_end + 1])
            absolute_start = offsets[sentence_start]
            absolute_end = offsets[tail_end] + 1
            if _looks_like_heading_block(sentence_text):
                sentence_text = _complete_heading_sentence(sentence_text)
            if not _is_discardable_sentence(sentence_text):
                sentences.append((sentence_text, absolute_start, absolute_end))

    return sentences


def _expand_two_digit_year(start_year: str, end_year: str) -> str:
    if len(end_year) == 4:
        return end_year
    century_prefix = start_year[:2]
    return f"{century_prefix}{end_year}"


def _append_time_mention(
    mentions: list[TimeMentionRecord],
    occupied: list[tuple[int, int]],
    start: int,
    end: int,
    text: str,
    normalized: str,
    mention_type: str,
) -> None:
    for existing_start, existing_end in occupied:
        if not (end <= existing_start or start >= existing_end):
            return

    occupied.append((start, end))
    mentions.append(
        TimeMentionRecord(
            text=text,
            normalized=normalized,
            type=mention_type,
            offset_start=start,
            offset_end=end,
        )
    )


def _extract_time_mentions(sentence_text: str) -> list[TimeMentionRecord]:
    mentions: list[TimeMentionRecord] = []
    occupied: list[tuple[int, int]] = []

    for match in re.finditer(
        r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        day = f"{int(match.group('day')):02d}"
        normalized = f"{match.group('year')}-{month}-{day}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "date",
        )

    for match in re.finditer(
        r"\b(?P<day>\d{1,2})\s+(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        day = f"{int(match.group('day')):02d}"
        normalized = f"{match.group('year')}-{month}-{day}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "date",
        )

    for match in re.finditer(
        r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        normalized = f"{match.group('year')}-{month}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "month_year",
        )

    for match in re.finditer(
        r"\b(?P<start>\d{4})\s*(?:-|–|/|to)\s*(?P<end>\d{2,4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        end_year = _expand_two_digit_year(match.group("start"), match.group("end"))
        normalized = f"{match.group('start')}-{end_year}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "year_range",
        )

    for match in re.finditer(r"\b(?:c\.|ca\.|circa)\s*(?P<year>\d{4})\b", sentence_text, flags=re.IGNORECASE):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group("year"),
            "circa_year",
        )

    for match in re.finditer(r"\b(?P<year>\d{4})s\b", sentence_text):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group(0),
            "decade",
        )

    for match in re.finditer(r"\b(1[5-9]\d{2}|20\d{2})\b", sentence_text):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group(0),
            "year",
        )

    return mentions


def build_sentences(
    documents_path: Path,
) -> tuple[list[SentenceRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    documents = load_documents(documents_path)
    sentences: list[SentenceRecord] = []
    errors: list[dict[str, Any]] = []
    doc_sentence_counts: list[dict[str, Any]] = []
    sentence_counter = 1

    for document in documents:
        if not document.clean_text.strip():
            errors.append(
                {
                    "doc_id": document.doc_id,
                    "source_id": document.source_id,
                    "error": "clean_text 为空，无法进行句子级预处理",
                }
            )
            continue

        sentence_index_in_doc = 1
        previous_sentence_text = ""

        blocks = _iter_blocks(document.clean_text)
        block_index = 0
        while block_index < len(blocks):
            block_text, block_start, _ = blocks[block_index]

            if _looks_like_heading_block(block_text):
                heading_end_index = block_index
                while heading_end_index + 1 < len(blocks) and _looks_like_heading_block(blocks[heading_end_index + 1][0]):
                    heading_end_index += 1

                heading_sentence = _build_heading_sentence(blocks[block_index : heading_end_index + 1])
                if heading_sentence is not None:
                    sentence_text, absolute_start, absolute_end = heading_sentence
                    if sentence_text != previous_sentence_text:
                        time_mentions = _extract_time_mentions(sentence_text)
                        sentences.append(
                            SentenceRecord(
                                sentence_id=f"sent_{sentence_counter:06d}",
                                doc_id=document.doc_id,
                                source_id=document.source_id,
                                sentence_index_in_doc=sentence_index_in_doc,
                                text=sentence_text,
                                offset_start=absolute_start,
                                offset_end=absolute_end,
                                normalized_time=[item.normalized for item in time_mentions],
                                time_mentions=time_mentions,
                            )
                        )
                        previous_sentence_text = sentence_text
                        sentence_counter += 1
                        sentence_index_in_doc += 1

                block_index = heading_end_index + 1
                continue

            for segment_text, segment_start, _ in _split_block_into_segments(block_text, block_start):
                for sentence_text, absolute_start, absolute_end in _split_segment_into_sentences(
                    segment_text,
                    segment_start,
                ):
                    if sentence_text == previous_sentence_text:
                        continue

                    time_mentions = _extract_time_mentions(sentence_text)
                    sentences.append(
                        SentenceRecord(
                            sentence_id=f"sent_{sentence_counter:06d}",
                            doc_id=document.doc_id,
                            source_id=document.source_id,
                            sentence_index_in_doc=sentence_index_in_doc,
                            text=sentence_text,
                            offset_start=absolute_start,
                            offset_end=absolute_end,
                            normalized_time=[item.normalized for item in time_mentions],
                            time_mentions=time_mentions,
                        )
                    )
                    previous_sentence_text = sentence_text
                    sentence_counter += 1
                    sentence_index_in_doc += 1
            block_index += 1

        doc_sentence_count = sentence_index_in_doc - 1
        doc_sentence_counts.append(
            {
                "doc_id": document.doc_id,
                "source_id": document.source_id,
                "sentence_count": doc_sentence_count,
            }
        )

        if doc_sentence_count == 0:
            errors.append(
                {
                    "doc_id": document.doc_id,
                    "source_id": document.source_id,
                    "error": "句子切分结果为空",
                }
            )

    return sentences, doc_sentence_counts, errors


def run_sentence_preprocess(
    documents_path: Path,
    output_path: Path,
    report_path: Path,
    strict: bool = False,
) -> tuple[int, int]:
    sentences, doc_sentence_counts, errors = build_sentences(documents_path=documents_path)
    write_jsonl(output_path, [asdict(sentence) for sentence in sentences])
    write_json(
        report_path,
        {
            "generated_at": utc_now_iso(),
            "document_count": len(doc_sentence_counts),
            "sentence_count": len(sentences),
            "error_count": len(errors),
            "doc_sentence_counts": doc_sentence_counts,
            "errors": errors,
        },
    )

    if strict and errors:
        raise RuntimeError(
            f"句子级预处理存在 {len(errors)} 个错误，请先查看 {report_path.as_posix()}。"
        )

    return len(sentences), len(errors)
