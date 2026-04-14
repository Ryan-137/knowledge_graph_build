from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from src.preprocess.shared import read_json, read_jsonl, write_jsonl
from src.schema.entity import CorefLinkRecord, CorefRuleConfig, EntityMentionRecord


#这个coreference resolution感觉做得太简单了，规则型的匹配，还是需要再调整一下
#只看最近的，还是不行

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COREF_RULES_PATH = REPO_ROOT / "configs" / "entity" / "coref_rules.json"


def _load_mentions(mentions_path: Path) -> list[EntityMentionRecord]:
    return [EntityMentionRecord(**item) for item in read_jsonl(mentions_path)]


def _load_coref_rule_config(config_path: Path) -> CorefRuleConfig:
    """把共指规则集中到配置文件，便于后续扩展更多代词和窗口策略。"""

    return CorefRuleConfig(**read_json(config_path))


def _build_pronoun_pattern(config: CorefRuleConfig) -> re.Pattern[str]:
    pronouns = config.person_pronouns + config.non_person_pronouns
    return re.compile(r"\b(" + "|".join(re.escape(item) for item in pronouns) + r")\b", flags=re.IGNORECASE)


def _is_compatible_pronoun(pronoun: str, mention: EntityMentionRecord, config: CorefRuleConfig) -> bool:
    normalized = pronoun.casefold()
    if normalized in set(config.person_pronouns):
        return mention.entity_type_pred == "Person"
    if normalized in set(config.non_person_pronouns):
        return mention.entity_type_pred != "Person"
    return False


def build_coref_links(
    sentences_path: Path,
    mentions_path: Path,
    coref_rules_path: Path = DEFAULT_COREF_RULES_PATH,
) -> list[CorefLinkRecord]:
    sentences = read_jsonl(sentences_path)
    mentions = _load_mentions(mentions_path)
    coref_config = _load_coref_rule_config(coref_rules_path)
    pronoun_pattern = _build_pronoun_pattern(coref_config)

    sentence_meta = {
        item["sentence_id"]: {
            "doc_id": item["doc_id"],
            "source_id": item["source_id"],
            "sentence_index_in_doc": int(item["sentence_index_in_doc"]),
            "text": item["text"],
            "offset_start": int(item["offset_start"]),
        }
        for item in sentences
    }##方便取用句子距离的信息

    #只在同一个文本找
    mentions_by_doc: dict[str, list[EntityMentionRecord]] = {}
    for mention in mentions:
        mentions_by_doc.setdefault(mention.doc_id, []).append(mention)

    for doc_mentions in mentions_by_doc.values():
        doc_mentions.sort(key=lambda item: (sentence_meta[item.sentence_id]["sentence_index_in_doc"], item.offset_start))

    links: list[CorefLinkRecord] = []
    coref_counter = 0
    pronoun_counter = 0

    for sentence in sentences:
        sentence_id = str(sentence["sentence_id"])
        doc_id = str(sentence["doc_id"])
        source_id = str(sentence["source_id"])
        sentence_text = str(sentence["text"])
        sentence_offset_start = int(sentence["offset_start"])
        sentence_index_in_doc = int(sentence["sentence_index_in_doc"])

        doc_mentions = mentions_by_doc.get(doc_id, [])

        for match in pronoun_pattern.finditer(sentence_text):
            pronoun_text = match.group(0)
            candidates = [
                mention
                for mention in doc_mentions
                if mention.offset_end <= sentence_offset_start + match.start()
                and sentence_index_in_doc - int(sentence_meta[mention.sentence_id]["sentence_index_in_doc"])
                <= coref_config.max_sentence_gap
                and _is_compatible_pronoun(pronoun_text, mention, coref_config)
            ]

            if not candidates:
                continue

            # 规则型共指先走“最近可兼容实体优先”，并故意把窗口限制在最近两句内，减少错连。
            antecedent = candidates[-1]
            antecedent_sentence_index = int(sentence_meta[antecedent.sentence_id]["sentence_index_in_doc"])
            sentence_gap = sentence_index_in_doc - antecedent_sentence_index
            confidence = coref_config.confidence_by_sentence_gap.get(
                str(sentence_gap),
                coref_config.fallback_confidence,
            )

            coref_counter += 1
            pronoun_counter += 1
            links.append(
                CorefLinkRecord(
                    coref_id=f"coref_{coref_counter:06d}",
                    from_mention_id=f"coref_src_{pronoun_counter:06d}",
                    to_mention_id=antecedent.mention_id,
                    doc_id=doc_id,
                    source_id=source_id,
                    sentence_id=sentence_id,
                    pronoun_text=pronoun_text,
                    pronoun_offset_start=sentence_offset_start + match.start(),
                    pronoun_offset_end=sentence_offset_start + match.end(),
                    coref_type="pronoun",
                    confidence=confidence,
                )
            )

    return links


def run_coref_resolution(
    sentences_path: Path,
    mentions_path: Path,
    output_path: Path,
    coref_rules_path: Path = DEFAULT_COREF_RULES_PATH,
) -> int:
    links = build_coref_links(
        sentences_path=sentences_path,
        mentions_path=mentions_path,
        coref_rules_path=coref_rules_path,
    )
    write_jsonl(output_path, [asdict(item) for item in links])
    return len(links)
