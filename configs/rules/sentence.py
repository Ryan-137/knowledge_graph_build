from __future__ import annotations

import re


MONTH_MAP = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "no",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "eq",
    "cf",
    "ca",
    "c",
    "ph.d",
}

SECTION_PREFIX_PATTERN = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?)\s+")
TOC_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")
BLOCK_PATTERN = re.compile(r"\S(?:.*?\S)?(?=(?:\n{2,}|$))", re.S)
LINE_PATTERN = re.compile(r"\S(?:.*?\S)?(?=(?:\n|$))", re.S)
