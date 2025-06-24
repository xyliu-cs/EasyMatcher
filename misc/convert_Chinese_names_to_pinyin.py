#!/usr/bin/env python3
"""
Normalize full-name strings in an Excel workbook.

Usage:
    python convert_Chinese_names_to_pinyin.py path/to/input.xlsx   # writes input_normalized.xlsx

Dependencies:
    pip install pandas openpyxl pypinyin
"""

import re
import sys
from pathlib import Path

import pandas as pd
from pypinyin import lazy_pinyin, Style

# --------------------------------------------------------------------------- #
#  A small list of the most common two-character Chinese family names.
#  Add more if you encounter new ones.
# --------------------------------------------------------------------------- #
DOUBLE_SURNAME = {
    "欧阳", "司马", "上官", "夏侯", "诸葛", "东方", "皇甫", "呼延",
    "慕容", "尉迟", "羊舌", "赫连", "端木", "拓跋", "长孙", "宇文",
    "司徒", "司空", "南宫", "百里",
}

CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")  # one CJK Unified-Ideograph


def contains_chinese(text: str) -> bool:
    """Return True if *text* includes any CJK character."""
    return bool(CHINESE_CHAR_RE.search(text or ""))


def pinyin_capitalize(pinyin_syllables):
    """Capitalize the first letter of every syllable and join them."""
    return "".join(syl.capitalize() for syl in pinyin_syllables)


def normalize_chinese_name(cn_name: str) -> str:
    """朱明丽  ->  Mingli ZHU"""
    cn_name = cn_name.strip()
    surname_len = 2 if cn_name[:2] in DOUBLE_SURNAME else 1
    surname = cn_name[:surname_len]
    given_name = cn_name[surname_len:]

    surname_py = "".join(lazy_pinyin(surname, style=Style.NORMAL)).upper()
    given_py = "".join(lazy_pinyin(given_name, style=Style.NORMAL)).capitalize()

    return f"{given_py} {surname_py}"


def normalize_english_name(en_name: str) -> str:
    """Kudria Sergei  ->  Kudria SERGEI"""
    parts = re.split(r"\s+", en_name.strip())
    if not parts:
        return en_name
    parts[-1] = parts[-1].upper()
    return " ".join(parts)


def normalize_name(full_name: str) -> str:
    if not isinstance(full_name, str) or not full_name.strip():
        return full_name  # leave NaN or empty cells untouched
    return (
        normalize_chinese_name(full_name)
        if contains_chinese(full_name)
        else normalize_english_name(full_name)
    )


def find_main_sheet(xls: pd.ExcelFile) -> str:
    """Return the first sheet whose name contains 'main' (case-insensitive),
    or fall back to the very first sheet."""
    for name in xls.sheet_names:
        if "main" in name.lower():
            return name
    return xls.sheet_names[0]


def find_full_name_column(df: pd.DataFrame) -> str:
    """Return the column whose header includes 'Full Name' (case-insensitive).
    Raises ValueError if not found."""
    for col in df.columns:
        if "full name" in str(col).lower():
            return col
    raise ValueError("No column containing 'Full Name' found.")


def main(path: str) -> None:
    path = Path(path).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    xls = pd.ExcelFile(path, engine="openpyxl")
    sheet = find_main_sheet(xls)
    df = pd.read_excel(xls, sheet_name=sheet)

    full_name_col = find_full_name_column(df)
    df["Normalized Names"] = df[full_name_col].apply(normalize_name)

    out_path = path.with_stem(f"{path.stem}_normalized")
    df.to_excel(out_path, index=False)
    print(f"✓ Normalized file written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_Chinese_names_to_pinyin.py path/to/input.xlsx", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
