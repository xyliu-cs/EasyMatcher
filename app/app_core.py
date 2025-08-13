# app_core.py
"""
EasyMatcher (Revised, with full logging, guaranteed match, and strict TA caps)

Run:
  pip install streamlit pandas openpyxl scikit-learn
  streamlit run app_core.py
"""

from __future__ import annotations

import re
import time
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

# Optional: scikit-learn for semantic matching
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ======================================================================
# Logging helper
# ======================================================================

class RunLogger:
    """Collects structured logs and exposes them as a DataFrame."""
    def __init__(self):
        self.rows: List[dict] = []

    def log(self, level: str, event: str, course: str = "", student_id: str = "",
            student_name: str = "", detail: str = ""):
        self.rows.append({
            "level": level.upper(),      # INFO/WARN/ERROR
            "event": event,              # e.g., MATCH, UNRECOGNIZED_NAME, SEMANTIC_SCORE
            "course_code": course,
            "student_id": student_id,
            "student_name": student_name,
            "detail": detail,
        })

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


LOG = RunLogger()


# ======================================================================
# Generic helpers
# ======================================================================

def _uploaded_name(obj) -> str:
    return getattr(obj, "name", "uploaded")

def _read_main_or_first(uploaded_file, prefer: List[str] | None = None) -> pd.DataFrame:
    """Prefer a named sheet (by exact or substring), else 'main', else first sheet."""
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    if prefer:
        for want in prefer:
            for s in sheet_names:
                if s == want or want.lower() in s.lower():
                    LOG.log("INFO", "SHEET_SELECTED", detail=f"{_uploaded_name(uploaded_file)} -> {s}")
                    return xl.parse(sheet_name=s)
    name_map = {s.lower(): s for s in sheet_names}
    target_sheet = name_map.get("main", sheet_names[0])
    LOG.log("INFO", "SHEET_SELECTED", detail=f"{_uploaded_name(uploaded_file)} -> {target_sheet}")
    return xl.parse(sheet_name=target_sheet)

def _normalize_header(h: str) -> str:
    """
    Lowercase, collapse whitespace, and strip punctuation for resilient matching.
    Makes 'Course instructor's name' ~ 'course instructors name' ~ 'Course instructor's name'.
    """
    s = re.sub(r"\s+", " ", str(h).strip().lower())
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_col(df: pd.DataFrame, substr: str,
              prefer_exact: bool = True,
              disallow_prefixes: tuple[str, ...] = ("unnamed:",),
              exclude_contains: tuple[str, ...] = ()) -> str | None:
    """
    Find a column by name (normalized). Prefer exact, then substring; skip 'Unnamed:*' and excluded tokens.
    """
    target_norm = _normalize_header(substr)
    cols = [(c, _normalize_header(c)) for c in df.columns]

    def allowed(orig: str, norm: str) -> bool:
        lo = str(orig).lower()
        if any(lo.startswith(p) for p in disallow_prefixes):
            return False
        if exclude_contains and any(tok.lower() in lo for tok in exclude_contains):
            return False
        return True

    if prefer_exact:
        for orig, norm in cols:
            if norm == target_norm and allowed(orig, norm):
                return orig
    for orig, norm in cols:
        if target_norm in norm and allowed(orig, norm):
            return orig
    return None

def _get_id(raw_id: str | float | int) -> str:
    """Convert raw student ID to a string, removing trailing .0."""
    if pd.isna(raw_id):
        return "unknown"
    if isinstance(raw_id, float) and float(raw_id).is_integer():
        return str(int(raw_id))
    return str(raw_id).strip()

def _canonicalize_name(raw: str | float | int) -> str | None:
    """Canonical EN name 'Firstname LASTNAME'. Returns None if 'None.' or empty."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower().startswith("none."):
        return None
    s = re.sub(r"^[:,;\s]+|[:,;\s]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if len(parts) < 2:
        return parts[0].title()
    if parts[0].isupper() and (len(parts) > 1 and not parts[1].isupper()):
        lastname, firstname_parts = parts[0], parts[1:]
    else:
        firstname_parts, lastname = parts[:-1], parts[-1]
    firstname = " ".join(p.title() for p in firstname_parts)
    lastname = lastname.upper()
    return f"{firstname} {lastname}"

def _strip_brackets_all(s: str) -> str:
    if not s:
        return s
    return re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|《[^》]*》", " ", s)

def _normalize_delimiters(s: str) -> str:
    if not s:
        return s
    s = (s.replace("、", ",")
           .replace("，", ",")
           .replace("；", ",")
           .replace("．", ".")
           .replace("：", ":"))
    s = re.sub(r"\s*(/|&|\+)\s*", ",", s)
    s = re.sub(r"\b(?:and|AND)\b", ",", s)
    s = s.replace(";", ",")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r"\s*,\s*", ",", s)
    return s.strip(" ,")

def _strip_trailing_symbols(s: str) -> str:
    if not s:
        return s
    return re.sub(r"[.,;:，；。、．]+$", "", s).strip()

def _parse_name_list(raw: str | float | int) -> List[str]:
    if pd.isna(raw):
        return []
    cleaned = _strip_brackets_all(str(raw))
    cleaned = _normalize_delimiters(cleaned)
    if not cleaned:
        return []
    names = cleaned.split(",")
    out: List[str] = []
    for n in names:
        n = _strip_trailing_symbols(n.strip())
        if not n:
            continue
        c = _canonicalize_name(n)
        if c:
            out.append(c)
    return list(dict.fromkeys(out))

def _parse_instructor_pref_cell(cell: str | float | int) -> Tuple[List[str], str]:
    if pd.isna(cell):
        return [], ""
    s = str(cell).strip()
    if s.lower().startswith("none."):
        return [], s[5:].strip()
    return _parse_name_list(s), ""

def _clean_token(s: str | float | int) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\*.*?$", "", s)
    s = re.sub(r"[()（）•·，、]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _split_instructors(raw: str | float | int) -> List[str]:
    if pd.isna(raw):
        return []
    s = str(raw).replace("\r", "\n")
    parts: List[str] = []
    for line in s.split("\n"):
        line = _clean_token(line)
        if not line:
            continue
        for p in re.split(r"[;,/]| & | and ", line):
            p = _clean_token(p)
            if p:
                parts.append(p)
    return list(dict.fromkeys(parts))

def _canon_eng_name(s: str | float | int) -> str:
    return _canonicalize_name(s) or ""

def _canon_chn_name(s: str | float | int) -> str:
    return _clean_token(s)

def _parse_int(val) -> Optional[int]:
    """Return int if val is numeric, rounding decimals to nearest integer; else None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        num = float(str(val).strip())
        return int(round(num))
    except (ValueError, TypeError):
        return None

def _count_assigned_including_leading(meta: dict) -> int:
    return sum(len(meta.get(k, [])) for k in ("leading_ta","ta_supervisor","ta_instr_only","ta_student_only"))

def _quota_remaining(meta: dict) -> int:
    """Remaining slots for this course (leading + all TA buckets count)."""
    q = _parse_int(meta.get("ta_slots", ""))
    if q is None or q < 0:
        q = 0
    assigned = _count_assigned_including_leading(meta)
    return max(0, q - assigned)

def _remaining_slots_map(courses: Dict[str, dict]) -> Dict[str, int]:
    """code -> remaining slots (non-negative)."""
    return {c: _quota_remaining(m) for c, m in courses.items()}

def _eligible_codes_by_priority(courses: Dict[str, dict], rem: Dict[str, int]) -> list[str]:
    """Course codes with remaining capacity, sorted by (type priority, code)."""
    return sorted(
        [c for c in courses if rem.get(c, 0) > 0],
        key=lambda c: (_course_rank(courses[c].get("type", "")), c)
    )


# ======================================================================
# Loaders
# ======================================================================

def load_teaching_assignment(df: pd.DataFrame) -> Dict[str, dict]:
    code_col = _find_col(df, "Course Code")
    title_col = _find_col(df, "Course Title") or _find_col(df, "Course Description")
    desc_col = _find_col(df, "Course Description(ENG)") or _find_col(df, "Course Description")
    inst_col = _find_col(df, "Course Instructors")
    type_col = _find_col(df, "Course Type")
    ta_num_col = _find_col(df, "TA number") or _find_col(df, "TA Number")
    ustf_num_col = _find_col(df, "USTF number") or _find_col(df, "USTF Number")

    courses: Dict[str, dict] = {}
    for _, row in df.iterrows():
        raw_code = row[code_col] if code_col else ""
        code = str(raw_code).replace(" ", "").strip()
        if not code:
            continue
        instructors = _split_instructors(row[inst_col] if inst_col else "")
        inst_set = set(filter(None, instructors + [_canon_eng_name(x) for x in instructors] + [_canon_chn_name(x) for x in instructors]))
        courses[code] = {
            "title": str(row[title_col]).strip() if title_col else "",
            "desc": str(row[desc_col]).strip() if (desc_col and not pd.isna(row[desc_col])) else "",
            "instructors_raw": instructors,
            "instructors_set": inst_set,
            "type": (str(row[type_col]).strip() if type_col else "").upper(),
            "ta_slots": row[ta_num_col] if ta_num_col else "",
            "ustf_slots": row[ustf_num_col] if ustf_num_col else "",
            # buckets
            "leading_ta": [],
            "ta_supervisor": [],
            "ta_instr_only": [],
            "ta_student_only": [],
            "ustf_instr": [],
            "ustf_ta": [],
            "ta_comments": [],
            "_pref_names": [],
            "_lead_names": [],
        }
    LOG.log("INFO", "COURSE_COUNT", detail=str(len(courses)))
    return courses

def load_ta_name_list(df: pd.DataFrame):
    """
    From the TA Name List file.
    Return:
      - id_to_profile: sid -> {
            name_zh, name_en, email, cohort, major, sup_en, sup_zh, field
        }
      - name_to_id: both EN canonical + CN name -> sid
    """
    zh_col = _find_col(df, "Name", prefer_exact=True, disallow_prefixes=("unnamed:",)) or \
             _find_col(df, "姓名", prefer_exact=True, disallow_prefixes=("unnamed:",))
    en_col = _find_col(df, "English Name", prefer_exact=True)
    sid_col = _find_col(df, "ID", prefer_exact=True) or _find_col(df, "Student ID", prefer_exact=True)
    email_col = _find_col(df, "Email", prefer_exact=False)
    cohort_col = _find_col(df, "Cohort", prefer_exact=True)
    major_col = _find_col(df, "Major/Fields", prefer_exact=True)
    sup_zh_col = _find_col(df, "Main Supervisor（CHI）", prefer_exact=True) or _find_col(df, "Main Supervisor (CHI)", prefer_exact=True)
    sup_en_col = _find_col(df, "Main Supervisor", prefer_exact=True,
                           exclude_contains=("（chi", "(chi", "chi）", "chi)"))
    field_col = _find_col(df, "官网Research Field", prefer_exact=False) or _find_col(df, "Research Field", prefer_exact=False)

    LOG.log("INFO", "NAME_LIST_COLUMNS",
            detail=f"ZH={zh_col} | EN={en_col} | ID={sid_col} | Email={email_col} | Cohort={cohort_col} | "
                   f"Major={major_col} | SupEN={sup_en_col} | SupZH={sup_zh_col} | Field={field_col}")

    id_to_profile: Dict[str, dict] = {}
    name_to_id: Dict[str, str] = {}

    for _, r in df.iterrows():
        sid = _get_id(r[sid_col]) if sid_col else "unknown"
        if sid == "unknown":
            LOG.log("WARN", "MISSING_STUDENT_ID", detail=str(r.to_dict()))
            continue

        name_zh = _canon_chn_name(r[zh_col]) if zh_col else ""
        name_en = _canon_eng_name(r[en_col]) if en_col else ""

        email = (str(r[email_col]).strip() if (email_col and not pd.isna(r[email_col])) else "")
        cohort = _get_id(r[cohort_col]) if cohort_col else ""
        major = (str(r[major_col]).strip() if (major_col and not pd.isna(r[major_col])) else "")
        sup_en = _clean_token(r[sup_en_col]) if sup_en_col else ""
        sup_zh = _clean_token(r[sup_zh_col]) if sup_zh_col else ""
        field = (str(r[field_col]).strip() if (field_col and not pd.isna(r[field_col])) else "")

        id_to_profile[sid] = {
            "name_zh": name_zh,
            "name_en": name_en,
            "email": email,
            "cohort": cohort,
            "major": major,
            "sup_en": sup_en,
            "sup_zh": sup_zh,
            "field": field,
        }

        if name_en:
            name_to_id[name_en] = sid
        if name_zh:
            name_to_id[name_zh] = sid

    LOG.log("INFO", "TA_NAME_LIST_SIZE", detail=str(len(id_to_profile)))
    return id_to_profile, name_to_id

def load_instructor_sheet(df: pd.DataFrame) -> Dict[str, dict]:
    course_code_col = _find_col(df, "Course code") or _find_col(df, "Course Code")
    instructor_col = _find_col(df, "Course instructor") or _find_col(df, "Course instructor's name") or _find_col(df, "Course instructors name")
    pref_ta_col = _find_col(df, "Instructor preference for corresponding TA")
    pref_lead_col = _find_col(df, "Instructor preference for Leading TA")
    ustf_col = _find_col(df, "USTF Number Needed") or _find_col(df, "USTF")
    remarks_col = _find_col(df, "Other remarks") or _find_col(df, "Remarks")

    courses: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code = str(row[course_code_col]).replace(" ", "").strip() if course_code_col else ""
        if not code:
            continue
        pref_names, req_comment = _parse_instructor_pref_cell(row[pref_ta_col] if pref_ta_col else "")
        lead_names, _ = _parse_instructor_pref_cell(row[pref_lead_col] if pref_lead_col else "")
        pref_names = [name for name in pref_names if name not in lead_names]
        # USTF parsing: numbers in [a,b], names inside (...)
        m = re.search(r"\[(\d+)\s*,\s*(\d+)\]", str(row[ustf_col])) if ustf_col else None
        names_part = re.search(r"\((.*?)\)", str(row[ustf_col])) if ustf_col else None
        ustf_names = _parse_name_list(names_part.group(1)) if names_part else []
        # comments
        comment_parts = []
        if req_comment:
            comment_parts.append(req_comment)
        raw_remark = row[remarks_col] if remarks_col else ""
        if not pd.isna(raw_remark):
            s = str(raw_remark).strip()
            if s:
                comment_parts.append(s)
        instr_comments = ". ".join(comment_parts)

        courses[code] = {
            "instructor": str(row[instructor_col]).strip() if instructor_col else "",
            "pref_names": pref_names,
            "lead_names": lead_names,
            "ustf_instr": ustf_names,
            "instr_comments": instr_comments,
            # placeholders to be filled later
            "leading_ta": [],
            "ta_supervisor": [],
            "ta_instr_only": [],
            "ta_student_only": [],
            "ustf_ta": [],
            "ta_comments": [],
            "_pref_names": pref_names,
            "_lead_names": lead_names,
        }
    LOG.log("INFO", "INSTR_PREF_COURSE_COUNT", detail=str(len(courses)))
    return courses

def load_student_prefs(df: pd.DataFrame) -> Dict[str, dict]:
    # name_col is optional here (used only for debug / expansion)
    _ = _find_col(df, "Full Name") or _find_col(df, "Normalized Name")
    sid_col  = _find_col(df, "Student ID")
    other_col = _find_col(df, "Other information")
    ustf_col = _find_col(df, "USTF Remark")

    pri_cols = list(filter(None, [
        _find_col(df, "First Priority"),
        _find_col(df, "Second Priority"),
        _find_col(df, "Third Priority"),
        _find_col(df, "Fourth Priority"),
    ]))

    out: Dict[str, dict] = {}
    for _, r in df.iterrows():
        sid = _get_id(r[sid_col]) if sid_col else "unknown"
        if not sid or sid == "unknown":
            LOG.log("WARN", "STUDENT_PREF_NO_ID", detail=str(r.to_dict()))
            continue
        prefs = []
        for c in pri_cols:
            if c is None:
                continue
            val = r[c]
            if not pd.isna(val) and str(val).strip():
                prefs.append(str(val).replace(" ", "").strip())
        ustf_remark = _parse_name_list(r[ustf_col] if ustf_col else "")
        other = "" if (not other_col or pd.isna(r[other_col])) else str(r[other_col]).strip()
        out[sid] = {"prefs": prefs, "ustf_remark": ustf_remark, "other": other}
    LOG.log("INFO", "STUDENT_PREF_COUNT", detail=str(len(out)))
    return out

def build_student_map(id_to_profile: Dict[str, dict], pref_map: Dict[str, dict]):
    student_map: Dict[str, dict] = {}
    name_to_id: Dict[str, str] = {}
    for sid, prof in id_to_profile.items():
        prefs = pref_map.get(sid, {}).get("prefs", [])
        ustf_remark = pref_map.get(sid, {}).get("ustf_remark", [])
        other = pref_map.get(sid, {}).get("other", "")
        display_name = prof["name_en"] or prof["name_zh"] or sid

        student_map[sid] = {
            "name": display_name,
            "name_zh": prof["name_zh"],
            "name_en": prof["name_en"],
            "email": prof["email"],
            "prefs": prefs,
            "major": prof["major"],
            "cohort": prof["cohort"],
            "supervisor_en": prof["sup_en"],
            "supervisor_zh": prof["sup_zh"],
            "field": prof["field"],
            "ustf_remark": ustf_remark,
            "other": other,
        }
        if prof["name_en"]:
            name_to_id[prof["name_en"]] = sid
        if prof["name_zh"]:
            name_to_id[prof["name_zh"]] = sid
    LOG.log("INFO", "STUDENT_MAP_SIZE", detail=str(len(student_map)))
    return student_map, name_to_id


# ======================================================================
# Matching logic (revised)
# ======================================================================

def _course_rank(t: str) -> int:
    t = (t or "").upper()
    if t == "PG":
        return 0
    if t == "UG":
        return 1
    if t == "TPG":
        return 2
    return 3

def perform_matching(courses_from_instr: Dict[str, dict],
                     courses_from_assign: Dict[str, dict],
                     student_map: Dict[str, dict],
                     name_to_id: Dict[str, str]) -> set[str]:
    """
    Priority:
      (1) Supervisor preference for own student
      (2) Instructor preference (remaining)
      (3) Course priority PG→UG→TPG
      (4) Student preferences
    """
    matched: set[str] = set()
    courses = courses_from_assign  # alias

    # Merge instructor prefs into assignment base & log unknown names
    for code, meta in courses.items():
        if code in courses_from_instr:
            pref_meta = courses_from_instr[code]
            meta["_pref_names"] = pref_meta.get("pref_names", [])
            meta["_lead_names"] = pref_meta.get("lead_names", [])
            meta["ustf_instr"] = [(n, None) for n in pref_meta.get("ustf_instr", [])]
            meta["instr_comments"] = pref_meta.get("instr_comments", "")
            for n in meta["_pref_names"] + meta["_lead_names"]:
                if n not in name_to_id:
                    LOG.log("WARN", "UNRECOGNIZED_NAME", course=code, detail=f"Instructor listed '{n}' not found in TA Name List")
        else:
            meta["_pref_names"] = []
            meta["_lead_names"] = []
            meta.setdefault("instr_comments", "")

    # Leading TA (reserve by name if resolvable)
    for code, meta in courses.items():
        leading_pairs: List[Tuple[str, Optional[str]]] = []
        for lead in meta.get("_lead_names", []):
            sid = name_to_id.get(lead)
            leading_pairs.append((lead, sid))
            if sid:
                matched.add(sid)
                LOG.log("INFO", "MATCH", course=code, student_id=sid, student_name=lead, detail="Leading TA (instructor)")
            else:
                LOG.log("WARN", "LEADING_TA_NO_ID", course=code, student_name=lead)
        meta["leading_ta"] = leading_pairs

    # (1) Supervisor preference
    for code, meta in courses.items():
        inst_set = meta.get("instructors_set", set())
        for pref_name in meta.get("_pref_names", []):
            sid = name_to_id.get(pref_name)
            if not sid or sid in matched:
                continue
            s = student_map.get(sid, {})
            sup_names = set(x for x in [
                _canon_eng_name(s.get("supervisor_en","")),
                _canon_chn_name(s.get("supervisor_zh","")),
                s.get("supervisor_en",""), s.get("supervisor_zh","")
            ] if x)
            if inst_set & sup_names:
                meta["ta_supervisor"].append((pref_name, sid))
                matched.add(sid)
                LOG.log("INFO", "MATCH", course=code, student_id=sid, student_name=pref_name,
                        detail="Supervisor preference (instructor=supervisor)")

    # (2) Instructor preference (remaining)
    for code, meta in courses.items():
        for pref_name in meta.get("_pref_names", []):
            sid = name_to_id.get(pref_name)
            if not sid or sid in matched:
                continue
            meta["ta_instr_only"].append((pref_name, sid))
            matched.add(sid)
            LOG.log("INFO", "MATCH", course=code, student_id=sid, student_name=pref_name,
                    detail="Instructor preference")

    # (3) Student preferences in course-priority order
    ordered_codes = sorted(courses.keys(), key=lambda c: (_course_rank(courses[c].get("type","")), c))
    for code in ordered_codes:
        meta = courses[code]
        for sid, sinfo in student_map.items():
            if sid in matched:
                continue
            if code in sinfo.get("prefs", []):
                meta["ta_student_only"].append((sinfo["name"], sid))
                matched.add(sid)
                LOG.log("INFO", "MATCH", course=code, student_id=sid, student_name=sinfo["name"],
                        detail="Student preference + course priority")

    return matched


# ======================================================================
# Semantic & fallbacks (guaranteed completion)
# ======================================================================

def _semantic_assign_unmatched(courses: Dict[str, dict], student_map: Dict[str, dict], matched: set[str]):
    """TF-IDF cosine(student background vs course text), respecting remaining capacity."""
    if not _HAS_SKLEARN:
        LOG.log("WARN", "SEMANTIC_DISABLED", detail="scikit-learn not installed")
        return

    course_codes = list(courses.keys())
    course_texts = [f"{courses[c].get('title','')} {courses[c].get('desc','')}".strip() for c in course_codes]
    if not any(t for t in course_texts):
        LOG.log("WARN", "SEMANTIC_SKIPPED", detail="Course texts are empty")
        return

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    Xc = vectorizer.fit_transform(course_texts)
    code_to_idx = {c: i for i, c in enumerate(course_codes)}

    rem = _remaining_slots_map(courses)

    for sid, s in student_map.items():
        if sid in matched:
            continue
        elig = _eligible_codes_by_priority(courses, rem)
        if not elig:
            break
        stxt = " ".join(filter(None, [s.get("major",""), s.get("field",""),
                                      s.get("supervisor_en",""), s.get("supervisor_zh","")])).strip()
        if not stxt:
            LOG.log("WARN", "NO_BACKGROUND_TEXT", student_id=sid, student_name=s.get("name",""))
            continue
        xs = vectorizer.transform([stxt])
        sims = cosine_similarity(xs, Xc).ravel()
        best_code, best_score = None, -1.0
        for code in elig:
            sc = float(sims[code_to_idx[code]])
            if sc > best_score:
                best_score, best_code = sc, code
        if best_code is not None and rem.get(best_code, 0) > 0:
            courses[best_code]["ta_student_only"].append((s.get("name", sid), sid))
            matched.add(sid)
            rem[best_code] -= 1
            LOG.log("INFO", "MATCH", course=best_code, student_id=sid,
                    student_name=s.get("name",""),
                    detail=f"Semantic background match (score={best_score:.4f})")

def _keyword_fallback_assign(courses: Dict[str, dict], student_map: Dict[str, dict], matched: set[str]):
    """Cheap keyword overlap fallback, respecting remaining capacity."""
    rem = _remaining_slots_map(courses)
    def ordered_eligible():
        return _eligible_codes_by_priority(courses, rem)
    for sid, s in student_map.items():
        if sid in matched:
            continue
        elig = ordered_eligible()
        if not elig:
            break
        tokens = set(re.findall(r"[A-Za-z]+", " ".join([
            s.get("major",""), s.get("field",""), s.get("supervisor_en",""), s.get("supervisor_zh","")
        ]).lower()))
        best_code, best_hit = None, -1
        for code in elig:
            text = (courses[code].get("title","") + " " + courses[code].get("desc","")).lower()
            hit = sum(1 for t in tokens if t and t in text)
            if hit > best_hit:
                best_hit, best_code = hit, code
        if best_code is not None and rem.get(best_code, 0) > 0:
            courses[best_code]["ta_student_only"].append((s.get("name", sid), sid))
            matched.add(sid)
            rem[best_code] -= 1
            LOG.log("INFO", "MATCH", course=best_code, student_id=sid,
                    student_name=s.get("name",""),
                    detail=f"Keyword fallback (hits={best_hit})")

def _balanced_fill_any_remaining(courses: Dict[str, dict], student_map: Dict[str, dict], matched: set[str]):
    """As a last resort, place remaining students by load-balance into courses with capacity."""
    rem = _remaining_slots_map(courses)
    def eligible():
        return _eligible_codes_by_priority(courses, rem)
    def count_tas(meta: dict) -> int:
        return sum(len(meta.get(k, [])) for k in
                   ("leading_ta","ta_supervisor","ta_instr_only","ta_student_only"))
    for sid, s in student_map.items():
        if sid in matched:
            continue
        elig = eligible()
        if not elig:
            break
        sup_names = set(x for x in [
            _canon_eng_name(s.get("supervisor_en","")),
            _canon_chn_name(s.get("supervisor_zh","")),
            s.get("supervisor_en",""), s.get("supervisor_zh","")
        ] if x)
        elig_sup = [c for c in elig if courses[c].get("instructors_set", set()) & sup_names]
        pool = elig_sup or elig
        best_code = min(pool, key=lambda c: (count_tas(courses[c]),
                                             _course_rank(courses[c].get("type","")), c))
        if rem.get(best_code, 0) > 0:
            courses[best_code]["ta_student_only"].append((s.get("name", sid), sid))
            matched.add(sid)
            rem[best_code] -= 1
            LOG.log("INFO", "MATCH", course=best_code, student_id=sid,
                    student_name=s.get("name",""),
                    detail="Balanced final assignment")

def semantic_and_fallback_fill(courses: Dict[str, dict], student_map: Dict[str, dict], matched: set[str]):
    _semantic_assign_unmatched(courses, student_map, matched)
    _keyword_fallback_assign(courses, student_map, matched)
    _balanced_fill_any_remaining(courses, student_map, matched)


# ======================================================================
# NEW: Strict per-course TA cap enforcement (based on Teaching Assignment)
# ======================================================================

def _unmatched_count(student_map: Dict[str, dict], matched_sids: set[str]) -> int:
    return sum(1 for sid in student_map if sid not in matched_sids)

def _total_open_slots(courses_assign: Dict[str, dict]) -> int:
    total = 0
    for _, meta in courses_assign.items():
        raw = meta.get("ta_slots", "")
        q = _parse_int(raw)
        # Treat non-integer / negative as 0 (hard cap)
        if q is None or q < 0:
            q = 0
        assigned = sum(len(meta.get(k, [])) for k in ("leading_ta", "ta_supervisor", "ta_instr_only", "ta_student_only"))
        total += max(0, q - assigned)
    return total

def enforce_ta_caps(courses: Dict[str, dict], student_map: Dict[str, dict], matched: set[str]) -> List[str]:
    """
    Strictly enforce per-course TA caps from Teaching Assignment ("ta_slots").
    - Leading TAs are counted toward the cap.
    - Any non-integer/blank TA number is treated as 0.
    - On overflow, we keep in this priority order: leading_ta -> ta_supervisor -> ta_instr_only -> ta_student_only.
    - Return the list of pruned student IDs (to be rematched globally).
    """
    pruned_sids: List[str] = []

    # Process courses in consistent priority order
    ordered_codes = sorted(courses.keys(), key=lambda c: (_course_rank(courses[c].get("type","")), c))

    for code in ordered_codes:
        meta = courses[code]
        raw_slots = meta.get("ta_slots", "")
        quota = _parse_int(raw_slots)
        if quota is None:
            quota = 0
            LOG.log("WARN", "TA_SLOTS_NON_NUMERIC_TREATED_AS_ZERO", course=code, detail=f"value='{raw_slots}' -> cap=0")
        if quota < 0:
            quota = 0

        # Flatten all TAs that count toward the cap (leading included)
        leading = list(meta.get("leading_ta", []))
        sup    = list(meta.get("ta_supervisor", []))
        instr  = list(meta.get("ta_instr_only", []))
        stud   = list(meta.get("ta_student_only", []))

        flat_all: List[Tuple[str, Optional[str], str]] = \
            [(n,sid,"leading_ta") for (n,sid) in leading] + \
            [(n,sid,"ta_supervisor") for (n,sid) in sup] + \
            [(n,sid,"ta_instr_only") for (n,sid) in instr] + \
            [(n,sid,"ta_student_only") for (n,sid) in stud]

        cur_n = len(flat_all)
        if cur_n <= quota:
            # Under / equal to quota: nothing to prune
            continue

        # Keep the first `quota` items by the priority above; prune the rest
        keep = flat_all[:quota]
        drop = flat_all[quota:]

        # Rebuild buckets from kept list
        leading_k, sup_k, instr_k, stud_k = [], [], [], []
        for name, sid, cat in keep:
            if cat == "leading_ta":
                leading_k.append((name, sid))
            elif cat == "ta_supervisor":
                sup_k.append((name, sid))
            elif cat == "ta_instr_only":
                instr_k.append((name, sid))
            else:
                stud_k.append((name, sid))

        meta["leading_ta"]     = leading_k
        meta["ta_supervisor"]  = sup_k
        meta["ta_instr_only"]  = instr_k
        meta["ta_student_only"]= stud_k

        # Mark pruned students as unmatched (if not present elsewhere)
        for name, sid, _ in drop:
            if not sid:
                continue
            present_elsewhere = False
            for c2, m2 in courses.items():
                if c2 == code:
                    continue
                for cat2 in ("leading_ta","ta_supervisor","ta_instr_only","ta_student_only"):
                    if (name, sid) in m2.get(cat2, []):
                        present_elsewhere = True
                        break
                if present_elsewhere:
                    break
            if not present_elsewhere and sid in matched:
                matched.remove(sid)
            pruned_sids.append(sid)
            LOG.log("INFO", "TA_OVERFLOW_PRUNE", course=code, student_id=sid, student_name=name,
                    detail=f"Pruned due to cap {quota}")

    return pruned_sids

def iterative_match_until_stable(
    courses_assign: Dict[str, dict],
    student_map: Dict[str, dict],
    matched_sids: set[str],
    max_passes: int = 12
) -> None:
    """
    Make matching globally exhaustive: whenever caps prune students,
    rematch them into remaining capacity, looping until stable.
    Uses existing `semantic_and_fallback_fill` and `enforce_ta_caps`.
    """
    prev_unmatched = None

    for _ in range(max_passes):
        # 1) ensure current state honors caps (may prune)
        enforce_ta_caps(courses_assign, student_map, matched_sids)

        # 2) fill any unmatched students into remaining capacity
        semantic_and_fallback_fill(courses_assign, student_map, matched_sids)

        # 3) re-enforce caps in case fill overshot any course
        enforce_ta_caps(courses_assign, student_map, matched_sids)

        # 4) check convergence / capacity
        cur_unmatched = _unmatched_count(student_map, matched_sids)
        open_slots = _total_open_slots(courses_assign)

        # Done if everyone placed or no capacity left anywhere
        if cur_unmatched == 0 or open_slots == 0:
            break

        # Stop if no improvement over last pass (stable)
        if prev_unmatched is not None and cur_unmatched >= prev_unmatched:
            break

        prev_unmatched = cur_unmatched


# ======================================================================
# Output builders
# ======================================================================

def build_assignment_dataframe(courses: Dict[str, dict],
                               id_to_profile: Dict[str, dict],
                               original_assign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build final assignment dataframe, preserving ALL columns from the uploaded Teaching Assignment.
    One TA per row, duplicating original row data only on the FIRST row for each course;
    subsequent rows leave original columns blank to avoid repetition.
    """
    def resolve_one(pair: Tuple[str, Optional[str]]) -> Tuple[str, str, str]:
        name, sid = pair
        if sid and sid in id_to_profile:
            prof = id_to_profile[sid]
            zh = prof.get("name_zh") or name
            en = prof.get("name_en") or name
            em = prof.get("email") or ""
            return zh, en, em
        return name, name, ""

    # Locate Course Code column to align rows
    code_col = _find_col(original_assign_df, "Course Code")
    if not code_col:
        raise ValueError("Course Code column not found in original Teaching Assignment file.")

    # TA-related columns we add/modify (not part of 'original columns' dedup rule)
    ta_cols = [
        "TA Name", "Leading TA", "TA ENG Name", "TA Email",
        "USTF Name", "USTF ENG Name", "USTF Email",
        "Comments (TA)", "Comments (Instructor)"
    ]

    rows = []
    for _, orig_row in original_assign_df.iterrows():
        code = str(orig_row[code_col]).replace(" ", "").strip()
        if code not in courses:
            continue

        m = courses[code]
        pairs_leading = list(m.get("leading_ta", []))
        pairs_keep = list(m.get("ta_supervisor", [])) + list(m.get("ta_instr_only", [])) + list(m.get("ta_student_only", []))

        # Precompute aggregated USTF text once per course
        ustf_pairs = list(m.get("ustf_instr", [])) + list(m.get("ustf_ta", []))
        ustf_zh_list, ustf_en_list, ustf_em_list = [], [], []
        for p in ustf_pairs:
            zh_u, en_u, em_u = resolve_one(p)
            ustf_zh_list.append(zh_u); ustf_en_list.append(en_u); ustf_em_list.append(em_u)
        ustf_zh = ", ".join(filter(None, ustf_zh_list))
        ustf_en = ", ".join(filter(None, ustf_en_list))
        ustf_em = ", ".join(filter(None, ustf_em_list))

        made_any_row = False
        is_first_for_course = True

        # Helper to create a row, blanking original columns if not first
        def make_row(is_leading_flag: bool, pair: Tuple[str, Optional[str]] | None):
            nonlocal is_first_for_course, made_any_row
            new_row = orig_row.copy()

            # If not the first row for this course, blank original columns to avoid repetition
            if not is_first_for_course:
                for col in original_assign_df.columns:
                    new_row[col] = ""

            if pair is None:
                # No TA case for this course
                new_row["TA Name"] = ""
                new_row["Leading TA"] = ""
                new_row["TA ENG Name"] = ""
                new_row["TA Email"] = ""
            else:
                zh, en, em = resolve_one(pair)
                new_row["TA Name"] = zh
                new_row["Leading TA"] = zh if is_leading_flag else ""
                new_row["TA ENG Name"] = en
                new_row["TA Email"] = em

            # Always fill (course-level) aggregated/derived columns
            new_row["USTF Name"] = ustf_zh
            new_row["USTF ENG Name"] = ustf_en
            new_row["USTF Email"] = ustf_em
            new_row["Comments (TA)"] = ". ".join(m.get("ta_comments", []))
            new_row["Comments (Instructor)"] = m.get("instr_comments", "")

            rows.append(new_row)
            made_any_row = True
            is_first_for_course = False

        # Emit rows: leading TAs first, then other TAs
        for is_leading, plist in [(True, pairs_leading), (False, pairs_keep)]:
            for pair in plist:
                make_row(is_leading, pair)

        # If no TA at all, still emit one placeholder row
        if not made_any_row:
            make_row(False, None)

    final_df = pd.DataFrame(rows)

    # Ensure TA-related columns exist even if empty
    for col in ta_cols:
        if col not in final_df.columns:
            final_df[col] = ""

    # Preserve original column order; append any new columns at the end
    col_order = list(original_assign_df.columns) + [c for c in ta_cols if c not in original_assign_df.columns]
    return final_df[col_order]


def build_unmatched_dataframe(student_map: Dict[str, dict], matched: set[str]) -> pd.DataFrame:
    rows = []
    for sid, s in student_map.items():
        if sid in matched:
            continue
        prefs = s.get("prefs", [])
        rows.append({
            "Student ID": sid,
            "Student Name (EN)": s.get("name_en",""),
            "Student Name (ZH)": s.get("name_zh",""),
            "Email": s.get("email",""),
            "Major/Fields": s.get("major",""),
            "Cohort": s.get("cohort",""),
            "Supervisor (EN)": s.get("supervisor_en",""),
            "Supervisor (ZH)": s.get("supervisor_zh",""),
            "First Priority": prefs[0] if len(prefs) > 0 else "",
            "Second Priority": prefs[1] if len(prefs) > 1 else "",
            "Third Priority": prefs[2] if len(prefs) > 2 else "",
            "Fourth Priority": prefs[3] if len(prefs) > 3 else "",
            "Other information": s.get("other",""),
        })
    return pd.DataFrame(rows)


# ======================================================================
# Streamlit UI
# ======================================================================

st.set_page_config(page_title="EasyMatcher CUHKSZ", layout="wide")
st.title("📚 EasyMatcher for Course TA/USTF — Revised, Logged, & Capped")

st.markdown(
    "- **Step 1**: Upload all four XLSX files.\n"
    "- **Step 2**: Review previews.\n"
    "- **Step 3**: (Optional) Manual edit.\n"
    "- **Step 4**: Download assignment + logs."
)

with st.expander("📋 Required Files / Column Hints", expanded=False):
    st.markdown(
        """
**Teaching Assignment (official):**
- Columns include: _Course Code_, _Course Title_, _Course Description(ENG)_ or _Course Description_, _Course Instructors_, _Course Type_, _TA number_, _USTF number_

**TA Name List:**
- Columns: _Name_ (ZH), _English Name_, _ID_, _Email_, _Cohort_, _Major/Fields_, _Main Supervisor（CHI）_, _Main Supervisor_, _官网Research Field_

**Instructor Preference:**
- Columns: _Course code_, _Course instructor's name_, _Instructor preference for corresponding TA_, _Instructor preference for Leading TA_, _USTF Number Needed_, _Other remarks_

**Student Preference:**
- Columns: _Student ID_, _Full Name_ (or _Normalized Name_), four priority columns, _USTF Remark_, _Other information_
        """
    )

# Uploaders
st.markdown("### 1) Teaching Assignment XLSX")
assign_file = st.file_uploader("Upload official Teaching Assignment 👇", type=["xlsx"], key="assign")

st.markdown("### 2) TA Name List XLSX")
namelist_file = st.file_uploader("Upload TA Name List 👇", type=["xlsx"], key="names")

st.markdown("### 3) Instructor Preference XLSX")
instr_file = st.file_uploader("Upload Instructor Preference 👇", type=["xlsx"], key="instr")

st.markdown("### 4) Student Preference XLSX")
stud_file = st.file_uploader("Upload Student Preference 👇", type=["xlsx"], key="stud")

if assign_file and namelist_file and instr_file and stud_file:
    try:
        assign_df = _read_main_or_first(assign_file, prefer=["UG+PG"])
        names_df  = _read_main_or_first(namelist_file)
        instr_df  = _read_main_or_first(instr_file)
        stud_df   = _read_main_or_first(stud_file)
    except Exception as e:
        st.error(f"Failed to read XLSX files: {e}")
        LOG.log("ERROR", "READ_FAIL", detail=str(e))
        st.stop()

    # Build structures
    courses_assign = load_teaching_assignment(assign_df)
    courses_instr  = load_instructor_sheet(instr_df)
    id_to_profile, name_to_id_from_names = load_ta_name_list(names_df)
    stud_pref_map = load_student_prefs(stud_df)

    # Merge student map & comprehensive name->id dictionary
    student_map, name_to_id = build_student_map(id_to_profile, stud_pref_map)

    # Revised matching (initial pass)
    matched_sids = perform_matching(courses_instr, courses_assign, student_map, name_to_id)

    # Iteratively rematch until stable: prune → fill → prune, repeat
    iterative_match_until_stable(courses_assign, student_map, matched_sids)

    # Build outputs
    assign_df_out = build_assignment_dataframe(courses_assign, id_to_profile, assign_df)
    unmatched_df  = build_unmatched_dataframe(student_map, matched_sids)

    # Session state
    st.session_state.assign_df = assign_df_out
    st.session_state.unmatched_df = unmatched_df
    st.session_state.courses = courses_assign
    st.session_state.student_map = student_map
    st.session_state.matched = matched_sids
    st.session_state.id_to_profile = id_to_profile
    st.session_state.name_to_id = name_to_id

    # Previews
    st.subheader("📑 Preview — Course Assignments (Teaching Assignment format)")
    st.dataframe(st.session_state.assign_df, use_container_width=True, height=420)

    # With guaranteed completion + caps, some students may become unmatched if caps are tight
    st.subheader("🚧 Students Still Unmatched")
    st.dataframe(st.session_state.unmatched_df, use_container_width=True, height=200)

    # Manual edit
    with st.expander("✏️ Manual Edit", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            action = st.selectbox("Action", ["Add", "Remove"])
            course_sel = st.text_input("Course code (exact)")
        with col2:
            student_id_sel = st.text_input("Student ID")
            if action == "Add":
                role = st.selectbox("Add as", ["Normal TA", "Leading TA", "USTF"])
            go = st.button("Apply change")

        if go and course_sel and student_id_sel:
            changed = False
            course_sel = course_sel.replace(" ", "").strip()
            if course_sel not in st.session_state.courses:
                st.error("Unknown course code.")
                LOG.log("ERROR", "MANUAL_EDIT_BAD_COURSE", detail=course_sel)
            elif student_id_sel not in st.session_state.student_map:
                st.error("Unknown student ID.")
                LOG.log("ERROR", "MANUAL_EDIT_BAD_STUDENT", detail=student_id_sel)
            else:
                s_info = st.session_state.student_map[student_id_sel]
                name_pair = (s_info["name"], student_id_sel)
                c_meta = st.session_state.courses[course_sel]

                if action == "Add":
                    present_in_course = any(
                        name_pair in c_meta.get(cat, [])
                        for cat in ("leading_ta", "ta_supervisor", "ta_instr_only", "ta_student_only", "ustf_instr", "ustf_ta")
                    )
                    if role == "Normal TA":
                        if not present_in_course:
                            c_meta["ta_instr_only"].append(name_pair)
                            st.session_state.matched.add(student_id_sel)
                            changed = True
                            LOG.log("INFO", "MANUAL_MATCH", course=course_sel, student_id=student_id_sel,
                                    student_name=s_info["name"], detail="Manual add: Normal TA")
                        else:
                            st.warning("Student already listed for this course.")
                    elif role == "Leading TA":
                        if not present_in_course:
                            c_meta.setdefault("leading_ta", []).append(name_pair)
                            st.session_state.matched.add(student_id_sel)
                            changed = True
                            LOG.log("INFO", "MANUAL_MATCH", course=course_sel, student_id=student_id_sel,
                                    student_name=s_info["name"], detail="Manual add: Leading TA")
                        else:
                            st.warning("Student already listed for this course.")
                    else:  # USTF
                        ustf_name = s_info["name"]
                        ustf_name_list = [n for n, _ in c_meta.get("ustf_instr", [])]
                        ustf_name_list.extend(n for n, _ in c_meta.get("ustf_ta", []))
                        if ustf_name not in ustf_name_list:
                            c_meta.setdefault("ustf_instr", []).append(name_pair)
                            st.session_state.matched.add(student_id_sel)
                            changed = True
                            LOG.log("INFO", "MANUAL_MATCH", course=course_sel, student_id=student_id_sel,
                                    student_name=s_info["name"], detail="Manual add: USTF")
                        else:
                            st.warning("Student already listed under USTF for this course.")
                else:  # Remove
                    removed = False
                    for cat in ("leading_ta", "ta_supervisor", "ta_instr_only", "ta_student_only", "ustf_instr", "ustf_ta"):
                        if name_pair in c_meta.get(cat, []):
                            c_meta[cat].remove(name_pair)
                            removed = True
                            changed = True
                            LOG.log("INFO", "MANUAL_REMOVE", course=course_sel, student_id=student_id_sel,
                                    student_name=s_info["name"], detail=f"Removed from {cat}")
                    if not removed:
                        st.warning("Student not found in this course.")

                    # If no longer assigned anywhere (including USTF), mark as unmatched
                    still_assigned = any(
                        name_pair in m.get(cat, [])
                        for m in st.session_state.courses.values()
                        for cat in ("leading_ta","ta_supervisor","ta_instr_only","ta_student_only","ustf_instr","ustf_ta")
                    )
                    if not still_assigned and student_id_sel in st.session_state.matched:
                        st.session_state.matched.remove(student_id_sel)
                        LOG.log("INFO", "MANUAL_MARK_UNMATCHED", student_id=student_id_sel, student_name=s_info["name"])

                if changed:
                    # After any manual change, re‑enforce caps for consistency
                    enforce_ta_caps(st.session_state.courses, st.session_state.student_map, st.session_state.matched)

                    # Rebuild via iterative fill until stable
                    iterative_match_until_stable(st.session_state.courses, st.session_state.student_map, st.session_state.matched)
                    st.session_state.assign_df = build_assignment_dataframe(
                        st.session_state.courses, st.session_state.id_to_profile
                    )
                    st.session_state.unmatched_df = build_unmatched_dataframe(
                        st.session_state.student_map, st.session_state.matched
                    )
                    st.success("Update applied ✔️")
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.info("No changes applied.")

    # Downloads: assignment + unmatched + log
    st.subheader("⬇️ Downloads")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        st.session_state.assign_df.to_excel(writer, index=False, sheet_name="UG+PG")
        st.session_state.unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched")
    buffer.seek(0)
    st.download_button(
        label="📥 Generate Assignment XLSX",
        data=buffer,
        file_name="Teaching_Assignment_Filled.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Log file (CSV)
    log_df = LOG.df()
    st.dataframe(log_df, use_container_width=True, height=260)
    log_csv = log_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="🧾 Download Run Log (CSV)",
        data=log_csv,
        file_name="run_log.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload all four files to start (Teaching Assignment, TA Name List, Instructor Preference, Student Preference).")