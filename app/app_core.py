# ta_matching_app.py
"""
Teaching Assistant Matching App
--------------------------------
A lightweight Streamlit web application that matches teaching assistants (TAs) to
courses by combining instructor and student preferences.  The app works on both
Windows and macOS (or any platform that can run Python) and requires only a web
browser for the UI.

How to run (once Python ≥3.9 is installed):
>>> pip install streamlit pandas openpyxl
>>> streamlit run app_core.py

Then open the URL shown in the terminal (typically http://localhost:8501) in a
browser.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _read_main_or_first(uploaded_file):
    """
    Return a DataFrame from the sheet named 'main' (case-insensitive) if present;
    otherwise use the first sheet.
    """
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    name_map = {s.lower(): s for s in sheet_names}
    target_sheet = name_map.get("main", sheet_names[0])
    return xl.parse(sheet_name=target_sheet)

def _find_col(df: pd.DataFrame, substr: str) -> str:
    """Locate the first column whose *header* contains ``substr`` (case-insensitive)."""
    for col in df.columns:
        if substr.lower() in col.lower():
            return col
    raise KeyError(f"No column containing '{substr}' found in: {df.columns.tolist()}")


def _canonicalize_name(raw: str | float | int) -> str | None:
    """Convert a raw name string into the canonical "Firstname LASTNAME" form.

    Returns ``None`` when the entry represents *no specific person* (e.g.
    ``"None. Ph.D. with strong math"``).
    """
    if pd.isna(raw):
        return None

    s = str(raw).strip()
    if not s:
        return None

    if s.lower().startswith("none."):
        # Requirement only – caller should capture the tail as a remark.
        return None

    # Remove any leading / trailing punctuation and collapse  multiple spaces.
    s = re.sub(r"^[:,;\s]+|[:,;\s]+$", "", s)
    s = re.sub(r"\s+", " ", s)

    parts = s.split(" ")
    if len(parts) < 2:
        # Single token – treat as already canonical.
        return parts[0].title()

    # Heuristic: LASTNAME is written with *all* caps in the "LAST FIRST" form.
    if parts[0].isupper() and not parts[1].isupper():
        lastname, firstname_parts = parts[0], parts[1:]
    else:
        firstname_parts, lastname = parts[:-1], parts[-1]

    firstname = " ".join(p.title() for p in firstname_parts)
    lastname = lastname.upper()
    return f"{firstname} {lastname}"


def _parse_name_list(raw: str | float | int) -> List[str]:
    """Split a string like ``"Haotian MA, Tianci HOU"`` into a list of canonical names."""
    if pd.isna(raw):
        return []
    names = [n.strip() for n in str(raw).replace(";", ",").split(",")]
    out: List[str] = []
    for n in names:
        c = _canonicalize_name(n)
        if c:
            out.append(c)
    return out


def _parse_instructor_pref_cell(cell: str | float | int) -> Tuple[List[str], str]:
    """Return (preferred_names, textual_requirement)."""
    if pd.isna(cell):
        return [], ""

    s = str(cell).strip()
    if s.lower().startswith("none."):
        return [], s[5:].strip()
    return _parse_name_list(s), ""


def _parse_ustf_cell(cell: str | float | int) -> Tuple[int, int, List[str]]:
    """Return (min_needed, max_needed, canonical_name_list)."""
    if pd.isna(cell):
        return 0, 0, []

    text = str(cell)
    m = re.search(r"\[(\d+)\s*,\s*(\d+)\]", text)
    min_n = int(m.group(1)) if m else 0
    max_n = int(m.group(2)) if m else 0
    # Strip the leading range spec then parse the remaining name part.
    match = re.search(r"\((.*?)\)", text)
    names_part = match.group(1) if match else ""
    names = _parse_name_list(names_part)
    return min_n, max_n, names

def _get_student_id(raw_id: str | float | int) -> str:
    """Convert a raw student ID to a string, removing any trailing .0."""
    if pd.isna(raw_id):
        return "unknown"
    if isinstance(raw_id, float) and raw_id.is_integer():
        return str(int(raw_id))
    return str(raw_id).strip()

# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def load_instructor_sheet(df: pd.DataFrame) -> Dict[str, dict]:
    """Extract and normalise information course-by-course.

    Returns a mapping *course_code* → metadata-dict.
    """
    course_code_col = _find_col(df, "Course code")
    instructor_col = _find_col(df, "Course instructor's name")
    pref_ta_col = _find_col(df, "Instructor preference for corresponding TA")
    pref_lead_col = _find_col(df, "Instructor preference for Leading TA")
    ustf_col = _find_col(df, "USTF Number Needed")
    remarks_col = _find_col(df, "Other remarks")

    courses: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code = str(row[course_code_col]).replace(" ", "").strip()
        if not code:
            continue  # Skip empty rows.

        pref_names, req_comment = _parse_instructor_pref_cell(row[pref_ta_col])
        lead_names, _ = _parse_instructor_pref_cell(row[pref_lead_col])
        # Remove names that are in lead_names from pref_names
        pref_names = [name for name in pref_names if name not in lead_names]

        min_u, max_u, ustf_names = _parse_ustf_cell(row[ustf_col])
        
        # Combine comments.
        comment_parts = []
        if req_comment:
            comment_parts.append(req_comment)
        raw_remark = row[remarks_col]
        if not pd.isna(raw_remark):
            s = str(raw_remark).strip()
            if s:
                comment_parts.append(s)
        instr_comments = ". ".join(comment_parts)
        
        courses[code] = {
            "instructor": str(row[instructor_col]).strip(),
            "pref_names": pref_names,
            "lead_names": lead_names,
            "ustf_instr": ustf_names,
            "ustf_range": (min_u, max_u),
            "instr_comments": instr_comments,
            # Place-holders for matching results
            "ta_both": [],
            "ta_instr_only": [],
            "ta_student_only": [],
            "ustf_ta": [],
            "ta_comments": [],
        }

    return courses


def load_student_sheet(df: pd.DataFrame) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Return (student_id_map, name→id).

    *student_id_map* maps student-id → student-info-dict.
    """
    name_col = _find_col(df, "Normalized Name")
    id_col = _find_col(df, "Student ID")
    major_col = _find_col(df, "Major/Fields")
    cohort_col = _find_col(df, "Cohort")
    supervisor_col = _find_col(df, "Supervisor")
    ustf_remark_col = _find_col(df, "USTF Remark")
    other_col = _find_col(df, "Other information")

    pri_cols = [
        _find_col(df, "First Priority"),
        _find_col(df, "Second Priority"),
        _find_col(df, "Third Priority"),
        _find_col(df, "Fourth Priority"),
    ]

    student_map: Dict[str, dict] = {}
    name_to_id: Dict[str, str] = {}

    for _, row in df.iterrows():
        sid = _get_student_id(row[id_col])
        name = _canonicalize_name(row[name_col]) or ""
        if not sid or not name:
            continue
        prefs = [str(row[c]).replace(" ", "").strip() for c in pri_cols if not pd.isna(row[c]) and str(row[c]).strip()]
        
        # process other information
        raw_other = row[other_col]
        if pd.isna(raw_other):
            other_str = ""
        else:
            other_str = str(raw_other).strip()
        
        student_map[sid] = {
            "name": name,
            "prefs": prefs,
            "major": str(row[major_col]).strip(),
            "cohort": str(row[cohort_col]).strip(),
            "supervisor": str(row[supervisor_col]).strip(),
            "ustf_remark": _parse_name_list(row[ustf_remark_col]),
            "other": other_str,
        }
        name_to_id[name] = sid

    return student_map, name_to_id


def perform_matching(courses: Dict[str, dict], student_map: Dict[str, dict], name_to_id: Dict[str, str]):
    """Populate *courses* in-place and return a set of *matched student-ids*."""

    matched: set[str] = set()

    # Pass 1 – two-sided preferences.
    for code, meta in courses.items():
        for pref_name in meta["pref_names"]:
            sid = name_to_id.get(pref_name)
            if not sid or sid in matched:
                continue
            if code in student_map[sid]["prefs"]:
                meta["ta_both"].append((pref_name, sid))
                matched.add(sid)

    # Pass 2 – instructor-only.
    for code, meta in courses.items():
        for pref_name in meta["pref_names"]:
            sid = name_to_id.get(pref_name)
            if not sid or sid in matched:
                continue
            meta["ta_instr_only"].append((pref_name, sid))
            matched.add(sid)

    # Pass 3 – student-only preferences.
    for sid, sinfo in student_map.items():
        if sid in matched:
            continue
        for pref_code in sinfo["prefs"]:
            if pref_code in courses:
                courses[pref_code]["ta_student_only"].append((sinfo["name"], sid))
                matched.add(sid)
                break  # Only one course per student.
    
    # Pass 4 – leading TAs.
    for code, meta in courses.items():
        leading_pairs: List[Tuple[str, str]] = []
        for lead_name in meta.get("lead_names", []):
            sid = name_to_id.get(lead_name)
            leading_pairs.append((lead_name, sid))     
        # Store for downstream use (e.g. build_assignment_dataframe)
        meta["leading_ta"] = leading_pairs

    # USTF from TA remarks + comments gathering.
    for code, meta in courses.items():
        # From assigned students (any category) pick up their ustf remarks and other info.
        for cat in ("ta_both", "ta_instr_only", "ta_student_only"):
            for name, sid in meta[cat]:
                ustf_from_ta = student_map[sid]["ustf_remark"]
                if ustf_from_ta:
                    meta["ustf_ta"].extend(u for u in ustf_from_ta if u not in meta["ustf_ta"])
                other = student_map[sid]["other"]
                if other:
                    meta["ta_comments"].append(f"{name}: {other}")

        # De-duplicate USTF lists.
        meta["ustf_instr"] = list(dict.fromkeys(meta["ustf_instr"]))
        meta["ustf_ta"] = list(dict.fromkeys(meta["ustf_ta"]))

    return matched


def build_assignment_dataframe(courses: Dict[str, dict]) -> pd.DataFrame:
    """Convert the *courses* dict to a DataFrame suitable for display / export."""
    rows = []
    for code, m in courses.items():
        make_str = lambda pairs: ", ".join(f"{n} ({sid})" for n, sid in pairs)
        row = {
            "Course code": code,
            "Instructor": m["instructor"],
            "Leading TA": make_str(m["leading_ta"]),
            "TA (both-side)": make_str(m["ta_both"]),
            "TA (single-side: instructor)": make_str(m["ta_instr_only"]),
            "TA (single-side: student)": make_str(m["ta_student_only"]),
            "USTF (instructor)": ", ".join(m["ustf_instr"]),
            "USTF (TA)": ", ".join(m["ustf_ta"]),
            "Comments (Instructor)": m["instr_comments"],
            "Comments (TA)": ". ".join(m["ta_comments"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_unmatched_dataframe(student_map: Dict[str, dict], matched: set[str]) -> pd.DataFrame:
    rows = []
    for sid, s in student_map.items():
        if sid in matched:
            continue
        row = {
            "Student Name": s["name"],
            "Student ID": sid,
            "Major/Fields": s["major"],
            "Cohort": s["cohort"],
            "Supervisor": s["supervisor"],
            "First Priority": s["prefs"][0] if len(s["prefs"]) > 0 else "",
            "Second Priority": s["prefs"][1] if len(s["prefs"]) > 1 else "",
            "Third Priority": s["prefs"][2] if len(s["prefs"]) > 2 else "",
            "Fourth Priority": s["prefs"][3] if len(s["prefs"]) > 3 else "",
            "Other information": s["other"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EasyMatcher CUHKSZ", layout="wide")
st.title("📚 EasyMatcher for Course TA/USTF (By Xiaoyuan Liu)")

st.markdown(
    "Upload the instructor-preference sheet **and** the student-preference sheet (both XLSX).  \n"
    "After the preview you may **Generate** the assignment file or manually add / remove students.")

st.markdown("## Instructor preference XLSX")
instr_file = st.file_uploader("Upload instructor's preference file 👇", type=["xlsx"], key="instr")
st.markdown("## Student preference XLSX")
stud_file = st.file_uploader("Upload student's preference file 👇", type=["xlsx"], key="stud")

if instr_file and stud_file:
    try:
        instr_df = _read_main_or_first(instr_file)
        stud_df  = _read_main_or_first(stud_file)
    except Exception as e:
        st.error(f"Failed to read XLSX files: {e}")
        st.stop()

    courses = load_instructor_sheet(instr_df)
    student_map, name_to_id = load_student_sheet(stud_df)
    matched_sids = perform_matching(courses, student_map, name_to_id)

    assign_df = build_assignment_dataframe(courses)
    unmatched_df = build_unmatched_dataframe(student_map, matched_sids)

    # Store in session_state for later editing
    if "assign_df" not in st.session_state:
        st.session_state.assign_df = assign_df
        st.session_state.unmatched_df = unmatched_df
        st.session_state.courses = courses
        st.session_state.student_map = student_map
        st.session_state.matched = matched_sids

    st.subheader("📑 Preview - Course Assignments")
    st.dataframe(st.session_state.assign_df, use_container_width=True)

    st.subheader("🚧 Students Still Unmatched")
    st.dataframe(st.session_state.unmatched_df, use_container_width=True)

    # ---------------------------------------------------------------------
    # Manual editing controls
    # ---------------------------------------------------------------------
    with st.expander("✏️ Manually add / remove student", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            action = st.selectbox("Action", ["Add", "Remove"])
            course_sel = st.text_input("Course code (exact)")
        with col2:
            student_id_sel = st.text_input("Student ID")
            go = st.button("Apply change")

        if go and course_sel and student_id_sel:
            course_sel = course_sel.replace(" ", "").strip()
            if course_sel not in st.session_state.courses:
                st.error("Unknown course code.")
            elif student_id_sel not in st.session_state.student_map:
                st.error("Unknown student ID.")
            else:
                s_info = st.session_state.student_map[student_id_sel]
                name_pair = (s_info["name"], student_id_sel)
                c_meta = st.session_state.courses[course_sel]

                if action == "Add":
                    # Prevent double add.
                    present = any(name_pair in c_meta[k] for k in ("ta_both", "ta_instr_only", "ta_student_only"))
                    if not present:
                        c_meta["ta_student_only"].append(name_pair)  # treat as student-side add.
                        # Remove from unmatched if present.
                        if student_id_sel in st.session_state.matched:
                            pass  # Already matched elsewhere.
                        else:
                            st.session_state.matched.add(student_id_sel)
                    else:
                        st.warning("Student already listed for this course.")
                else:  # Remove
                    removed = False
                    for cat in ("ta_both", "ta_instr_only", "ta_student_only"):
                        if name_pair in c_meta[cat]:
                            c_meta[cat].remove(name_pair)
                            removed = True
                    if not removed:
                        st.warning("Student not found in this course.")
                    # Add back to unmatched list if student no longer assigned anywhere.
                    still_assigned = any(
                        name_pair in m[cat]
                        for m in st.session_state.courses.values()
                        for cat in ("ta_both", "ta_instr_only", "ta_student_only")
                    )
                    if not still_assigned and student_id_sel in st.session_state.matched:
                        st.session_state.matched.remove(student_id_sel)

                # Rebuild DataFrames in session_state
                st.session_state.assign_df = build_assignment_dataframe(st.session_state.courses)
                st.session_state.unmatched_df = build_unmatched_dataframe(
                    st.session_state.student_map, st.session_state.matched
                )
                st.success("Update applied ✔️")

    # ---------------------------------------------------------------------
    # Generate XLSX for download
    # ---------------------------------------------------------------------
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        st.session_state.assign_df.to_excel(writer, index=False, sheet_name="Assignments")
        st.session_state.unmatched_df.to_excel(writer, index=False, sheet_name="Unmatched")
    buffer.seek(0)

    st.download_button(
        label="📥 Generate XLSX",
        data=buffer,
        file_name="TA_Assignments.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
