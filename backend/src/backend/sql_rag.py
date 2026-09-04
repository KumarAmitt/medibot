from __future__ import annotations

import logging
import re
import sqlite3
from langchain_core.prompts import ChatPromptTemplate

from backend.config import DB_PATH
from backend.llm import invoke_llm

logger = logging.getLogger(__name__)

SCHEMA_OVERVIEW = """
Tables in mediassist.db:

1) claims
   - claim_id TEXT
   - patient_id TEXT
   - patient_name TEXT
   - department TEXT          -- nephrology, cardiology, neurology, gynaecology,
                              -- orthopaedics, general_medicine, emergency
   - claim_type TEXT          -- reimbursement, cashless
   - diagnosis_code TEXT      -- ICD-10, e.g. I21.4
   - insurer TEXT             -- Star Health, HDFC Ergo, ICICI Lombard, New India Assurance,
                              -- United India, Bajaj Allianz, Niva Bupa, Care Health
   - claimed_amount REAL
   - approved_amount REAL     -- NULL when not approved
   - status TEXT              -- pending, approved, rejected, submitted, escalated
   - submitted_date TEXT      -- ISO date YYYY-MM-DD
   - resolved_date TEXT       -- ISO date or NULL

2) maintenance_tickets
   - ticket_id TEXT
   - equipment_name TEXT
   - equipment_id TEXT
   - category TEXT            -- sterilisation, infusion, radiology, monitoring,
                              -- surgical, laboratory
   - campus TEXT
   - issue_type TEXT          -- preventive_maintenance, sensor_failure,
                              -- battery_replacement, fault_reported, calibration_due
   - fault_code TEXT
   - raised_by TEXT
   - raised_date TEXT         -- ISO date YYYY-MM-DD
   - resolved_date TEXT
   - status TEXT              -- open, in_progress, escalated, resolved
   - resolution_note TEXT

Rules:
- SQLite dialect only.
- SELECT queries only. Never INSERT/UPDATE/DELETE/DROP/ATTACH.
- Dates are stored as TEXT in YYYY-MM-DD format. Use date() / strftime() for comparisons.
- Records are primarily from calendar year 2024. If the user says "last month" or
  "this year" and that window is empty, use the most recent month/year present in
  the relevant table and mention that in the final answer.
- Return a single SQL statement. No markdown, no commentary.
"""

SQL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You translate healthcare operations questions into a single SQLite SELECT.\n"
            f"{SCHEMA_OVERVIEW}",
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are MediBot, an analytics assistant for MediAssist Health Network. "
            "Turn the SQL result into a concise, accurate natural-language answer. "
            "Quote numbers exactly as returned. If the result is empty, say so clearly. "
            "Do not invent rows that were not returned.",
        ),
        (
            "human",
            "Question: {question}\n\nSQL:\n{sql}\n\nResult rows:\n{result}",
        ),
    ]
)


def fallback_sql(question: str) -> str:
    text = question.lower()
    if "escalated" in text and "claim" in text:
        return "SELECT COUNT(*) AS escalated_claims FROM claims WHERE status = 'escalated'"
    if "open" in text and "ticket" in text and ("category" in text or "which" in text):
        return (
            "SELECT category, COUNT(*) AS open_tickets FROM maintenance_tickets "
            "WHERE status IN ('open', 'in_progress') GROUP BY category ORDER BY open_tickets DESC"
        )
    if "rejected" in text and "insurer" in text:
        return (
            "SELECT insurer, COUNT(*) AS rejected_claims FROM claims "
            "WHERE status = 'rejected' GROUP BY insurer ORDER BY rejected_claims DESC"
        )
    if "cashless" in text and ("total" in text or "amount" in text or "approved" in text):
        return (
            "SELECT ROUND(SUM(claimed_amount), 2) AS total_claimed, "
            "ROUND(SUM(approved_amount), 2) AS total_approved "
            "FROM claims WHERE claim_type = 'cashless' AND status = 'approved'"
        )
    if "how many" in text and "claim" in text:
        return "SELECT status, COUNT(*) AS n FROM claims GROUP BY status"
    if "ticket" in text:
        return "SELECT status, COUNT(*) AS n FROM maintenance_tickets GROUP BY status"
    return "SELECT status, COUNT(*) AS n FROM claims GROUP BY status"


def extract_sql(raw: str) -> str:
    """Clean LLM output down to a single SELECT statement."""
    text = raw.strip()
    fence = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    match = re.search(r"(SELECT\b[\s\S]+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("The model did not produce a SELECT statement.")
    sql = match.group(1).strip().rstrip(";")

    lowered = sql.lower()
    forbidden = (
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "attach ",
        "detach ",
        "pragma ",
        "create ",
        "replace ",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError("Only read-only SELECT queries are permitted.")
    if ";" in sql:
        raise ValueError("Multiple SQL statements are not permitted.")
    return sql


def _run_sql(sql: str) -> list[dict]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(sql)
        return [dict(row) for row in cursor.fetchall()]


def sql_rag_chain(question: str) -> str:
    """
    SQL RAG in three explicit steps:
    1. Translate the question into SQL with an LLM
    2. Extract only the SQL statement from the raw LLM output
    3. Execute it, then ask the LLM to phrase a natural-language answer
    """
    try:
        raw_sql = invoke_llm(SQL_PROMPT, {"question": question}, temperature=0)
        logger.info("Raw SQL LLM output: %s", raw_sql)
        sql = extract_sql(raw_sql)
    except Exception as exc:
        logger.warning("SQL generation via Groq failed (%s); using heuristic SQL", exc)
        sql = fallback_sql(question)
    logger.info("Cleaned SQL: %s", sql)

    rows = _run_sql(sql)
    preview = rows[:50]
    result_text = "\n".join(str(row) for row in preview) if preview else "(no rows)"
    if len(rows) > 50:
        result_text += f"\n... ({len(rows) - 50} more rows omitted)"

    try:
        return invoke_llm(
            ANSWER_PROMPT,
            {"question": question, "sql": sql, "result": result_text},
            temperature=0,
        )
    except Exception as exc:
        logger.warning("SQL answer phrasing via Groq failed (%s)", exc)
        return f"Query: {sql}\n\nResult:\n{result_text}"
