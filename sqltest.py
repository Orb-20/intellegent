#!/usr/bin/env python3
"""
tests/run_sql_generator_tests_mistral.py

Extensive test runner for services/api/rag/sql_generator.py tailored for Jan 2022 data.
This script will:

- Force-use Mistral as the LLM provider (init_llm(prefer="mistral", allow_fallback=False)).
- Run a battery of diverse queries (simple -> very complex) focusing on January 2022 where appropriate.
- For each question:
    * Attempt LLM generation via try_generate_sql_with_llm(context, question).
    * If LLM path yields empty or unsafe SQL, fall back to rule_based_translator(question, schema).
    * Sanitize and run safety checks using config.sanitize_sql and config.is_safe_select.
    * Optionally run EXPLAIN if `--explain` is passed and a DB engine is configured.
- Produce CSV + JSON reports with diagnostics.
- Be safe for quotas: supports --no-llm and --delay between LLM calls.

Usage:
    python tests/run_sql_generator_tests_mistral.py [--no-llm] [--delay 0.5] [--out report_prefix] [--explain]

Note:
- Ensure MISTRAL_API_KEY and MISTRAL_MODEL are configured in your environment/config.
- The script prefers Mistral only (no automatic fallback to Gemini) to avoid Gemini quota issues.
"""

import argparse
import csv
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

# Import the modules under test
from services.api.rag import sql_generator, llm_utils, config

# Optional DB validate
try:
    from services.api.rag import db_utils
    from sqlalchemy import text
    DB_AVAILABLE = hasattr(db_utils, "engine") and db_utils.engine is not None
except Exception:
    DB_AVAILABLE = False

# Ensure we use Mistral exclusively (fail early if not available)
try:
    # prefer mistral and disallow fallback to Gemini to keep it safe
    llm_utils.init_llm(prefer="mistral", allow_fallback=False)
    llm_utils.init_chains(force=True)
    print(f"Initialized LLM provider (forced): Mistral")
except Exception as exc:
    print("Warning: failed to initialize Mistral as primary provider. LLM calls may fail.")
    print(str(exc))

# Build a comprehensive and diverse set of test questions
def build_test_questions_jan2022() -> List[str]:
    q: List[str] = []

    # Simple queries (single value / small selects)
    q += [
        "What is the average temperature in January 2022?",
        "What is the average salinity in Jan 2022?",
        "How many profiles were collected in January 2022?",
        "List 10 most recent profiles from January 2022",
        "What is the maximum pressure recorded in Jan 2022?"
    ]

    # Temporal specifics around Jan 2022
    q += [
        "Average temperature for January 2022 by day",
        "Count of profiles per day in January 2022",
        "List profiles collected between 2022-01-01 and 2022-01-31",
        "Show me average temp_degc for each week in January 2022",
    ]

    # Region-focused
    q += [
        "Average temperature in the Indian Ocean in January 2022",
        "Average salinity between 10N and 20N and 60E and 70E in Jan 2022",
        "List profile_id and juld for profiles in the North Atlantic in January 2022",
    ]

    # Depth-focused
    q += [
        "Average temperature below 1000 dbar in January 2022",
        "Salinity and temperature pairs for depths deeper than 1500 dbar in Jan 2022",
        "Profile measurements between 0 and 200 dbar for January 2022",
    ]

    # Aggregation and grouping
    q += [
        "Average temp_degc per platform_number for January 2022",
        "Count profiles per project_name in January 2022",
        "Monthly averages for January 2022 (trivial but tests grouping)",
    ]

    # Complex comparisons
    q += [
        "Compare average salinity in January 2022 vs January 2021 in the Indian Ocean",
        "Find profiles where psal_psu > 36 and temp_degc < 5 in Jan 2022",
        "Top 5 profiles by maximum temperature in January 2022",
    ]

    # Queries that require joins between profiles and levels
    q += [
        "Profile_id, juld, temp_degc where pres_dbar = 500 in January 2022",
        "List profiles with both temperature and salinity measurements in Jan 2022",
        "Return average temp_degc grouped by depth bin (0-100,100-500,500-1000) in Jan 2022",
    ]

    # Spatial + temporal + depth combos
    q += [
        "Average temperature below 200 dbar between 10N-20N & 60E-70E in Jan 2022",
        "Count of unique platforms in Jan 2022 in the South Pacific",
    ]

    # Ambiguous / natural-language edge cases
    q += [
        "Surface temperature in Jan 2022",  # tests interpretation of "surface"
        "Temperature trends during January 2022",  # tests timeseries request
        "Profiles with anomalously high salinity in Jan 2022",  # ambiguous anomaly
    ]

    # Very complex / multi-part queries
    q += [
        "For January 2022, find profiles where temperature decreased with depth (temp at 0-50 dbar > temp at 500-550 dbar)",
        "For each platform_number in Jan 2022, compute avg temp_degc and the count of profiles, order by avg temp desc limit 10",
        "Find profiles in Jan 2022 that have max salinity > 35 and return profile_id, juld, platform_number",
        "Compute monthly anomaly for Jan 2022 relative to Jan 2010-2021 average (this is intentionally complex)",
    ]

    # Add many variants programmatically to reach ~100 items, with focus on Jan 2022
    templates = [
        "What is the average {var} in January 2022?",
        "List {n} profiles in January 2022 with {var} measurements",
        "Count profiles in January 2022 where {var} > {val}",
        "Average {var} by platform_number in January 2022",
        "List profile_id, juld for profiles in January 2022 in the {region}",
        "Show {var} distribution in January 2022",
    ]
    variables = ["temperature", "salinity", "temp_degc", "psal_psu", "pressure", "pres_dbar"]
    regions = ["Indian Ocean", "North Atlantic", "South Pacific", "Gulf of Aden", "Arabian Sea"]
    values = ["0", "2", "5", "20", "36"]

    for t in templates:
        for var in variables:
            q.append(t.format(var=var, n=5, val="2", region=regions[0]))

    # More engineered complex queries (window functions, comparisons)
    q += [
        "Return profile_id and the difference between max_temp and min_temp within each profile for January 2022",
        "Find profiles whose temperature standard deviation over January 2022 is greater than 1",
        "For January 2022, compute median temperature (approx using percentile aggregation) per profile",
        "Find profiles with gaps (missing levels) in January 2022",  # tests for missing-handling
    ]

    # Fill up until 100 with safe transformations and duplicates avoided
    i = 0
    while len(q) < 100:
        base = q[i % len(q)]
        if i % 4 == 0:
            q.append(base + " (detailed)")
        elif i % 4 == 1:
            q.append("Top 5 " + base)
        elif i % 4 == 2:
            q.append("Distinct " + base)
        else:
            q.append("Summary " + base)
        i += 1

    # Deduplicate while preserving order
    seen = set()
    out = []
    for item in q:
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) >= 100:
            break

    return out

# Helper: optional EXPLAIN (best-effort)
def try_explain(sql: str) -> Tuple[Optional[bool], Optional[str]]:
    if not DB_AVAILABLE:
        return (None, "DB not available")
    try:
        with db_utils.engine.connect() as conn:
            conn.execute(text(f"EXPLAIN {sql}"))
        return (True, None)
    except Exception as exc:
        return (False, str(exc))

def run_tests(no_llm: bool = False, delay: float = 0.0, out_prefix: str = "sql_generation_report_jan2022", explain: bool = False):
    questions = build_test_questions_jan2022()
    rows: List[Dict[str, Any]] = []

    print(f"Starting tests: count={len(questions)}, no_llm={no_llm}, delay={delay}, explain={explain}, DB_AVAILABLE={DB_AVAILABLE}")

    for idx, question in enumerate(questions, start=1):
        start_time = time.time()
        record: Dict[str, Any] = {
            "id": idx,
            "question": question,
            "method": None,
            "raw_sql": None,
            "sanitized_sql": None,
            "is_safe": None,
            "explain_ok": None,
            "explain_error": None,
            "diagnostics": [],
            "duration_sec": None,
            "exception": None,
        }

        try:
            # Force context to mention Jan 2022 to encourage date-aware SQL
            context = "Use ARGO data. Focus on January 2022 (2022-01-01 to 2022-01-31)."
            raw_sql = ""
            method = None

            if no_llm:
                # Directly use rule-based translator
                rb_sql, rb_diags = sql_generator.rule_based_translator(question, getattr(config, "CANONICAL_COLUMNS", {}))
                raw_sql = rb_sql
                record["diagnostics"].extend(rb_diags or [])
                method = "rule_based"
            else:
                # Use LLM path (which in your current sql_generator may call llm_chain directly)
                try:
                    # Try generation; depending on your sql_generator implementation this returns sanitized or raw SQL
                    gen_sql = sql_generator.try_generate_sql_with_llm(context, question)
                    # If function returns sanitized SQL already, treat as sanitized candidate
                    if gen_sql:
                        raw_sql = gen_sql
                        method = "llm"
                    else:
                        method = "llm_empty"
                except Exception as exc:
                    record["exception"] = traceback.format_exc()
                    method = "llm_exception"
                    raw_sql = ""

                if not raw_sql:
                    # Fallback rule-based translator
                    fb_sql, fb_diags = sql_generator.rule_based_translator(question, getattr(config, "CANONICAL_COLUMNS", {}))
                    raw_sql = fb_sql
                    record["diagnostics"].extend(fb_diags or [])
                    method = "rule_based_fallback"

            record["method"] = method
            record["raw_sql"] = raw_sql

            # Sanitize using config.sanitize_sql if available
            try:
                sanitized = config.sanitize_sql(raw_sql or "")
            except Exception as exc:
                sanitized = raw_sql
                record["diagnostics"].append(f"sanitize_exception: {exc}")

            record["sanitized_sql"] = sanitized

            # Safety check
            try:
                safe = bool(config.is_safe_select(sanitized))
            except Exception as exc:
                safe = False
                record["diagnostics"].append(f"is_safe_select_exception: {exc}")
            record["is_safe"] = safe

            # Optional DB explain validation
            if explain and safe and sanitized:
                ok, err = try_explain(sanitized)
                record["explain_ok"] = ok
                record["explain_error"] = err
            else:
                record["explain_ok"] = None
                record["explain_error"] = "skipped"

        except Exception as exc:
            record["exception"] = traceback.format_exc()
            record["diagnostics"].append("unhandled_exception")

        finally:
            record["duration_sec"] = round(time.time() - start_time, 3)
            rows.append(record)
            print(f"[{idx}/{len(questions)}] method={record['method']} safe={record['is_safe']} dur={record['duration_sec']}s q='{question[:80]}'")
            if delay and not no_llm:
                time.sleep(delay)

    # Generate summary
    summary = {
        "total": len(rows),
        "by_method": {},
        "safe_count": sum(1 for r in rows if r["is_safe"]),
        "unsafe_count": sum(1 for r in rows if r["is_safe"] is False),
        "empty_sql": sum(1 for r in rows if not r["raw_sql"]),
        "explain_ok": sum(1 for r in rows if r.get("explain_ok") is True),
    }
    for r in rows:
        m = r["method"] or "unknown"
        summary["by_method"].setdefault(m, 0)
        summary["by_method"][m] += 1

    # Write outputs
    csv_path = out_prefix + ".csv"
    json_path = out_prefix + ".json"
    fieldnames = [
        "id", "question", "method", "raw_sql", "sanitized_sql", "is_safe",
        "explain_ok", "explain_error", "diagnostics", "duration_sec", "exception"
    ]

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                out_row = {k: r.get(k) for k in fieldnames}
                if isinstance(out_row.get("diagnostics"), list):
                    out_row["diagnostics"] = "; ".join(map(str, out_row["diagnostics"]))
                writer.writerow(out_row)
        print(f"Wrote CSV report to {csv_path}")
    except Exception as exc:
        print("Failed to write CSV:", exc)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, default=str)
        print(f"Wrote JSON report to {json_path}")
    except Exception as exc:
        print("Failed to write JSON:", exc)

    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Total tests: {summary['total']}")
    print("By method:")
    for k, v in summary["by_method"].items():
        print(f"  {k}: {v}")
    print(f"Safe SQL count: {summary['safe_count']}")
    print(f"Unsafe SQL count: {summary['unsafe_count']}")
    print(f"Empty SQL produced: {summary['empty_sql']}")
    print(f"EXPLAIN OK: {summary['explain_ok']}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    return summary, rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive SQL generator tests (Mistral-first, Jan 2022).")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM; use rule-based translator only.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay (seconds) between LLM calls to avoid quota spikes.")
    parser.add_argument("--out", type=str, default="sql_generation_report_jan2022", help="Output filename prefix.")
    parser.add_argument("--explain", action="store_true", help="Attempt EXPLAIN on DB for each safe SQL (requires db_utils.engine).")
    args = parser.parse_args()

    run_tests(no_llm=args.no_llm, delay=args.delay, out_prefix=args.out, explain=args.explain)
