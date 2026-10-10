"""SQLite case file. One connection per call so the review thread and the server stay apart."""

import json
import sqlite3
from datetime import datetime, timezone

from cxr.paths import DB_PATH, ensure_dirs


def connect() -> sqlite3.Connection:
    ensure_dirs()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS agent_reports (
                study_id INTEGER NOT NULL,
                agent TEXT NOT NULL,
                name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                heatmap_path TEXT,
                probabilities_json TEXT NOT NULL,
                PRIMARY KEY (study_id, agent),
                FOREIGN KEY (study_id) REFERENCES studies(id)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                study_id INTEGER PRIMARY KEY,
                outcome TEXT NOT NULL,
                fusion_mode TEXT NOT NULL,
                summary TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                stds_json TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                contested_json TEXT NOT NULL,
                FOREIGN KEY (study_id) REFERENCES studies(id)
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset TEXT NOT NULL,
                train_count INTEGER NOT NULL,
                test_count INTEGER NOT NULL,
                macro_f1 REAL NOT NULL,
                notes TEXT NOT NULL
            );
            """
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_study(image_path: str, notes: str) -> int:
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO studies (created_at, image_path, notes) VALUES (?, ?, ?)",
            (utc_now(), image_path, notes),
        )
        return int(cursor.lastrowid)


def save_report(study_id: int, report: dict) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO agent_reports (
                study_id, agent, name, algorithm, status, summary, detail, error,
                heatmap_path, probabilities_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_id,
                report["key"],
                report["name"],
                report["algorithm"],
                report["status"],
                report["summary"],
                report.get("detail") or "",
                report.get("error") or "",
                report.get("heatmap_path"),
                json.dumps(report["probabilities"]),
            ),
        )


def save_decision(study_id: int, decision: dict) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO decisions (
                study_id, outcome, fusion_mode, summary, scores_json, stds_json,
                findings_json, contested_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                study_id,
                decision["outcome"],
                decision["fusion_mode"],
                decision["summary"],
                json.dumps(decision["scores"]),
                json.dumps(decision["stds"]),
                json.dumps(decision["findings"]),
                json.dumps(decision["contested"]),
            ),
        )


def insert_training_run(dataset: str, train_count: int, test_count: int, macro_f1: float, notes: str) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO training_runs (created_at, dataset, train_count, test_count, macro_f1, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (utc_now(), dataset, train_count, test_count, macro_f1, notes),
        )


def latest_training() -> dict | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def list_studies() -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.created_at, s.notes, d.outcome, d.summary, d.findings_json
            FROM studies s
            JOIN decisions d ON d.study_id = s.id
            ORDER BY s.id DESC
            """
        ).fetchall()
    studies = []
    for row in rows:
        findings = json.loads(row["findings_json"])
        studies.append(
            {
                "id": row["id"],
                "code": study_code(row["id"]),
                "created_at": row["created_at"],
                "notes": row["notes"],
                "outcome": row["outcome"],
                "summary": row["summary"],
                "top_finding": findings[0]["label"] if findings else "",
            }
        )
    return studies


def study_code(study_id: int) -> str:
    return f"CXR-{study_id:04d}"


def get_study(study_id: int) -> dict | None:
    with connect() as connection:
        study = connection.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if study is None:
            return None
        decision = connection.execute(
            "SELECT * FROM decisions WHERE study_id = ?", (study_id,)
        ).fetchone()
        reports = connection.execute(
            "SELECT * FROM agent_reports WHERE study_id = ? ORDER BY rowid",
            (study_id,),
        ).fetchall()
    if decision is None:
        return None
    agents = []
    for report in reports:
        heatmap_url = None
        if report["heatmap_path"]:
            heatmap_url = f"/api/studies/{study_id}/heatmap/{report['agent']}"
        agents.append(
            {
                "key": report["agent"],
                "name": report["name"],
                "algorithm": report["algorithm"],
                "status": report["status"],
                "summary": report["summary"],
                "detail": report["detail"],
                "error": report["error"],
                "heatmap_url": heatmap_url,
                "probabilities": json.loads(report["probabilities_json"]),
            }
        )
    findings = json.loads(decision["findings_json"])
    return {
        "id": study["id"],
        "code": study_code(study["id"]),
        "created_at": study["created_at"],
        "notes": study["notes"],
        "image_url": f"/api/studies/{study_id}/image",
        "outcome": decision["outcome"],
        "fusion_mode": decision["fusion_mode"],
        "summary": decision["summary"],
        "scores": json.loads(decision["scores_json"]),
        "stds": json.loads(decision["stds_json"]),
        "findings": findings,
        "contested": json.loads(decision["contested_json"]),
        "agents": agents,
    }


def delete_study(study_id: int) -> dict | None:
    with connect() as connection:
        study = connection.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if study is None:
            return None
        paths = [study["image_path"]]
        for report in connection.execute(
            "SELECT heatmap_path FROM agent_reports WHERE study_id = ?", (study_id,)
        ):
            if report["heatmap_path"]:
                paths.append(report["heatmap_path"])
        connection.execute("DELETE FROM agent_reports WHERE study_id = ?", (study_id,))
        connection.execute("DELETE FROM decisions WHERE study_id = ?", (study_id,))
        connection.execute("DELETE FROM studies WHERE id = ?", (study_id,))
    return {"image_path": study["image_path"], "paths": paths}
