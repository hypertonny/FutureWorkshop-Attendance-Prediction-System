"""
Lightweight API + static frontend server for the Workshop Predictor UI.

Run:
  python api_server.py --port 8000

This serves:
  - /                 -> frontend/index.html
  - /api/health       -> health check
  - /api/overview     -> dataset and model summary
  - /api/options      -> form dropdown options
  - /api/charts       -> aggregated chart payloads
    - /api/topic-analysis -> topic-level aggregates for analysis page
    - /api/model-details  -> model comparison + maintenance timeline
  - /api/predict      -> real model prediction for a planned event
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.predict import predict_single_event


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_PATH = BASE_DIR / "master_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_DATA_CACHE = {
    "mtime": None,
    "df": None,
}


def _ensure_dataset_exists() -> None:
    if DATASET_PATH.exists():
        return

    from generate_data import generate_default_dataset

    generate_default_dataset(output_path=str(DATASET_PATH))


def load_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load dataset with mtime-based caching to keep API fast."""
    _ensure_dataset_exists()
    mtime = DATASET_PATH.stat().st_mtime

    if (
        force_reload
        or _DATA_CACHE["df"] is None
        or _DATA_CACHE["mtime"] is None
        or _DATA_CACHE["mtime"] != mtime
    ):
        df = pd.read_csv(DATASET_PATH)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        _DATA_CACHE["df"] = df
        _DATA_CACHE["mtime"] = mtime

    return _DATA_CACHE["df"].copy()


def _safe_unique(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(str(v) for v in df[col].dropna().unique().tolist())


def _load_model_payload() -> dict:
    comparison = {}
    winner = None
    metrics = {}
    threshold = 0.5

    comp_path = MODELS_DIR / "model_comparison.json"
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)
        winner = comparison.get("winner")

    meta = {}
    if winner:
        meta_path = MODELS_DIR / f"{winner}_latest_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

    if not meta:
        for name in ["xgboost", "random_forest", "logistic_regression"]:
            meta_path = MODELS_DIR / f"{name}_latest_meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                winner = name
                break

    if meta:
        metrics = meta.get("metrics", {})
        threshold = meta.get("threshold", metrics.get("threshold", 0.5))

    return {
        "winner": winner,
        "metrics": metrics,
        "threshold": threshold,
        "comparison": comparison,
    }


def _to_float_list(series: pd.Series) -> list[float]:
    return [round(float(v), 4) for v in series.tolist()]


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_overview_payload(df: pd.DataFrame) -> dict:
    model = _load_model_payload()
    avg_attendance = float(df["attended"].mean()) if "attended" in df.columns else 0.0

    return {
        "total_events": int(df["event_id"].nunique()) if "event_id" in df.columns else 0,
        "total_students": int(df["student_id"].nunique()) if "student_id" in df.columns else 0,
        "registrations": int(len(df.index)),
        "avg_attendance": round(avg_attendance, 4),
        "model": model,
        "updated_at": _utc_iso_z(),
    }


def build_options_payload(df: pd.DataFrame) -> dict:
    present_days = set(_safe_unique(df, "day_of_week"))
    ordered_days = [d for d in DAY_ORDER if d in present_days]
    if not ordered_days:
        ordered_days = DAY_ORDER[:]

    return {
        "topics": _safe_unique(df, "topic"),
        "speaker_types": _safe_unique(df, "speaker_type"),
        "days": ordered_days,
        "time_slots": _safe_unique(df, "time_slot"),
        "modes": _safe_unique(df, "mode"),
        "promotion_levels": _safe_unique(df, "promotion_level"),
        "exam_proximity": [1, 2, 3],
    }


def build_charts_payload(df: pd.DataFrame) -> dict:
    payload = {}

    # Topic
    if {"topic", "attended"}.issubset(df.columns):
        topic = (
            df.groupby("topic")["attended"]
            .mean()
            .sort_values(ascending=False)
        )
        payload["topic"] = {
            "labels": topic.index.tolist(),
            "rates": _to_float_list(topic),
        }
    else:
        payload["topic"] = {"labels": [], "rates": []}

    # Day
    if {"day_of_week", "attended"}.issubset(df.columns):
        day_rates = df.groupby("day_of_week")["attended"].mean()
        day_labels = [d for d in DAY_ORDER if d in day_rates.index]
        payload["day"] = {
            "labels": day_labels,
            "rates": [round(float(day_rates.get(d, 0.0)), 4) for d in day_labels],
        }
    else:
        payload["day"] = {"labels": [], "rates": []}

    # Monthly
    monthly = {
        "labels": [],
        "registrations": [],
        "attended": [],
        "rates": [],
    }
    if {"event_date", "attended"}.issubset(df.columns):
        monthly_df = df.dropna(subset=["event_date"]).copy()
        if not monthly_df.empty:
            monthly_df["month"] = monthly_df["event_date"].dt.to_period("M").astype(str)
            monthly_grouped = monthly_df.groupby("month").agg(
                registrations=("attended", "count"),
                attended=("attended", "sum"),
                rates=("attended", "mean"),
            )
            monthly = {
                "labels": monthly_grouped.index.tolist(),
                "registrations": [int(v) for v in monthly_grouped["registrations"].tolist()],
                "attended": [int(v) for v in monthly_grouped["attended"].tolist()],
                "rates": _to_float_list(monthly_grouped["rates"]),
            }
    payload["monthly"] = monthly

    # Speaker
    if {"speaker_type", "attended"}.issubset(df.columns):
        speaker = (
            df.groupby("speaker_type")["attended"]
            .mean()
            .sort_values(ascending=False)
        )
        payload["speaker"] = {
            "labels": speaker.index.tolist(),
            "rates": _to_float_list(speaker),
        }
    else:
        payload["speaker"] = {"labels": [], "rates": []}

    # Mode
    if {"mode", "attended"}.issubset(df.columns):
        mode = df.groupby("mode")["attended"].mean().sort_values(ascending=False)
        payload["mode"] = {
            "labels": mode.index.tolist(),
            "rates": _to_float_list(mode),
        }
    else:
        payload["mode"] = {"labels": [], "rates": []}

    # Club
    club_order = ["Low", "Medium", "High"]
    if {"club_activity_level", "attended"}.issubset(df.columns):
        club = df.groupby("club_activity_level")["attended"].mean()
        labels = [c for c in club_order if c in club.index]
        payload["club"] = {
            "labels": labels,
            "rates": [round(float(club.get(c, 0.0)), 4) for c in labels],
        }
    else:
        payload["club"] = {"labels": [], "rates": []}

    # Exam
    if {"exam_proximity", "attended"}.issubset(df.columns):
        exam = df.groupby("exam_proximity")["attended"].mean()
        labels = [1, 2, 3]
        payload["exam"] = {
            "labels": labels,
            "rates": [round(float(exam.get(l, 0.0)), 4) for l in labels],
        }
    else:
        payload["exam"] = {"labels": [1, 2, 3], "rates": [0.0, 0.0, 0.0]}

    # Department
    if {"department", "attended"}.issubset(df.columns):
        dept = (
            df.groupby("department")["attended"]
            .mean()
            .sort_values(ascending=False)
        )
        payload["department"] = {
            "labels": dept.index.tolist(),
            "rates": _to_float_list(dept),
        }
    else:
        payload["department"] = {"labels": [], "rates": []}

    # Semester
    if {"semester", "attended"}.issubset(df.columns):
        sem = (
            df.groupby("semester")["attended"]
            .mean()
            .sort_index()
        )
        payload["semester"] = {
            "labels": [str(int(v)) for v in sem.index.tolist()],
            "rates": _to_float_list(sem),
        }
    else:
        payload["semester"] = {"labels": [], "rates": []}

    # Heatmap day x slot
    heat_days = []
    heat_slots = []
    heat_matrix = []
    if {"day_of_week", "time_slot", "attended"}.issubset(df.columns):
        present_days = set(df["day_of_week"].dropna().astype(str).tolist())
        heat_days = [d for d in DAY_ORDER if d in present_days]
        heat_slots = sorted(df["time_slot"].dropna().astype(str).unique().tolist())
        pivot = (
            df.groupby(["day_of_week", "time_slot"])["attended"]
            .mean()
            .unstack(fill_value=0.0)
            .reindex(index=heat_days, columns=heat_slots, fill_value=0.0)
        )
        heat_matrix = [
            [round(float(v), 4) for v in row]
            for row in pivot.to_numpy().tolist()
        ]

    payload["heatmap"] = {
        "days": heat_days,
        "slots": heat_slots,
        "matrix": heat_matrix,
    }

    return payload


def build_topic_analysis_payload(
    df: pd.DataFrame,
    topic: str | None = None,
    school: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    selected_topic = (topic or "All Topics").strip()
    selected_school = (school or "All Schools").strip()
    available_topics = _safe_unique(df, "topic")

    if selected_school and selected_school != "All Schools" and "department" in df.columns:
        df = df[df["department"].astype(str) == selected_school].copy()

    if (date_from or date_to) and "event_date" in df.columns:
        date_series = pd.to_datetime(df["event_date"], errors="coerce")
        if date_from:
            from_dt = pd.to_datetime(date_from, errors="coerce")
            if pd.notna(from_dt):
                df = df[date_series >= from_dt].copy()
                date_series = pd.to_datetime(df["event_date"], errors="coerce")
        if date_to:
            to_dt = pd.to_datetime(date_to, errors="coerce")
            if pd.notna(to_dt):
                df = df[date_series <= to_dt].copy()

    if selected_topic and selected_topic != "All Topics":
        topic_df = df[df["topic"].astype(str) == selected_topic].copy() if "topic" in df.columns else df.iloc[0:0].copy()
    else:
        selected_topic = "All Topics"
        topic_df = df.copy()

    if topic_df.empty:
        return {
            "selected_topic": selected_topic,
            "selected_school": selected_school,
            "available_topics": available_topics,
            "summary": {
                "events": 0,
                "registrations": 0,
                "attended": 0,
                "rate": 0.0,
            },
            "department": {"labels": [], "rates": []},
            "semester": {"labels": [], "rates": []},
            "mode": {"labels": [], "rates": [], "counts": [], "attended": []},
            "club": {"labels": ["Low", "Medium", "High"], "rates": [0.0, 0.0, 0.0]},
        }

    summary = {
        "events": int(topic_df["event_id"].nunique()) if "event_id" in topic_df.columns else 0,
        "registrations": int(len(topic_df.index)),
        "attended": int(topic_df["attended"].sum()) if "attended" in topic_df.columns else 0,
        "rate": round(float(topic_df["attended"].mean()), 4) if "attended" in topic_df.columns else 0.0,
    }

    department = {"labels": [], "rates": []}
    if {"department", "attended"}.issubset(topic_df.columns):
        dept_rates = topic_df.groupby("department")["attended"].mean().sort_values(ascending=False)
        department = {
            "labels": dept_rates.index.tolist(),
            "rates": _to_float_list(dept_rates),
        }

    semester = {"labels": [], "rates": []}
    if {"semester", "attended"}.issubset(topic_df.columns):
        sem_rates = topic_df.groupby("semester")["attended"].mean().sort_index()
        semester = {
            "labels": [str(int(v)) for v in sem_rates.index.tolist()],
            "rates": _to_float_list(sem_rates),
        }

    mode = {"labels": [], "rates": [], "counts": [], "attended": []}
    if {"mode", "attended"}.issubset(topic_df.columns):
        mode_grouped = topic_df.groupby("mode").agg(
            count=("attended", "count"),
            attended=("attended", "sum"),
            rate=("attended", "mean"),
        )
        mode = {
            "labels": mode_grouped.index.tolist(),
            "rates": _to_float_list(mode_grouped["rate"]),
            "counts": [int(v) for v in mode_grouped["count"].tolist()],
            "attended": [int(v) for v in mode_grouped["attended"].tolist()],
        }

    club_order = ["Low", "Medium", "High"]
    club = {"labels": club_order, "rates": [0.0, 0.0, 0.0]}
    if {"club_activity_level", "attended"}.issubset(topic_df.columns):
        club_rates = topic_df.groupby("club_activity_level")["attended"].mean()
        club = {
            "labels": club_order,
            "rates": [round(float(club_rates.get(level, 0.0)), 4) for level in club_order],
        }

    return {
        "selected_topic": selected_topic,
        "selected_school": selected_school,
        "available_topics": available_topics,
        "summary": summary,
        "department": department,
        "semester": semester,
        "mode": mode,
        "club": club,
    }


def build_model_details_payload() -> dict:
    model_payload = _load_model_payload()
    winner = model_payload.get("winner")

    winner_meta = {}
    if winner:
        winner_meta_path = MODELS_DIR / f"{winner}_latest_meta.json"
        if winner_meta_path.exists():
            with open(winner_meta_path, "r", encoding="utf-8") as f:
                winner_meta = json.load(f)

    retrain_history = []
    retrain_log_path = MODELS_DIR / "retrain_log.txt"
    if retrain_log_path.exists():
        with open(retrain_log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        retrain_history = lines[-10:]

    timeline = [
        {
            "phase": "Model Retraining",
            "frequency": "Every semester start",
            "trigger": "New semester begins",
            "action": "python src/retrain.py",
        },
        {
            "phase": "Data Refresh",
            "frequency": "After every 10+ events",
            "trigger": "New attendance logs available",
            "action": "python src/retrain.py --from-db",
        },
        {
            "phase": "Performance Audit",
            "frequency": "Monthly",
            "trigger": "Observed prediction quality drop",
            "action": "Review threshold sweep + comparison metrics",
        },
        {
            "phase": "Data Cleanup",
            "frequency": "End of semester",
            "trigger": "Semester close",
            "action": "Archive old slices and refresh baseline",
        },
        {
            "phase": "Dependency Updates",
            "frequency": "Quarterly",
            "trigger": "Security/library patch releases",
            "action": "Update requirements and run smoke tests",
        },
    ]

    features = winner_meta.get("feature_columns", []) if isinstance(winner_meta, dict) else []
    comparison = model_payload.get("comparison", {})

    return {
        "winner": winner,
        "winner_display": winner.replace("_", " ").title() if winner else "Unknown",
        "metrics": model_payload.get("metrics", {}),
        "threshold": model_payload.get("threshold", 0.5),
        "comparison": comparison,
        "feature_count": len(features),
        "top_features": features[:15],
        "retrain_history": retrain_history,
        "maintenance_timeline": timeline,
        "updated_at": _utc_iso_z(),
    }


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class WorkshopRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/api/health":
                _json_response(self, 200, {"ok": True, "status": "healthy"})
                return

            if path == "/api/overview":
                df = load_dataset()
                _json_response(self, 200, build_overview_payload(df))
                return

            if path == "/api/options":
                df = load_dataset()
                _json_response(self, 200, build_options_payload(df))
                return

            if path == "/api/charts":
                df = load_dataset()
                _json_response(self, 200, build_charts_payload(df))
                return

            if path == "/api/topic-analysis":
                df = load_dataset()
                topic = query.get("topic", [None])[0]
                school = query.get("school", [None])[0]
                date_from = query.get("date_from", [None])[0]
                date_to = query.get("date_to", [None])[0]
                _json_response(self, 200, build_topic_analysis_payload(df, topic, school, date_from, date_to))
                return

            if path == "/api/model-details":
                _json_response(self, 200, build_model_details_payload())
                return
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        if path == "/":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/predict":
            _json_response(self, 404, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            required = [
                "topic",
                "speaker_type",
                "day_of_week",
                "time_slot",
                "mode",
                "duration_minutes",
                "exam_proximity",
                "promotion_level",
                "num_registrations",
            ]

            missing = [key for key in required if key not in payload]
            if missing:
                _json_response(
                    self,
                    400,
                    {"ok": False, "error": f"Missing fields: {', '.join(missing)}"},
                )
                return

            event_details = {
                "topic": str(payload["topic"]),
                "speaker_type": str(payload["speaker_type"]),
                "day_of_week": str(payload["day_of_week"]),
                "time_slot": str(payload["time_slot"]),
                "mode": str(payload["mode"]),
                "duration_minutes": int(payload["duration_minutes"]),
                "exam_proximity": int(payload["exam_proximity"]),
                "promotion_level": str(payload["promotion_level"]),
                "num_registrations": int(payload["num_registrations"]),
                "event_date": str(payload.get("event_date") or datetime.now(timezone.utc).date().isoformat()),
            }

            df = load_dataset()
            result = predict_single_event(event_details, df)
            _json_response(self, 200, {"ok": True, **result})
        except ValueError as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), WorkshopRequestHandler)
    print(f"[API] Serving frontend at http://{host}:{port}")
    print(
        "[API] Endpoints: /api/health, /api/overview, /api/options, /api/charts, "
        "/api/topic-analysis, /api/model-details, /api/predict"
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Workshop Predictor frontend + API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)