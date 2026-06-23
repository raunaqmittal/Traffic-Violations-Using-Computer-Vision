"""
Database repository — all insert and query operations.
One function per operation. No ORM patterns.
"""

import csv
from pathlib import Path
from sqlalchemy import select, func, and_
from sqlalchemy.engine import Engine
from src.database.schema import violations_table, get_engine
from src.models import ViolationRecord


def insert_violation(record: ViolationRecord, engine: Engine) -> int:
    x1, y1, x2, y2 = record.bbox
    row = {
        "violation_type": record.violation_type,
        "confidence": record.confidence,
        "status": record.status,
        "vehicle_id": record.vehicle_id,
        "camera_id": record.camera_id,
        "plate_number": record.plate_number,
        "plate_confidence": record.plate_confidence,
        "timestamp": record.timestamp,
        "frame_id": record.frame_id,
        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
        "is_blurry": record.is_blurry,
        "evidence_image_path": record.evidence_image_path,
        "evidence_json_path": record.evidence_json_path,
        "evidence_sha256": record.evidence_sha256,
    }
    with engine.connect() as conn:
        result = conn.execute(violations_table.insert().values(**row))
        conn.commit()
        return result.inserted_primary_key[0]


def query_violations(
    engine: Engine,
    violation_type: str | None = None,
    plate_number: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    limit: int = 500,
) -> list[dict]:
    filters = []
    if violation_type:
        filters.append(violations_table.c.violation_type == violation_type)
    if plate_number:
        filters.append(violations_table.c.plate_number.ilike(f"%{plate_number}%"))
    if date_from:
        filters.append(violations_table.c.timestamp >= date_from)
    if date_to:
        filters.append(violations_table.c.timestamp <= date_to)
    if status:
        filters.append(violations_table.c.status == status)

    stmt = select(violations_table)
    if filters:
        stmt = stmt.where(and_(*filters))
    stmt = stmt.order_by(violations_table.c.timestamp.desc()).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().fetchall()
    return [dict(r) for r in rows]


def count_by_type(engine: Engine) -> dict[str, int]:
    stmt = (
        select(violations_table.c.violation_type, func.count().label("count"))
        .group_by(violations_table.c.violation_type)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return {r[0]: r[1] for r in rows}


def count_by_date(engine: Engine) -> list[dict]:
    stmt = (
        select(
            func.substr(violations_table.c.timestamp, 1, 10).label("date"),
            func.count().label("count"),
        )
        .group_by("date")
        .order_by("date")
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return [{"date": r[0], "count": r[1]} for r in rows]


def export_csv(engine: Engine, output_path: str, **filters) -> str:
    rows = query_violations(engine, **filters, limit=10000)
    if not rows:
        return output_path
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
