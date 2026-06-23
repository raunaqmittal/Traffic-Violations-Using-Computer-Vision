"""
Database schema definition using SQLAlchemy Core (table-level, no ORM classes).
Supports SQLite by default; upgrade to Postgres by changing the connection string.
"""

from sqlalchemy import (
    MetaData, Table, Column,
    Integer, Float, String, Boolean, DateTime, Text,
    create_engine,
)
from datetime import datetime


metadata = MetaData()

violations_table = Table(
    "violations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("violation_type", String(50), nullable=False, index=True),
    Column("confidence", Float, nullable=False),
    Column("status", String(20), nullable=False, default="pending"),
    Column("vehicle_id", Integer, nullable=False),
    Column("camera_id", String(50), nullable=False, default="cam_001"),
    Column("plate_number", String(20)),
    Column("plate_confidence", Float),
    Column("timestamp", String(30), nullable=False, index=True),
    Column("frame_id", Integer),
    Column("bbox_x1", Integer),
    Column("bbox_y1", Integer),
    Column("bbox_x2", Integer),
    Column("bbox_y2", Integer),
    Column("is_blurry", Boolean, default=False),
    Column("evidence_image_path", Text),
    Column("evidence_json_path", Text),
    Column("evidence_sha256", String(64)),
)


def get_engine(db_path: str = "artifacts/violations.db"):
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str = "artifacts/violations.db"):
    engine = get_engine(db_path)
    metadata.create_all(engine)
    return engine
