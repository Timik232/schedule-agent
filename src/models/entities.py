from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LessonEntity(BaseModel):
    """Raw lesson record fetched from database queries."""

    lesson_id: int
    start: datetime
    end: datetime
    subject: str
    lesson_type: Optional[str]
    room: Optional[str]
    teacher: Optional[str]
    academic_group: Optional[str]
