from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class QueryIntent(str, Enum):
    """Supported intents derived from user queries."""

    FIRST_CLASS = "first_class"
    TEACHER_SCHEDULE = "teacher_schedule"
    GROUP_SCHEDULE = "group_schedule"
    ROOM_SCHEDULE = "room_schedule"
    FIND_MEETING_SLOT = "find_meeting_slot"
    GENERAL_QUESTION = "general_question"


class ExtractedEntities(BaseModel):
    """Entities extracted from the user query using the LLM."""

    intent: QueryIntent = Field(..., description="Type of user request")
    groups: list[str] = Field(default_factory=list, description="Normalized group names")
    teachers: list[str] = Field(default_factory=list, description="Normalized teacher names")
    rooms: list[str] = Field(default_factory=list, description="Normalized room names")
    date: Optional[date] = Field(default=None, description="Specific date in ISO format")
    date_range: Optional[tuple[date, date]] = Field(
        default=None, description="Inclusive start and end dates"
    )
    time_range: Optional[tuple[time, time]] = Field(
        default=None, description="Inclusive time range for filtering"
    )

    @field_validator("groups", "teachers", "rooms", mode="before")
    @classmethod
    def normalize_entities(cls, value: Optional[list[str]]) -> list[str]:
        if not value:
            return []
        return [item.strip().lower() for item in value if item.strip()]


class LessonResponse(BaseModel):
    """Normalized lesson data returned to end users."""

    date: date
    start_time: time
    end_time: time
    subject: str
    teacher: Optional[str] = None
    room: Optional[str] = None
    lesson_type: Optional[str] = None


class ScheduleLessonResponse(BaseModel):
    """Lesson representation used for schedule listings."""

    date: date
    start_time: time
    end_time: time
    subject: str
    groups: list[str] = Field(default_factory=list)
    rooms: list[str] = Field(default_factory=list)
    lesson_type: Optional[str] = None


class TeacherScheduleResponse(BaseModel):
    """Teacher schedule payload containing multiple lessons."""

    teacher: str
    date_start: date
    date_end: date
    lessons: list[ScheduleLessonResponse] = Field(default_factory=list)


class MeetingSlot(BaseModel):
    """Available meeting slot suggestion produced by the optimizer."""

    start: datetime
    end: datetime
    duration_minutes: int
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class MeetingSlotsResponse(BaseModel):
    """Payload aggregating meeting slot suggestions with context."""

    slots: list[MeetingSlot] = Field(default_factory=list)
    requested_groups: list[str] = Field(default_factory=list)
    requested_teacher: Optional[str] = None


class ToolName(str, Enum):
    """Registry of deterministic and fallback tools."""

    GET_FIRST_CLASS_FOR_GROUP = "get_first_class_for_group"
    GET_TEACHER_SCHEDULE = "get_teacher_schedule"
    GET_GROUP_SCHEDULE = "get_group_schedule"
    GET_ROOM_SCHEDULE = "get_room_schedule"
    FIND_COMMON_FREE_SLOTS = "find_common_free_slots"
    TEXT_TO_SQL = "text_to_sql"


class ToolCall(BaseModel):
    """Instruction that describes which tool to execute with what parameters."""

    tool: ToolName
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentReply(BaseModel):
    """Structured response returned by the agent to API clients."""

    response: str
    intent: Optional[QueryIntent] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    data: Any = None
    error: Optional[str] = None
