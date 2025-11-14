"""LangGraph node implementations."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date as date_type, datetime, time as time_type
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from ..config import get_settings
from ..models.schemas import (
    ExtractedEntities,
    LessonResponse,
    MeetingSlot,
    MeetingSlotsResponse,
    QueryIntent,
    ScheduleLessonResponse,
    TeacherScheduleResponse,
    ToolCall,
    ToolName,
)
from ..utils.logging import get_logger
from ..utils.validators import ValidationError as GuardValidationError, normalize_time_range
from ..utils.prompts import (
    FORMAT_RESPONSE_SYSTEM_PROMPT,
    build_entity_extraction_prompt,
    build_format_response_prompt,
)
from ..tools import meeting_slots, sql_templates
from .state import AgentState, AgentStateUpdate

LOGGER = get_logger(__name__)
TOOL_DISPATCH: dict[ToolName, Callable[..., Awaitable[Any]]] = {
    ToolName.GET_FIRST_CLASS_FOR_GROUP: sql_templates.get_first_class_for_group,
    ToolName.GET_TEACHER_SCHEDULE: sql_templates.get_teacher_schedule,
    ToolName.FIND_COMMON_FREE_SLOTS: meeting_slots.plan_meeting_slots,
}


def _fallback_entities(query: str) -> ExtractedEntities:
    """Heuristic fallback when LLM extraction fails."""

    lowered = query.lower()
    if any(keyword in lowered for keyword in ("препод", "преподав", "teacher")):
        return ExtractedEntities(intent=QueryIntent.TEACHER_SCHEDULE, groups=[], teachers=[], rooms=[], date=None)
    return ExtractedEntities(intent=QueryIntent.FIRST_CLASS, groups=[], teachers=[], rooms=[], date=None)


def _post_process_entities(entities: ExtractedEntities) -> ExtractedEntities:
    """Apply heuristics to fill missing fields such as date ranges."""

    updates: dict[str, Any] = {}

    if entities.date and not entities.date_range:
        updates["date_range"] = (entities.date, entities.date)

    if entities.intent == QueryIntent.GENERAL_QUESTION:
        if entities.groups and entities.date:
            updates["intent"] = QueryIntent.FIRST_CLASS
        elif entities.teachers and (entities.date_range or entities.date):
            updates["intent"] = QueryIntent.TEACHER_SCHEDULE
            if not entities.date_range and entities.date:
                updates["date_range"] = (entities.date, entities.date)

    if entities.intent == QueryIntent.TEACHER_SCHEDULE and not entities.date_range and entities.date:
        updates["date_range"] = (entities.date, entities.date)

    if updates:
        entities = entities.model_copy(update=updates)

    return entities


async def parse_query_node(state: AgentState) -> AgentStateUpdate:
    """Extract intent and entities from the user query."""

    query = state.get("user_query", "")
    if not query:
        return {"error": "Empty user query"}

    settings = get_settings()
    llm = ChatOpenAI(**settings.llm_kwargs)
    prompt = build_entity_extraction_prompt(query)

    try:
        response = await llm.ainvoke(  # type: ignore[attr-defined]
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": ExtractedEntities.model_json_schema(),
            },
        )
        entities = ExtractedEntities.model_validate_json(response.content)
    except (ValidationError, ValueError) as exc:
        LOGGER.bind(query=query, error=str(exc)).warning("Entity extraction failed; using fallback")
        entities = _fallback_entities(query)
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(query=query, error=str(exc)).exception("LLM extraction failed")
        entities = _fallback_entities(query)

    entities = _post_process_entities(entities)

    LOGGER.bind(intent=entities.intent.value, groups=len(entities.groups), teachers=len(entities.teachers)).debug(
        "Parsed user query"
    )

    return {
        "intent": entities.intent,
        "entities": entities,
    }


async def select_tool_node(state: AgentState) -> AgentStateUpdate:
    if state.get("error"):
        return {}

    intent = state.get("intent")
    entities = state.get("entities")

    LOGGER.bind(intent=intent.value if intent else None).debug("Selecting tool for intent")

    if intent == QueryIntent.FIRST_CLASS:
        if not entities or not entities.groups:
            return {"error": "Academic group not provided"}
        if entities.date is None:
            return {"error": "Date not provided"}

        group_norm = entities.groups[0]
        params = {
            "group_norm": group_norm,
            "lesson_date": entities.date,
        }
        LOGGER.bind(tool=ToolName.GET_FIRST_CLASS_FOR_GROUP.value, group=group_norm).debug("Prepared tool call")
        return {
            "tool_calls": [
                ToolCall(
                    tool=ToolName.GET_FIRST_CLASS_FOR_GROUP,
                    parameters=params,
                )
            ]
        }

    if intent == QueryIntent.TEACHER_SCHEDULE:
        if not entities or not entities.teachers:
            return {"error": "Teacher not provided"}
        if not entities.date_range:
            return {"error": "Date range not provided"}

        date_start, date_end = entities.date_range
        teacher_norm = entities.teachers[0]
        params = {
            "teacher_norm": teacher_norm,
            "date_start": date_start,
            "date_end": date_end,
        }
        LOGGER.bind(tool=ToolName.GET_TEACHER_SCHEDULE.value, teacher=teacher_norm, date_start=str(date_start), date_end=str(date_end)).debug(
            "Prepared tool call"
        )
        return {
            "tool_calls": [
                ToolCall(
                    tool=ToolName.GET_TEACHER_SCHEDULE,
                    parameters=params,
                )
            ]
        }

    if intent == QueryIntent.FIND_MEETING_SLOT:
        if not entities or not entities.groups:
            return {"error": "Groups not provided"}
        if not entities.date_range:
            return {"error": "Date range not provided"}

        date_start, date_end = entities.date_range
        try:
            if entities.time_range:
                time_start_raw, time_end_raw = entities.time_range
            else:
                time_start_raw = None
                time_end_raw = None

            parsed_start = _ensure_time(time_start_raw, time_type(8, 0)) if time_start_raw is not None else None
            parsed_end = _ensure_time(time_end_raw, time_type(20, 0)) if time_end_raw is not None else None
        except ValueError:
            return {"error": "Invalid time range"}

        try:
            time_start, time_end = normalize_time_range(parsed_start, parsed_end)
        except GuardValidationError as exc:
            return {"error": str(exc)}

        search_start = datetime.combine(date_start, time_start)
        search_end = datetime.combine(date_end, time_end)
        if search_end <= search_start:
            return {"error": "Search window is empty"}

        teacher_norm = entities.teachers[0] if entities.teachers else None
        params = {
            "teacher_norm": teacher_norm,
            "group_norms": entities.groups,
            "search_start": search_start,
            "search_end": search_end,
            "slot_duration_minutes": 90,
            "step_minutes": 15,
            "max_results": 10,
        }
        LOGGER.bind(
            tool=ToolName.FIND_COMMON_FREE_SLOTS.value,
            teacher=teacher_norm,
            groups=len(entities.groups),
        ).debug("Prepared tool call")
        return {
            "tool_calls": [
                ToolCall(
                    tool=ToolName.FIND_COMMON_FREE_SLOTS,
                    parameters=params,
                )
            ]
        }

    return {"error": f"Unsupported intent: {intent}"}


def _ensure_date(value: Any) -> date_type:
    if isinstance(value, date_type):
        return value
    if isinstance(value, str):
        return date_type.fromisoformat(value)
    raise ValueError(f"Cannot convert {value!r} to date")


def _ensure_time(value: Any, default: time_type) -> time_type:
    if value is None:
        return default
    if isinstance(value, time_type):
        return value
    if isinstance(value, str):
        return time_type.fromisoformat(value)
    raise ValueError(f"Cannot convert {value!r} to time")


def _normalize_tool_result(tool_call: ToolCall, outcome: Any) -> Any:
    if tool_call.tool == ToolName.GET_FIRST_CLASS_FOR_GROUP:
        if not outcome:
            return None
        try:
            return LessonResponse.model_validate(outcome).model_dump()
        except ValidationError:
            return outcome

    if tool_call.tool == ToolName.GET_TEACHER_SCHEDULE:
        lessons: list[ScheduleLessonResponse] = []
        for row in outcome or []:
            lesson_payload = {
                "date": row.get("date"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "subject": row.get("subject"),
                "groups": list(row.get("groups") or []),
                "rooms": list(row.get("rooms") or []),
                "lesson_type": row.get("lesson_type"),
            }
            try:
                lessons.append(ScheduleLessonResponse.model_validate(lesson_payload))
            except ValidationError:
                LOGGER.warning("Failed to validate lesson payload", payload=lesson_payload)
        try:
            date_start = _ensure_date(tool_call.parameters.get("date_start"))
            date_end = _ensure_date(tool_call.parameters.get("date_end"))
        except Exception:  # noqa: BLE001
            date_start = date_end = date_type.today()

        teacher = str(tool_call.parameters.get("teacher_norm", ""))
        return TeacherScheduleResponse(
            teacher=teacher,
            date_start=date_start,
            date_end=date_end,
            lessons=lessons,
        ).model_dump()

    if tool_call.tool == ToolName.FIND_COMMON_FREE_SLOTS:
        if not isinstance(outcome, dict):
            return outcome

        slots: list[MeetingSlot] = []
        for item in outcome.get("slots", []):
            try:
                slots.append(MeetingSlot.model_validate(item))
            except ValidationError:
                LOGGER.warning("Failed to validate meeting slot", payload=item)

        requested_groups = [str(group) for group in outcome.get("requested_groups") or []]
        requested_teacher = outcome.get("requested_teacher")
        requested_teacher_str = str(requested_teacher) if requested_teacher else None

        return MeetingSlotsResponse(
            slots=slots,
            requested_groups=requested_groups,
            requested_teacher=requested_teacher_str,
        ).model_dump()

    return outcome


async def execute_tools_node(state: AgentState) -> AgentStateUpdate:
    if state.get("error"):
        return {}

    tool_calls = state.get("tool_calls", [])
    results: list[dict[str, Any]] = []

    for call in tool_calls:
        tool_call = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        tool_fn = TOOL_DISPATCH.get(tool_call.tool)
        if tool_fn is None:
            return {"error": f"Unsupported tool: {tool_call.tool}"}

        LOGGER.bind(tool=tool_call.tool.value).debug("Executing tool")
        try:
            outcome = await tool_fn(**tool_call.parameters)
        except Exception as exc:  # noqa: BLE001
            LOGGER.bind(tool=tool_call.tool, error=str(exc)).exception("Tool execution failed")
            return {"error": "Tool execution failed"}

        payload = _normalize_tool_result(tool_call, outcome)
        results.append({"tool": tool_call.tool, "result": payload})
        LOGGER.bind(tool=tool_call.tool.value).debug("Tool execution completed")

    return {"tool_results": results}


async def format_response_node(state: AgentState) -> AgentStateUpdate:
    if state.get("error"):
        return {
            "final_response": "Произошла ошибка при обработке запроса.",
        }

    results = state.get("tool_results", [])
    if not results:
        return {
            "final_response": "Нет данных для отображения.",
        }

    payload = results[-1]["result"]
    LOGGER.bind(payload_type=type(payload).__name__).debug("Formatting response")
    if isinstance(payload, list):
        prompt_payload = {"items": payload}
    elif isinstance(payload, dict):
        prompt_payload = payload
    else:
        prompt_payload = {"value": payload}

    payload_json = json.dumps(prompt_payload, ensure_ascii=False, default=str)
    user_prompt = build_format_response_prompt(payload_json)
    settings = get_settings()
    llm = ChatOpenAI(**settings.llm_kwargs)

    try:
        completion = await llm.ainvoke(
            [
                SystemMessage(content=FORMAT_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        message = completion.content if isinstance(completion.content, str) else str(completion.content)
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(error=str(exc)).exception("Formatting response failed")
        message = "Не удалось сформировать ответ, но данные доступны."

    return {
        "final_response": message,
        "final_payload": prompt_payload,
    }
