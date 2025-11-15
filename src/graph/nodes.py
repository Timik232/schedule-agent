"""LangGraph node implementations."""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from datetime import date as date_type, datetime, time as time_type, timedelta
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ..config import LLMProvider, get_settings
from ..models.schemas import (
    ExtractedEntities,
    GroupScheduleLessonResponse,
    GroupScheduleResponse,
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
from ..utils.llm_factory import get_chat_model
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
    ToolName.GET_GROUP_SCHEDULE: sql_templates.get_group_schedule,
    ToolName.GET_ROOM_SCHEDULE: sql_templates.get_room_schedule,
    ToolName.FIND_COMMON_FREE_SLOTS: meeting_slots.plan_meeting_slots,
}

LLM_CLIENT_FACTORY: Callable[..., BaseChatModel] = get_chat_model


def configure_llm_client_factory(factory: Optional[Callable[..., BaseChatModel]]) -> None:
    """Configure which callable instantiates chat models for LangGraph nodes."""

    global LLM_CLIENT_FACTORY
    LLM_CLIENT_FACTORY = factory or get_chat_model


def _acquire_llm_client(*, streaming: bool = False) -> BaseChatModel:
    return LLM_CLIENT_FACTORY(streaming=streaming)


GROUP_CODE_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}-\d{2}-\d{2}", re.IGNORECASE)
DATE_TOKEN_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё0-9./-]+")


def _normalize_explicit_date(token: str) -> Optional[str]:
    cleaned = token.strip().replace(",", "")
    if not cleaned:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned

    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", cleaned)
    if match:
        day, month, year = match.groups()
        try:
            return date_type(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

    match = re.fullmatch(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", cleaned)
    if match:
        year, month, day = match.groups()
        try:
            return date_type(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

    return None


def _extract_first_date_token(query: str) -> Optional[date_type]:
    for token in DATE_TOKEN_PATTERN.findall(query.lower()):
        normalized_relative = _normalize_relative_date(token)
        if normalized_relative:
            return date_type.fromisoformat(normalized_relative)

        explicit_iso = _normalize_explicit_date(token)
        if explicit_iso:
            return date_type.fromisoformat(explicit_iso)

    return None


def _extract_group_codes(query: str) -> list[str]:
    groups: list[str] = []
    for match in GROUP_CODE_PATTERN.findall(query):
        normalized = match.lower()
        if normalized not in groups:
            groups.append(normalized)
    return groups


def _fallback_entities(query: str) -> ExtractedEntities:
    """Heuristic fallback when LLM extraction fails."""

    lowered = query.lower()
    groups = _extract_group_codes(query)
    date_value = _extract_first_date_token(query)

    intent = QueryIntent.FIRST_CLASS
    if any(keyword in lowered for keyword in ("препод", "преподав", "teacher")):
        intent = QueryIntent.TEACHER_SCHEDULE
    elif any(keyword in lowered for keyword in ("встрече", "встреч", "слот")) and groups:
        intent = QueryIntent.FIND_MEETING_SLOT
    elif any(keyword in lowered for keyword in ("расписан", "заняти", "пары")) and groups:
        intent = QueryIntent.GROUP_SCHEDULE

    date_range: Optional[tuple[date_type, date_type]] = None
    if date_value and intent in {QueryIntent.GROUP_SCHEDULE, QueryIntent.TEACHER_SCHEDULE}:
        date_range = (date_value, date_value)

    return ExtractedEntities(
        intent=intent,
        groups=groups,
        teachers=[],
        rooms=[],
        date=date_value,
        date_range=date_range,
    )


RELATIVE_DATE_OFFSETS: dict[str, int] = {
    "сегодня": 0,
    "завтра": 1,
    "послезавтра": 2,
    "вчера": -1,
    "позавчера": -2,
}


def _normalize_relative_date(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None

    value = raw.strip().lower()
    for prefix in ("на ", "в ", "во "):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break

    value = value.replace(".", "")
    offset = RELATIVE_DATE_OFFSETS.get(value)
    if offset is None:
        return None

    return (date_type.today() + timedelta(days=offset)).isoformat()


def _normalize_llm_entities_payload(raw_payload: str, query: str) -> str:
    try:
        payload = json.loads(raw_payload)
    except Exception:
        return raw_payload

    if not isinstance(payload, dict):
        return raw_payload

    changed = False
    fallback: Optional[ExtractedEntities] = None

    if "group" in payload and "groups" not in payload:
        groups_value = payload.pop("group")
        if isinstance(groups_value, str):
            payload["groups"] = [groups_value]
        elif isinstance(groups_value, list):
            payload["groups"] = groups_value
        changed = True

    if "teacher" in payload and "teachers" not in payload:
        teacher_value = payload.pop("teacher")
        if isinstance(teacher_value, str):
            payload["teachers"] = [teacher_value]
        elif isinstance(teacher_value, list):
            payload["teachers"] = teacher_value
        changed = True

    if "room" in payload and "rooms" not in payload:
        room_value = payload.pop("room")
        if isinstance(room_value, str):
            payload["rooms"] = [room_value]
        elif isinstance(room_value, list):
            payload["rooms"] = room_value
        changed = True

    normalized_date = _normalize_relative_date(payload.get("date"))
    if normalized_date is not None:
        payload["date"] = normalized_date
        changed = True

    if isinstance(payload.get("date_range"), list):
        date_range_list = payload["date_range"]
        normalized_range: list[str] = []
        for item in date_range_list:
            if isinstance(item, str):
                normalized = _normalize_relative_date(item) or item
                normalized_range.append(normalized)
            else:
                normalized_range.append(item)
        payload["date_range"] = normalized_range
        changed = True

    if not payload.get("intent"):
        fallback = fallback or _fallback_entities(query)
        payload["intent"] = fallback.intent.value
        changed = True

    if changed:
        return json.dumps(payload, ensure_ascii=False)

    return raw_payload


def _post_process_entities(entities: ExtractedEntities) -> ExtractedEntities:
    """Apply heuristics to fill missing fields such as date ranges."""

    updates: dict[str, Any] = {}

    if entities.date and not entities.date_range:
        updates["date_range"] = (entities.date, entities.date)

    if entities.intent == QueryIntent.GENERAL_QUESTION:
        if entities.groups and (entities.date_range or entities.date):
            updates["intent"] = QueryIntent.GROUP_SCHEDULE
        elif entities.teachers and (entities.date_range or entities.date):
            updates["intent"] = QueryIntent.TEACHER_SCHEDULE
            if not entities.date_range and entities.date:
                updates["date_range"] = (entities.date, entities.date)

    if entities.intent == QueryIntent.TEACHER_SCHEDULE and not entities.date_range and entities.date:
        updates["date_range"] = (entities.date, entities.date)

    if entities.intent == QueryIntent.GROUP_SCHEDULE and not entities.date_range and entities.date:
        updates["date_range"] = (entities.date, entities.date)

    if updates:
        entities = entities.model_copy(update=updates)

    return entities


async def parse_query_node(state: AgentState) -> AgentStateUpdate:
    """Extract intent and entities from the user query."""

    query = state.get("user_query", "")
    if not query:
        return {"error": "Empty user query"}

    try:
        llm = _acquire_llm_client()
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(error=str(exc)).exception("Failed to acquire LLM client for parsing")
        entities = _post_process_entities(_fallback_entities(query))
        return {
            "intent": entities.intent,
            "entities": entities,
        }
    prompt = build_entity_extraction_prompt(query)
    settings = get_settings()
    provider = settings.llm_client_config.provider
    supports_structured = provider == LLMProvider.COPILOT

    try:
        if supports_structured:
            response = await llm.ainvoke(  # type: ignore[attr-defined]
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": ExtractedEntities.model_json_schema(),
                },
            )
            payload = response.content
            raw_entities = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        else:
            response = await llm.ainvoke(prompt)  # type: ignore[attr-defined]
            payload = response.content
            raw_entities = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

        normalized_payload = _normalize_llm_entities_payload(raw_entities, query)
        entities = ExtractedEntities.model_validate_json(normalized_payload)
    except (ValidationError, ValueError, TypeError) as exc:
        LOGGER.bind(query=query, provider=provider.value, error=str(exc)).warning("Entity extraction failed; using fallback")
        entities = _fallback_entities(query)
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(query=query, provider=provider.value, error=str(exc)).error("LLM extraction failed")
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

    if intent == QueryIntent.GROUP_SCHEDULE:
        if not entities or not entities.groups:
            return {"error": "Academic group not provided"}
        if not entities.date_range:
            return {"error": "Date range not provided"}

        date_start, date_end = entities.date_range
        group_norm = entities.groups[0]
        params = {
            "group_norm": group_norm,
            "date_start": date_start,
            "date_end": date_end,
        }
        LOGGER.bind(
            tool=ToolName.GET_GROUP_SCHEDULE.value,
            group=group_norm,
            date_start=str(date_start),
            date_end=str(date_end),
        ).debug("Prepared tool call")
        return {
            "tool_calls": [
                ToolCall(
                    tool=ToolName.GET_GROUP_SCHEDULE,
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
            return {}
        try:
            return LessonResponse.model_validate(outcome).model_dump()
        except ValidationError:
            return outcome

    if tool_call.tool == ToolName.GET_TEACHER_SCHEDULE:
        teacher_lessons: list[ScheduleLessonResponse] = []
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
                teacher_lessons.append(ScheduleLessonResponse.model_validate(lesson_payload))
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
            lessons=teacher_lessons,
        ).model_dump()

    if tool_call.tool == ToolName.GET_GROUP_SCHEDULE:
        group_lessons: list[GroupScheduleLessonResponse] = []
        for row in outcome or []:
            lesson_payload = {
                "date": row.get("date"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "subject": row.get("subject"),
                "teachers": list(row.get("teachers") or []),
                "rooms": list(row.get("rooms") or []),
                "lesson_type": row.get("lesson_type"),
            }
            try:
                group_lessons.append(GroupScheduleLessonResponse.model_validate(lesson_payload))
            except ValidationError:
                LOGGER.warning("Failed to validate group lesson payload", payload=lesson_payload)

        try:
            date_start = _ensure_date(tool_call.parameters.get("date_start"))
            date_end = _ensure_date(tool_call.parameters.get("date_end"))
        except Exception:  # noqa: BLE001
            date_start = date_end = date_type.today()

        group_norm = str(tool_call.parameters.get("group_norm", ""))
        return GroupScheduleResponse(
            group=group_norm,
            date_start=date_start,
            date_end=date_end,
            lessons=group_lessons,
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

    try:
        llm = _acquire_llm_client()
    except Exception as exc:  # noqa: BLE001
        LOGGER.bind(error=str(exc)).exception("Failed to acquire LLM client for formatting")
        message = "Не удалось сформировать ответ, но данные доступны."
        return {
            "final_response": message,
            "final_payload": prompt_payload,
        }

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
