"""
Imperator — LangGraph ReAct agent (no Context Broker).

Outer graph: resolves conversation_id, invokes inner ReAct subgraph
Inner graph: ReAct loop with AE tools, checkpointed via PostgresSaver

Conversation state persisted via PostgresSaver on the inner subgraph.
The outer graph resolves the thread_id BEFORE invoking the subgraph.

ARCH-05: ReAct loop is graph edges, not a while loop inside a node.
"""

import logging
import os
import uuid
from typing import Annotated, Optional

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

_log = logging.getLogger("base_pmad_te_ncb.imperator")

_MAX_ITERATIONS = 10

_DEFAULT_THREAD_ID: str | None = None
_DEFAULT_THREAD_FILE = "/data/imperator_default_thread.txt"


def _get_default_thread_id() -> str:
    global _DEFAULT_THREAD_ID
    if _DEFAULT_THREAD_ID is not None:
        return _DEFAULT_THREAD_ID
    try:
        if os.path.exists(_DEFAULT_THREAD_FILE):
            with open(_DEFAULT_THREAD_FILE, encoding="utf-8") as f:
                _DEFAULT_THREAD_ID = f.read().strip()
                if _DEFAULT_THREAD_ID:
                    return _DEFAULT_THREAD_ID
    except OSError:
        pass
    _DEFAULT_THREAD_ID = str(uuid.uuid4())
    try:
        os.makedirs(os.path.dirname(_DEFAULT_THREAD_FILE), exist_ok=True)
        with open(_DEFAULT_THREAD_FILE, "w", encoding="utf-8") as f:
            f.write(_DEFAULT_THREAD_ID)
    except OSError as exc:
        _log.warning("Failed to persist default thread ID: %s", exc)
    _log.info("Created default Imperator thread: %s", _DEFAULT_THREAD_ID)
    return _DEFAULT_THREAD_ID


def _load_system_prompt() -> str:
    prompt_path = "/config/prompts/imperator_identity.md"
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError):
        _log.warning("System prompt not found at %s", prompt_path)
        return "You are the Imperator, the host pMAD's conversational agent."


# ── Inner ReAct graph state ──────────────────────────────────────────


class ReactState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    response_text: Optional[str]
    error: Optional[str]
    iteration_count: int


# ── Inner ReAct graph nodes ──────────────────────────────────────────


async def llm_call_node(state: ReactState) -> dict:
    from app.config import get_chat_model, async_load_config, get_tuning
    from app.tools import get_tools_for_model, TOOL_REGISTRY

    config = await async_load_config()
    tool_names = list(TOOL_REGISTRY.keys())
    active_tools = get_tools_for_model("host", tool_names)

    llm = get_chat_model(config, role="imperator")
    llm_with_tools = llm.bind_tools(active_tools) if active_tools else llm

    messages = list(state["messages"])
    max_messages = get_tuning(config, "imperator_max_react_messages", 40)
    if len(messages) > max_messages:
        cut_index = len(messages) - (max_messages - 1)
        while cut_index < len(messages) and isinstance(messages[cut_index], ToolMessage):
            cut_index += 1
        messages = [messages[0]] + messages[cut_index:]

    _log.info("Imperator LLM call: %d messages", len(messages))

    try:
        response = await llm_with_tools.ainvoke(messages)
    except (openai.APIError, ValueError, RuntimeError, OSError) as exc:
        _log.error("Imperator LLM call failed: %s", exc)
        return {
            "messages": [AIMessage(content="I encountered an error processing your request.")],
            "error": str(exc),
        }

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def should_continue(state: ReactState) -> str:
    if state.get("error"):
        return "extract_response"
    messages = state.get("messages", [])
    if not messages:
        return "extract_response"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state.get("iteration_count", 0) >= _MAX_ITERATIONS:
            return "max_iterations_fallback"
        return "tool_node"
    return "extract_response"


async def max_iterations_fallback(state: ReactState) -> dict:
    return {
        "messages": [AIMessage(content=(
            "I was unable to complete that request within the allowed "
            "number of steps. Please try again."
        ))],
    }


def extract_response(state: ReactState) -> dict:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return {"response_text": str(msg.content)}
    return {"response_text": "[No response generated]"}


# ── Outer graph state ────────────────────────────────────────────────


class OuterState(TypedDict):
    payload: dict
    response_text: Optional[str]
    conversation_id: Optional[str]


# ── Build ────────────────────────────────────────────────────────────


def build_imperator_flow(config: dict | None = None):
    """Build the Imperator as an outer graph wrapping a checkpointed ReAct subgraph.

    Outer graph: resolves conversation_id, parses payload, invokes subgraph
    Inner graph: ReAct loop with tools, checkpointed via PostgresSaver
    """
    from app.config import get_chat_model
    from app.tools import get_tools_for_model, TOOL_REGISTRY
    from app.checkpointer import get_checkpointer

    # Build inner ReAct graph with checkpointer
    tool_names = list(TOOL_REGISTRY.keys())
    active_tools = get_tools_for_model("host", tool_names)
    tool_node_instance = ToolNode(active_tools) if active_tools else None

    inner = StateGraph(ReactState)
    inner.add_node("llm_call_node", llm_call_node)
    inner.add_node("extract_response", extract_response)
    inner.add_node("max_iterations_fallback", max_iterations_fallback)
    if tool_node_instance:
        inner.add_node("tool_node", tool_node_instance)

    inner.set_entry_point("llm_call_node")
    if tool_node_instance:
        inner.add_conditional_edges("llm_call_node", should_continue, {
            "tool_node": "tool_node",
            "max_iterations_fallback": "max_iterations_fallback",
            "extract_response": "extract_response",
        })
        inner.add_edge("tool_node", "llm_call_node")
    else:
        inner.add_edge("llm_call_node", "extract_response")
    inner.add_edge("max_iterations_fallback", "extract_response")
    inner.add_edge("extract_response", END)

    cp = get_checkpointer()
    _log.info("Compiling inner ReAct graph with checkpointer: %s", type(cp).__name__)
    compiled_inner = inner.compile(checkpointer=cp)

    # Build outer graph — no checkpointer, just preprocessing
    async def resolve_and_invoke(state: OuterState) -> dict:
        """Parse payload, resolve thread_id, invoke inner subgraph."""
        payload = state.get("payload", {})

        # Resolve conversation_id → thread_id
        conv_id = payload.get("conversation_id", "")
        if conv_id == "new":
            conv_id = str(uuid.uuid4())
            _log.info("New conversation thread: %s", conv_id)
        elif not conv_id:
            conv_id = _get_default_thread_id()

        # Parse messages from OpenAI payload
        raw_messages = payload.get("messages", [])
        lc_messages = []
        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "unknown")))
            else:
                lc_messages.append(HumanMessage(content=content))

        # Extract just the new user message
        new_user_msg = None
        for m in reversed(lc_messages):
            if isinstance(m, HumanMessage):
                new_user_msg = m
                break
        if not new_user_msg:
            new_user_msg = HumanMessage(content="")

        # Invoke inner subgraph with thread_id config
        inner_config = {"configurable": {"thread_id": conv_id}}

        # Check if this is a new thread (no prior messages in checkpointer)
        # by invoking with just the new message — the checkpointer will
        # load prior state automatically. On first turn, also add system prompt.
        checkpoint = await cp.aget_tuple(inner_config)
        if checkpoint and checkpoint.checkpoint.get("channel_values", {}).get("messages"):
            # Resumed — just send new user message
            inner_input = {"messages": [new_user_msg]}
        else:
            # New thread — send system prompt + user message
            system_content = _load_system_prompt()
            inner_input = {"messages": [SystemMessage(content=system_content), new_user_msg]}

        result = await compiled_inner.ainvoke(inner_input, config=inner_config)

        return {
            "response_text": result.get("response_text", ""),
            "conversation_id": conv_id,
        }

    outer = StateGraph(OuterState)
    outer.add_node("resolve_and_invoke", resolve_and_invoke)
    outer.set_entry_point("resolve_and_invoke")
    outer.add_edge("resolve_and_invoke", END)

    # Outer graph has NO checkpointer — it's stateless
    return outer.compile()
