"""
Imperator — LangGraph ReAct agent (no Context Broker).

Receives the full OpenAI payload, parses messages and conversation_id,
runs a ReAct loop with AE-provided tools, and returns response_text +
conversation_id.

Conversation state persisted via LangGraph PostgresSaver.
conversation_id from the payload = thread_id for the checkpointer.

Three conversation modes:
  - Pass a conversation_id → resume that thread
  - Pass nothing → resume the default Imperator thread
  - Pass "new" → create a new thread, return its ID

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

# Default thread ID for the Imperator's persistent conversation
_DEFAULT_THREAD_ID: str | None = None
_DEFAULT_THREAD_FILE = "/data/imperator_default_thread.txt"


def _get_default_thread_id() -> str:
    """Get or create the default Imperator thread ID.

    Persisted to a file so it survives container restarts.
    """
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


# ── State ────────────────────────────────────────────────────────────────


class ImperatorState(TypedDict):
    """State for the Imperator ReAct agent."""

    payload: dict  # Full OpenAI request body
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_id: Optional[str]
    response_text: Optional[str]
    error: Optional[str]
    iteration_count: int


# ── Nodes ────────────────────────────────────────────────────────────────


async def init_node(state: ImperatorState) -> dict:
    """Parse the OpenAI payload and set up the conversation.

    On first turn (no prior messages from checkpointer): builds full message
    list with system prompt + user message.
    On resumed turns (checkpointer loaded prior messages): only appends the
    new user message — system prompt and history are already in state.
    """
    payload = state.get("payload", {})
    existing_messages = state.get("messages", [])
    _log.info("init_node: existing_messages=%d, payload_model=%s",
              len(existing_messages), payload.get("model", "?"))

    # Resolve conversation_id
    conv_id = payload.get("conversation_id")
    if conv_id == "new":
        conv_id = str(uuid.uuid4())
        _log.info("Created new conversation thread: %s", conv_id)
    elif not conv_id:
        conv_id = _get_default_thread_id()

    # Extract the last user message from the payload
    raw_messages = payload.get("messages", [])
    new_user_msg = None
    for m in reversed(raw_messages):
        if m.get("role") == "user":
            new_user_msg = HumanMessage(content=m.get("content", ""))
            break

    if not new_user_msg:
        new_user_msg = HumanMessage(content="")

    # Resumed conversation: prior messages loaded by checkpointer
    if existing_messages:
        return {
            "messages": [new_user_msg],
            "conversation_id": conv_id,
            "iteration_count": 0,
        }

    # First turn: build full message list with system prompt
    system_content = _load_system_prompt()
    messages = []
    if system_content:
        messages.append(SystemMessage(content=system_content))
    messages.append(new_user_msg)

    return {
        "messages": messages,
        "conversation_id": conv_id,
        "iteration_count": 0,
    }


async def llm_call_node(state: ImperatorState) -> dict:
    """Call the LLM with bound tools.

    Uses get_chat_model from the AE config for the LLM.
    Uses get_tools_for_model from the AE tool registry for tools.
    """
    from app.config import get_chat_model, async_load_config, get_tuning

    config = await async_load_config()
    model_name = state.get("payload", {}).get("model", "host")

    # Get tools for this model from the AE
    from app.tools import get_tools_for_model, TOOL_REGISTRY

    # Host Imperator gets all tools
    if model_name == "host":
        tool_names = list(TOOL_REGISTRY.keys())
    else:
        tool_names = list(TOOL_REGISTRY.keys())  # TODO: read from eMAD config

    active_tools = get_tools_for_model(model_name, tool_names)

    # Get the LLM
    llm = get_chat_model(config, role="imperator")
    if active_tools:
        llm_with_tools = llm.bind_tools(active_tools)
    else:
        llm_with_tools = llm

    messages = list(state["messages"])

    # Truncate if too many messages
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


def should_continue(state: ImperatorState) -> str:
    """Route: tool_node if tool calls, else extract_response."""
    if state.get("error"):
        return "extract_response"

    messages = state.get("messages", [])
    if not messages:
        return "extract_response"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state.get("iteration_count", 0) >= _MAX_ITERATIONS:
            _log.warning("Hit max iterations (%d) — forcing end", _MAX_ITERATIONS)
            return "max_iterations_fallback"
        return "tool_node"

    return "extract_response"


async def max_iterations_fallback(state: ImperatorState) -> dict:
    """Fallback when max iterations reached."""
    return {
        "messages": [AIMessage(content=(
            "I was unable to complete that request within the allowed "
            "number of steps. Please try again, or break the request "
            "into smaller parts."
        ))],
    }


def extract_response(state: ImperatorState) -> dict:
    """Extract final response text and conversation_id."""
    response_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = str(msg.content)
            break

    if not response_text:
        response_text = "[No response generated]"

    return {
        "response_text": response_text,
        "conversation_id": state.get("conversation_id"),
    }


# ── Helpers ──────────────────────────────────────────────────────────────


def _load_system_prompt() -> str:
    """Load the Imperator system prompt from the config/prompts directory."""
    prompt_path = "/config/prompts/imperator_identity.md"
    try:
        with open(prompt_path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError):
        _log.warning("System prompt not found at %s", prompt_path)
        return "You are the Imperator, the host pMAD's conversational agent."


# ── Graph builder ────────────────────────────────────────────────────────


def build_imperator_flow(config: dict | None = None):
    """Build and compile the Imperator StateGraph with PostgresSaver.

    Returns a compiled graph that:
    - Receives {"payload": <full OpenAI body>} as initial state
    - Returns {"response_text": str, "conversation_id": str}
    - Persists conversation state via PostgresSaver
    """
    from app.config import async_load_config, get_chat_model
    from app.tools import get_tools_for_model, TOOL_REGISTRY

    # Get all tools for the host Imperator to create the ToolNode
    tool_names = list(TOOL_REGISTRY.keys())
    active_tools = get_tools_for_model("host", tool_names)

    tool_node_instance = ToolNode(active_tools) if active_tools else None

    workflow = StateGraph(ImperatorState)

    workflow.add_node("init_node", init_node)
    workflow.add_node("llm_call_node", llm_call_node)
    workflow.add_node("extract_response", extract_response)
    workflow.add_node("max_iterations_fallback", max_iterations_fallback)

    if tool_node_instance:
        workflow.add_node("tool_node", tool_node_instance)

    workflow.set_entry_point("init_node")
    workflow.add_edge("init_node", "llm_call_node")

    if tool_node_instance:
        workflow.add_conditional_edges(
            "llm_call_node",
            should_continue,
            {
                "tool_node": "tool_node",
                "max_iterations_fallback": "max_iterations_fallback",
                "extract_response": "extract_response",
            },
        )
        workflow.add_edge("tool_node", "llm_call_node")
    else:
        workflow.add_edge("llm_call_node", "extract_response")

    workflow.add_edge("max_iterations_fallback", "extract_response")
    workflow.add_edge("extract_response", END)

    # Use PostgresSaver for persistent conversation state
    from app.checkpointer import get_checkpointer

    return workflow.compile(checkpointer=get_checkpointer())
