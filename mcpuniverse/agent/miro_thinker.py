"""
A MiroThinker agent implementation using MCP XML-style text-based tool calling.

This module contains the MiroThinker agent class and its configuration.
Unlike FunctionCall which uses LLM native tool calling, MiroThinker embeds
tool descriptions in the system prompt and parses XML-style <use_mcp_tool>
tags from the LLM's plain text response.
"""
# pylint: disable=broad-exception-caught
import os
import re
import json
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from mcp.types import TextContent, Tool

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.common.logger import get_logger
from mcpuniverse.tracer import Tracer
from mcpuniverse.callbacks.base import (
    send_message_async,
    CallbackMessage,
    MessageType
)
from .base import BaseAgentConfig, BaseAgent
from .utils import render_prompt_template
from .types import AgentResponse

DEFAULT_CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# ---------------------------------------------------------------------------
# Custom tool names and their MCP XML descriptions
# ---------------------------------------------------------------------------
CUSTOM_TOOL_NAMES = {"answer", "write_todos"}

CUSTOM_TOOLS_DESCRIPTION = """
## Server name: system

### Tool name: answer
Description: Report the final answer or result summary for the completed task.

Call this tool ONCE at the very end, after you have fully completed the task. Provide a clear, complete summary of your findings or results.

Input JSON schema: {"type": "object", "properties": {"content": {"type": "string", "description": "A clear, complete summary of the task result or answer."}}, "required": ["content"]}

### Tool name: write_todos
Description: Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

Input JSON schema: {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}, "tool_call_id": {"type": "string"}}, "required": ["todos", "tool_call_id"]}
""".strip()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from *text*, returning the remainder."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def get_mcp_tools_description(tools: Dict[str, List[Tool]]) -> str:
    """
    Convert MCP Tool objects to MiroThinker prompt format.

    Produces the text block that is embedded in the system prompt so the
    model knows which tools are available and how to call them.

    Args:
        tools: MCP tools organised by server name.

    Returns:
        A formatted string describing all tools.
    """
    sections: List[str] = []
    for server_name, tool_list in tools.items():
        sections.append(f"## Server name: {server_name}")
        for tool in tool_list:
            section = (
                f"### Tool name: {tool.name}\n"
                f"Description: {tool.description}\n\n"
                f"Input JSON schema: {json.dumps(tool.inputSchema, ensure_ascii=False)}"
            )
            sections.append(section)
    return "\n\n".join(sections)


def parse_mcp_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a MCP XML-style tool call from model response text.

    Looks for the first ``<use_mcp_tool>`` block and extracts server_name,
    tool_name and arguments.

    Returns:
        A dict ``{"server": ..., "tool": ..., "arguments": ...}`` compatible
        with ``BaseAgent.call_tool()``, or *None* if no tool call is found.
    """
    match = re.search(r'<use_mcp_tool>(.*?)</use_mcp_tool>', response_text, re.DOTALL)
    if not match:
        return None

    content = match.group(1)
    server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
    tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
    args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)

    server_name = server_match.group(1).strip() if server_match else None
    tool_name = tool_match.group(1).strip() if tool_match else None

    arguments: Dict[str, Any] = {}
    if args_match:
        raw_args = args_match.group(1).strip()
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            # Attempt repair via json_repair (optional dependency)
            try:
                from json_repair import repair_json  # type: ignore
                repaired = repair_json(raw_args)
                arguments = json.loads(repaired)
            except Exception:
                arguments = {}

    if server_name and tool_name:
        return {"server": server_name, "tool": tool_name, "arguments": arguments}
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MiroThinkerConfig(BaseAgentConfig):
    """
    Configuration class for MiroThinker agents.

    Attributes:
        system_prompt: The system prompt template file or string.
        context_examples: Additional context examples for the agent.
        max_iterations: Maximum number of reasoning iterations.
        summarize_tool_response: Whether to summarize tool responses using the LLM.
        use_custom_tools: Whether to use custom tools (answer, write_todos).
    """
    system_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "miro_thinker_prompt.j2")
    context_examples: str = ""
    max_iterations: int = 5
    summarize_tool_response: bool = False
    use_custom_tools: bool = False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MiroThinker(BaseAgent):
    """
    MiroThinker agent using MCP XML-style text-based tool calling.

    The model receives tool descriptions inside the system prompt and emits
    ``<use_mcp_tool>`` XML blocks which are parsed with regex.  Tool results
    are fed back as ``role: user`` messages.
    """

    config_class = MiroThinkerConfig
    alias = ["miro_thinker", "miro-thinker"]

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._logger = get_logger(f"{self.__class__.__name__}:{self._name}")
        self._history: List[str] = []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str) -> str:
        """Build the initial prompt with tool descriptions embedded."""
        tools_desc = get_mcp_tools_description(self._tools) if self._tools else ""
        if self._config.use_custom_tools:
            if tools_desc:
                tools_desc += "\n\n" + CUSTOM_TOOLS_DESCRIPTION
            else:
                tools_desc = CUSTOM_TOOLS_DESCRIPTION

        params: Dict[str, Any] = {
            "INSTRUCTION": self._config.instruction,
            "QUESTION": question,
            "MAX_STEPS": self._config.max_iterations,
            "MCP_TOOLS_DESCRIPTION": tools_desc,
            "DATE": datetime.now().strftime("%Y-%m-%d"),
        }
        if self._config.context_examples:
            params["CONTEXT_EXAMPLES"] = self._config.context_examples
        params.update(self._config.template_vars)

        return render_prompt_template(
            prompt_template=self._config.system_prompt,
            **params,
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _add_history(self, history_type: str, message: str):
        self._history.append(f"{history_type.title()}: {message}")

    def get_history(self) -> str:
        return "\n".join(self._history)

    def clear_history(self):
        self._history = []

    def reset(self):
        self.clear_history()

    # ------------------------------------------------------------------
    # Response handlers (mirroring FunctionCall patterns)
    # ------------------------------------------------------------------

    async def _handle_none_response(
        self, iter_num: int, callbacks: List[Any], tracer: Tracer,
    ) -> AgentResponse:
        """Handle case where LLM returns None."""
        self._logger.error("LLM returned None response, stopping execution")
        error_msg = (
            "The language model encountered a critical error and couldn't generate a response. "
            "This may be due to API failures, context length limits, or other issues."
        )
        self._add_history(history_type="error", message=error_msg)

        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=__file__,
                type=MessageType.LOG,
                metadata={
                    "event": "error",
                    "data": "".join([
                        f"{'=' * 66}\n",
                        f"Iteration: {iter_num + 1}\n",
                        f"{'-' * 66}\n",
                        f"\\033[31mCritical Error: LLM returned None response\\n\\033[0m",
                        f"\\033[33mDetails: {error_msg}\\n\\033[0m",
                    ]),
                },
            ),
        )

        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response=error_msg,
            trace_id=tracer.trace_id,
        )

    async def _handle_content_response(
        self,
        content: str,
        messages: List[Dict[str, Any]],
        iter_num: int,
        callbacks: List[Any],
        tracer: Tracer,
    ) -> Optional[AgentResponse]:
        """Handle a plain-text response that contains no tool call.

        Tries to parse JSON with an ``answer`` field; otherwise treats
        the text as intermediate thought and continues.
        """
        try:
            response_text = content.strip().strip('`').strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

            parsed_response = json.loads(response_text)

            if "answer" in parsed_response:
                self._add_history(history_type="answer", message=parsed_response["answer"])
                await send_message_async(
                    callbacks,
                    message=CallbackMessage(
                        source=__file__,
                        type=MessageType.LOG,
                        metadata={
                            "event": "plain_text",
                            "data": "".join([
                                f"{'=' * 66}\n",
                                f"Iteration: {iter_num + 1}\n",
                                f"{'-' * 66}\n",
                                f"\\033[32mThought: {parsed_response.get('thought', '')}\\n\\n\\033[0m",
                                f"\\033[31mAnswer: {parsed_response['answer']}\\n\\033[0m",
                            ]),
                        },
                    ),
                )
                return AgentResponse(
                    name=self._name,
                    class_name=self.__class__.__name__,
                    response=parsed_response["answer"],
                    trace_id=tracer.trace_id,
                )

            # No answer field – treat as thought
            self._add_history(history_type="thought", message=content)
            await send_message_async(
                callbacks,
                message=CallbackMessage(
                    source=__file__,
                    type=MessageType.LOG,
                    metadata={
                        "event": "plain_text",
                        "data": "".join([
                            f"{'=' * 66}\n",
                            f"Iteration: {iter_num + 1}\n",
                            f"{'-' * 66}\n",
                            f"\\033[32mThought: {content}\\n\\033[0m",
                        ]),
                    },
                ),
            )
            return None

        except json.JSONDecodeError as e:
            self._logger.error("Failed to parse response: %s", str(e))
            error_msg = (
                "Encountered an error in parsing LLM response:\n"
                f"{content}\n\n"
                "Please try again."
            )
            self._add_history(history_type="error", message=error_msg)
            messages.append({"role": "user", "content": error_msg})
            return None

        except Exception as e:
            self._logger.error("Failed to process response: %s", str(e))
            error_msg = (
                f"Encountered an unexpected error for the LLM response:\n"
                f"{content}.\n\nPlease try again."
            )
            self._add_history(history_type="error", message=error_msg)
            messages.append({"role": "user", "content": error_msg})
            return None

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    async def _execute(
        self,
        message: Union[str, List[str]],
        output_format: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> AgentResponse:
        if isinstance(message, (list, tuple)):
            message = "\n".join(message)
        if output_format is not None:
            message += f"\n{self._get_output_format_prompt(output_format)}"

        tracer: Tracer = kwargs.get("tracer", Tracer())
        callbacks: List[Any] = kwargs.get("callbacks", [])

        # Build initial prompt (tools embedded in text)
        initial_prompt = self._build_prompt(message)

        # Conversation uses user messages (no system role needed – prompt is
        # the first user message, matching FunctionCall's pattern).
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": initial_prompt},
        ]

        for iter_num in range(self._config.max_iterations):
            # Step countdown (skip first iteration)
            if iter_num > 0:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have {self._config.max_iterations - iter_num} "
                        "steps remaining. Please continue."
                    ),
                })

            self._add_history(history_type=f"Step {iter_num + 1}", message="")

            # Generate – no tools parameter (text-only)
            response = self._llm.generate(
                messages=messages,
                tracer=tracer,
                callbacks=callbacks,
            )

            # --- None response ---
            if response is None:
                return await self._handle_none_response(iter_num, callbacks, tracer)

            # --- Extract text from response ---
            if hasattr(response, "choices") and response.choices:
                response_text = (response.choices[0].message.content or "").strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                self._logger.error("Unexpected response type: %s", type(response))
                messages.append({
                    "role": "user",
                    "content": "Received an unexpected response format. Please try again.",
                })
                continue

            if not response_text:
                messages.append({
                    "role": "user",
                    "content": "Received an empty response from the LLM. Please try again.",
                })
                continue

            # Record assistant turn (full text including <think>)
            messages.append({"role": "assistant", "content": response_text})

            # Strip <think>...</think> for parsing purposes;
            # record thinking content in history for observability.
            think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
            if think_match:
                self._add_history(
                    history_type="thinking",
                    message=think_match.group(1).strip(),
                )
            cleaned_text = _strip_think_tags(response_text)

            # --- Try to parse MCP XML tool call ---
            tool_call = parse_mcp_tool_call(cleaned_text)

            if tool_call is not None:
                tool_name = tool_call["tool"]
                server_name = tool_call["server"]
                arguments = tool_call["arguments"]

                self._add_history(
                    history_type="action",
                    message=f"Using tool `{tool_name}` on server `{server_name}`",
                )
                self._add_history(
                    history_type="action input",
                    message=str(arguments),
                )

                # --- Custom tools ---
                if self._config.use_custom_tools and tool_name in CUSTOM_TOOL_NAMES:
                    await send_message_async(
                        callbacks,
                        message=CallbackMessage(
                            source=__file__,
                            type=MessageType.LOG,
                            metadata={
                                "event": "plain_text",
                                "data": "".join([
                                    f"{'=' * 66}\n",
                                    f"Iteration: {iter_num + 1}\n",
                                    f"{'-' * 66}\n",
                                    f"\033[31mCustom Tool: {tool_name}\n\n\033[0m",
                                    f"\033[33mArguments: {json.dumps(arguments)}\n\033[0m",
                                ]),
                            },
                        ),
                    )

                    if tool_name == "answer":
                        answer_content = arguments.get("content", "")
                        self._add_history(history_type="answer", message=answer_content)
                        return AgentResponse(
                            name=self._name,
                            class_name=self.__class__.__name__,
                            response=answer_content,
                            trace_id=tracer.trace_id,
                        )

                    if tool_name == "write_todos":
                        self._add_history(
                            history_type="result",
                            message="Todos updated successfully.",
                        )
                        messages.append({
                            "role": "user",
                            "content": "Todos updated successfully.",
                        })
                        continue

                # --- MCP tool ---
                try:
                    tool_result = await self.call_tool(
                        tool_call,
                        tracer=tracer,
                        callbacks=callbacks,
                    )

                    tool_content = tool_result.content[0]
                    if not isinstance(tool_content, TextContent):
                        raise ValueError("Tool output is not a text")

                    result_text = tool_content.text.strip()

                    if self._config.summarize_tool_response:
                        context = json.dumps({
                            "server": server_name,
                            "tool": tool_name,
                            "arguments": arguments,
                        }, indent=2)
                        result_text = await self.summarize_tool_response(
                            result_text,
                            context=context,
                            tracer=tracer,
                        )

                    self._add_history(history_type="result", message=result_text)

                    # Feed result back as user message
                    messages.append({
                        "role": "user",
                        "content": result_text,
                    })

                    await send_message_async(
                        callbacks,
                        message=CallbackMessage(
                            source=__file__,
                            type=MessageType.LOG,
                            metadata={
                                "event": "plain_text",
                                "data": "".join([
                                    f"{'=' * 66}\n",
                                    f"Iteration: {iter_num + 1}\n",
                                    f"{'-' * 66}\n",
                                    f"\033[31mAction: {tool_name} on {server_name}\n\n\033[0m",
                                    f"\033[33mResult: {result_text}\n\033[0m",
                                ]),
                            },
                        ),
                    )

                except Exception as e:
                    error_msg = str(e)[:300]
                    self._add_history(history_type="result", message=error_msg)
                    messages.append({
                        "role": "user",
                        "content": f"Error: {error_msg}",
                    })

                continue

            # --- No tool call: treat as content response ---
            result = await self._handle_content_response(
                cleaned_text, messages, iter_num, callbacks, tracer,
            )
            if result is not None:
                return result

        # Exceeded max iterations
        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response=(
                "I'm sorry, but I couldn't find a satisfactory answer within the "
                "allowed number of iterations."
            ),
            trace_id=tracer.trace_id,
        )
