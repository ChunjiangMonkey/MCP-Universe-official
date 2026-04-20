"""
A function calling agent implementation with function calling support.
This module contains the function calling agent class and its configuration.
This implementation uses LLM native function calling instead of text-based tool calling.
"""
# pylint: disable=broad-exception-caught
import os
import re
import json
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass
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
from .utils import build_system_prompt
from .types import AgentResponse

DEFAULT_CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

CUSTOM_TOOL_ANSWER = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Report the final answer or result summary for the completed task.\n\n        Call this tool ONCE at the very end, after you have fully completed the task. Provide a clear, complete summary of your findings or results.\n\n        Args:\n            content: A clear, complete summary of the task result or answer.\n\n        Returns:\n            Confirmation message.\n        ",
        "parameters": {
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "type": "object"
        }
    }
}

CUSTOM_TOOL_WRITE_TODOS = {
    "type": "function",
    "function": {
        "name": "write_todos",
        "description": "Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\n\nOnly use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.\n\n## When to Use This Tool\nUse this tool in these scenarios:\n\n1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n3. User explicitly requests todo list - When the user directly asks you to use the todo list\n4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n5. The plan may need future revisions or updates based on results from the first few steps\n\n## How to Use This Tool\n1. When you start working on a task - Mark it as in_progress BEFORE beginning work.\n2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.\n3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.\n4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.\n\n## When NOT to Use This Tool\nIt is important to skip using this tool when:\n1. There is only a single, straightforward task\n2. The task is trivial and tracking it provides no benefit\n3. The task can be completed in less than 3 trivial steps\n4. The task is purely conversational or informational\n\n## Task States and Management\n\n1. **Task States**: Use these states to track progress:\n   - pending: Task not yet started\n   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)\n   - completed: Task finished successfully\n\n2. **Task Management**:\n   - Update task status in real-time as you work\n   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n   - Complete current tasks before starting new ones\n   - Remove tasks that are no longer relevant from the list entirely\n   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.\n   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.\n\n3. **Task Completion Requirements**:\n   - ONLY mark a task as completed when you have FULLY accomplished it\n   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n   - When blocked, create a new task describing what needs to be resolved\n   - Never mark a task as completed if:\n     - There are unresolved issues or errors\n     - Work is partial or incomplete\n     - You encountered blockers that prevent completion\n     - You couldn't find necessary resources or dependencies\n     - Quality standards haven't been met\n\n4. **Task Breakdown**:\n   - Create specific, actionable items\n   - Break complex tasks into smaller, manageable steps\n   - Use clear, descriptive task names\n\nBeing proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully\nRemember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.",
        "parameters": {
            "properties": {
                "todos": {
                    "items": {
                        "description": "A single todo item with content and status.",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "enum": ["pending", "in_progress", "completed"],
                                "type": "string"
                            }
                        },
                        "required": ["content", "status"],
                        "type": "object"
                    },
                    "type": "array"
                },
                "tool_call_id": {"type": "string"}
            },
            "required": ["todos", "tool_call_id"],
            "type": "object"
        }
    }
}

CUSTOM_TOOL_NAMES = {"answer", "write_todos"}


@dataclass
class FunctionCallConfig(BaseAgentConfig):
    """
    Configuration class for ReAct agents with function calling.

    Attributes:
        system_prompt (str): The system prompt template file or string.
        context_examples (str): Additional context examples for the agent.
        max_iterations (int): Maximum number of reasoning iterations.
        summarize_tool_response (bool): Whether to summarize tool responses using the LLM.
    """
    system_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "function_call_prompt.j2")
    context_examples: str = ""
    max_iterations: int = 5
    summarize_tool_response: bool = False
    use_custom_tools: bool = False
    append_iteration_user_message: bool = True


class FunctionCall(BaseAgent):
    """
    Function calling agent implementation with function calling support.

    This class implements the function calling paradigm,
    allowing the agent to alternate between reasoning and acting to solve tasks.
    It uses LLM native function calling instead of text-based tool descriptions.

    Attributes:
        config_class (Type[FunctionCallConfig]): The configuration class for this agent.
        alias (List[str]): Alternative names for this agent type.
    """
    config_class = FunctionCallConfig
    alias = ["function_call", "fc", "function-call"]

    def __init__(
            self,
            mcp_manager: MCPManager,
            llm: BaseLLM,
            config: Optional[Union[Dict, str]] = None
    ):
        """
        Initialize a function calling agent.

        Args:
            mcp_manager (MCPManager): An MCP server manager for handling tool interactions.
            llm (BaseLLM): A language model for generating responses.
            config (Optional[Union[Dict, str]]): Agent configuration as a dictionary or file path.
        """
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._logger = get_logger(f"{self.__class__.__name__}:{self._name}")
        self._history: List[str] = []

    def _convert_mcp_tools_to_function_calls(
        self, tools: Dict[str, List[Tool]]
    ) -> List[Dict[str, Any]]:
        """
        Convert MCP tools to function call format for LLM.

        Args:
            tools (Dict[str, List[Tool]]): MCP tools organized by server name

        Returns:
            List[Dict[str, Any]]: Tools in function call format
        """
        function_calls = []
        for server_name, tool_list in tools.items():
            for tool in tool_list:
                function_call = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool.name}",
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                function_calls.append(function_call)
        return function_calls

    def _parse_function_call_name(self, function_name: str) -> tuple[str, str]:
        """
        Parse function call name to extract server and tool names.

        Args:
            function_name (str): Function name in format "server__tool"

        Returns:
            tuple[str, str]: (server_name, tool_name)
        """
        if "__" in function_name:
            return function_name.split("__", 1)
        # Fallback: try to find the tool in any server
        for server_name, tool_list in self._tools.items():
            for tool in tool_list:
                if tool.name == function_name:
                    return server_name, tool.name
        raise ValueError(f"Cannot parse function name: {function_name}")

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from text that might contain additional content before or after the JSON.

        Args:
            text (str): Text that may contain JSON

        Returns:
            str: Extracted JSON string

        Raises:
            ValueError: If no valid JSON is found in the text
        """
        # First, try to parse the text as-is (it might already be pure JSON)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to find JSON objects or arrays in the text
        # Look for patterns that start with { or [ and have matching closing brackets
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Simple nested arrays
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',  # More complex nested objects
            r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'  # More complex nested arrays
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Validate that this is actually valid JSON
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue

        # If no JSON patterns worked, try to find content between first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            candidate = text[first_brace:last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try the same for arrays
        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            candidate = text[first_bracket:last_bracket + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # If all else fails, return the original text and let json.loads handle the error
        return text

    def _build_prompt(self, question: str):
        """
        Construct the prompt for the language model.

        Args:
            question (str): The user's question or task.

        Returns:
            str: The constructed prompt including system instructions, context, and history.
        """
        params = {
            "INSTRUCTION": self._config.instruction,
            "QUESTION": question,
            "MAX_STEPS": self._config.max_iterations
        }
        if self._config.context_examples:
            params.update({"CONTEXT_EXAMPLES": self._config.context_examples})
        params.update(self._config.template_vars)
        # Note: HISTORY is no longer included in prompt as we use conversation format
        # Don't include tools in the prompt since we use function calling
        system_prompt = self._config.system_prompt
        if self._config.use_custom_tools:
            system_prompt = os.path.join(
                DEFAULT_CONFIG_FOLDER, "custom_function_call_prompt.j2"
            )
        return build_system_prompt(
            system_prompt_template=system_prompt,
            tool_prompt_template="",  # No tool prompt needed
            tools=None,  # No tools in text format
            **params
        )

    def _add_history(self, history_type: str, message: str):
        """
        Add a record to the agent's conversation history.

        Args:
            history_type (str): The type of the history entry (e.g., "thought", "action", "result").
            message (str): The content of the history entry.
        """
        self._history.append(f"{history_type.title()}: {message}")

    async def _handle_none_response(
        self, iter_num: int, callbacks: List[Any], tracer: Tracer
    ) -> AgentResponse:
        """Handle case where LLM returns None response."""
        self._logger.error("LLM returned None response, stopping execution")
        error_msg = (
            "The language model encountered a critical error and couldn't generate a response. "
            "This may be due to API failures, context length limits, or other issues."
        )
        self._add_history(
            history_type="error",
            message=error_msg
        )

        # Send error message to callbacks
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
                        "\\033[31mCritical Error: LLM returned None response\\n\\033[0m",
                        f"\\033[33mDetails: {error_msg}\\n\\033[0m"
                    ])
                }
            )
        )

        # Return immediately instead of continuing iterations
        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response=error_msg,
            trace_id=tracer.trace_id
        )

    async def _handle_content_response(
        self,
        content: str,
        messages: List[Dict[str, Any]],
        iter_num: int,
        callbacks: List[Any],
        tracer: Tracer
    ) -> Optional[AgentResponse]:
        """Handle content-based responses from LLM."""
        try:
            # Add content as assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": content
            })

            # Try to parse as JSON response
            # Handle cases where JSON might be embedded in other text
            response_text = content.strip().strip('`').strip()

            # Remove "json" prefix if present
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

            # Try to extract JSON from the text
            json_text = self._extract_json_from_text(response_text)
            parsed_response = json.loads(json_text)

            # Check if this is a final answer
            if "answer" in parsed_response:
                self._add_history(
                    history_type="answer",
                    message=parsed_response["answer"]
                )
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
                                f"\\033[32mThought: {parsed_response['thought']}\\n\\n\\033[0m",
                                f"\\033[31mAnswer: {parsed_response['answer']}\\n\\033[0m"
                            ])
                        }
                    )
                )
                return AgentResponse(
                    name=self._name,
                    class_name=self.__class__.__name__,
                    response=parsed_response["answer"],
                    trace_id=tracer.trace_id
                )

            # If no answer field, treat entire response as thought
            self._add_history(
                history_type="thought",
                message=content
            )
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
                            f"\\033[32mThought: {content}\\n\\033[0m"
                        ])
                    }
                )
            )
            # Continue to next iteration
            return None

        except json.JSONDecodeError as e:
            self._logger.error("Failed to parse response: %s", str(e))
            error_msg = (
                "Encountered an error in parsing LLM response:\n"
                f"{content}\n\n"
                "Please try again."
            )
            self._add_history(
                history_type="error",
                message=error_msg
            )
            # Add error as user message to guide the next iteration
            messages.append({
                "role": "user",
                "content": error_msg
            })
            return None
        except Exception as e:
            self._logger.error("Failed to process response: %s", str(e))
            error_msg = (f"Encountered an unexpected error for the LLM response:\\n"
                       f"{content}.\\n\\nPlease try again.")
            self._add_history(
                history_type="error",
                message=error_msg
            )
            # Add error as user message to guide the next iteration
            messages.append({
                "role": "user",
                "content": error_msg
            })
            return None

    async def _execute(
            self,
            message: Union[str, List[str]],
            output_format: Optional[Union[str, Dict]] = None,
            **kwargs
    ) -> AgentResponse:
        """
        Execute the function calling agent's reasoning and action loop using function calling.

        This method processes the user's message, generates thoughts and actions using
        LLM native function calling, and returns a final answer or explanation.

        Args:
            message (Union[str, List[str]]): The user's message or a list of messages.
            output_format (Optional[Union[str, Dict]]): Desired format for the output.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentResponse: The agent's final response, including the answer and trace information.
        """
        if isinstance(message, (list, tuple)):
            message = "\n".join(message)
        if output_format is not None:
            message += f"""
            {self._get_output_format_prompt(output_format)}"""
        tracer = kwargs.get("tracer", Tracer())
        callbacks = kwargs.get("callbacks", [])

        # Convert MCP tools to function call format
        tools = self._convert_mcp_tools_to_function_calls(self._tools) if self._tools else []
        if self._config.use_custom_tools:
            tools.extend([CUSTOM_TOOL_ANSWER, CUSTOM_TOOL_WRITE_TODOS])

        # Build initial system prompt (without history)
        initial_prompt = self._build_prompt(message)

        # Initialize messages with system-like user message
        messages = [{"role": "user", "content": initial_prompt}]

        for iter_num in range(self._config.max_iterations):
            # Add step counter only if not the first iteration to avoid cluttering
            if self._config.append_iteration_user_message and iter_num > 0:
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have {self._config.max_iterations - iter_num} steps remaining. "
                        "Please continue."
                    )
                })

            self._add_history(
                history_type=f"Step {iter_num + 1}",
                message="",
            )

            # Generate response with function calling support
            if tools:
                response = self._llm.generate(
                    messages=messages,
                    tools=tools,
                    tracer=tracer,
                    callbacks=callbacks
                )
            else:
                response = self._llm.generate(
                    messages=messages,
                    tracer=tracer,
                    callbacks=callbacks
                )

            # Handle different types of responses
            if response is None:
                return await self._handle_none_response(iter_num, callbacks, tracer)
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                message_obj = choice.message

                # Handle responses that may contain both tool_calls and content (e.g., GPT-5)
                has_tool_calls = hasattr(message_obj, 'tool_calls') and message_obj.tool_calls
                has_content = hasattr(message_obj, 'content') and message_obj.content
                # for official kimi-k2-thinking
                has_reasoning_content = (
                    hasattr(message_obj, 'reasoning_content') and
                    message_obj.reasoning_content
                )
                # for openrouter
                has_reasoning_details = (
                    hasattr(message_obj, 'reasoning_details') and
                    message_obj.reasoning_details
                )

                if has_reasoning_content:
                    content = message_obj.reasoning_content.strip()
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                    if content:
                        self._add_history(
                            history_type="reasoning_content",
                            message=f"LLM reasoning: {content}"
                        )
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
                                        f"\033[32mThought with reasoning: {content}\n\033[0m"
                                    ])
                                }
                            )
                        )

                if has_reasoning_details:
                    reasoning_details = message_obj.reasoning_details
                    # Extract text from reasoning_details (list of dicts)
                    reasoning_text = ""
                    if isinstance(reasoning_details, list):
                        for item in reasoning_details:
                            if isinstance(item, dict) and item.get("type") == "reasoning.text":
                                reasoning_text = item.get("text", "")
                                break
                    elif hasattr(reasoning_details, "text"):
                        reasoning_text = reasoning_details.text

                    if reasoning_text:
                        self._add_history(
                            history_type="reasoning_details",
                            message=f"LLM reasoning details: {reasoning_text}"
                        )
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
                                        f"\033[32mReasoning details: {reasoning_text}\n\033[0m"
                                    ])
                                }
                            )
                        )

                if has_tool_calls:
                    # Handle function calls first
                    answer_response = await self._handle_function_calls(
                        message_obj.tool_calls,
                        message_obj.reasoning_details if has_reasoning_details else None,
                        messages,
                        iter_num,
                        tracer,
                        callbacks
                    )
                    if answer_response is not None:
                        return answer_response
                    # If there's also content, process it as additional thought/reasoning
                    if has_content:
                        content = message_obj.content.strip()
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                        if content:
                            self._add_history(
                                history_type="thought",
                                message=f"LLM reasoning with tool calls: {content}"
                            )
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
                                            f"\033[32mThought with tools: {content}\n\033[0m"
                                        ])
                                    }
                                )
                            )
                    continue
                if has_content:
                    content = message_obj.content.strip()
                    result = await self._handle_content_response(
                        content, messages, iter_num, callbacks, tracer
                    )
                    if result is not None:
                        return result
                    continue

                if not has_tool_calls and not has_content and not has_reasoning_content:
                    # Handle case where message has no content or tool_calls
                    self._logger.warning("Received message with no content or tool_calls")
                    error_msg = "Received an empty response from the LLM. Please try again."
                    self._add_history(
                        history_type="error",
                        message=error_msg
                    )
                    # Add error as user message to guide the next iteration
                    messages.append({
                        "role": "user",
                        "content": error_msg
                    })
            elif isinstance(response, str):
                # Fallback for string responses
                content = response.strip()
                if content:
                    self._add_history(
                        history_type="answer",
                        message=content
                    )
                    return AgentResponse(
                        name=self._name,
                        class_name=self.__class__.__name__,
                        response=content,
                        trace_id=tracer.trace_id
                    )
            else:
                # Handle unexpected response types
                self._logger.error("Received unexpected response type: %s", type(response))
                error_msg = "Received an unexpected response format. Please try again."
                self._add_history(
                    history_type="error",
                    message=error_msg
                )
                # Add error as user message to guide the next iteration
                messages.append({
                    "role": "user",
                    "content": error_msg
                })

        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response=(
                "I'm sorry, but I couldn't find a satisfactory answer within the "
                "allowed number of iterations."
            ),
            trace_id=tracer.trace_id
        )

    async def _handle_function_calls(
            self,
            tool_calls: List[Any],
            reasoning_details: List[Any],
            messages: List[Dict[str, str]],
            iter_num: int,
            tracer: Tracer,
            callbacks: List[Any]
    ) -> Optional[AgentResponse]:
        """
        Handle function calls from the LLM response.

        Args:
            tool_calls: List of tool calls from the LLM
            reasoning_details: Reasoning details from the LLM
            messages: Conversation messages list to update
            iter_num: Current iteration number
            tracer: Tracer for logging
            callbacks: Callbacks for logging

        Returns:
            Optional[AgentResponse]: An AgentResponse if the 'answer' custom tool
                was called, None otherwise.
        """
        # Add assistant message with tool calls to conversation
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id,
                    "type": "function",
                    "function": {
                        "name": (tool_call.get("function", {}).get("name")
                                if isinstance(tool_call, dict)
                                else tool_call.function.name),
                        "arguments": (tool_call.get("function", {}).get("arguments")
                                     if isinstance(tool_call, dict)
                                     else tool_call.function.arguments)
                    }
                } for tool_call in tool_calls
            ]
        }
        if reasoning_details:
            assistant_message["reasoning_details"] = reasoning_details
        messages.append(assistant_message)

        # Execute each tool call
        for tool_call in tool_calls:
            try:
                # Handle both dict and object formats for tool_call
                if isinstance(tool_call, dict):
                    function_name = tool_call.get("function", {}).get("name")
                    function_arguments = tool_call.get("function", {}).get("arguments")
                    tool_call_id = tool_call.get("id")
                else:
                    function_name = tool_call.function.name
                    function_arguments = tool_call.function.arguments
                    tool_call_id = tool_call.id

                # Intercept custom tools (answer, write_todos)
                if (self._config.use_custom_tools
                        and function_name in CUSTOM_TOOL_NAMES):
                    # Parse arguments
                    if isinstance(function_arguments, str):
                        arguments = json.loads(function_arguments)
                    else:
                        arguments = function_arguments

                    self._add_history(
                        history_type="action",
                        message=f"Using custom tool `{function_name}`"
                    )
                    self._add_history(
                        history_type="action input",
                        message=str(arguments)
                    )

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
                                    f"\033[31mCustom Tool: {function_name}\n\n\033[0m",
                                    f"\033[33mArguments: {json.dumps(arguments)}\n\033[0m",
                                ])
                            }
                        )
                    )

                    if function_name == "answer":
                        answer_content = arguments.get("content", "")
                        self._add_history(
                            history_type="answer",
                            message=answer_content
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Answer recorded."
                        })
                        return AgentResponse(
                            name=self._name,
                            class_name=self.__class__.__name__,
                            response=answer_content,
                            trace_id=tracer.trace_id
                        )
                    elif function_name == "write_todos":
                        self._add_history(
                            history_type="result",
                            message="Todos updated successfully."
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Todos updated successfully."
                        })
                        continue

                # Parse function name to get server and tool names
                server_name, tool_name = self._parse_function_call_name(function_name)

                # Parse arguments
                if isinstance(function_arguments, str):
                    arguments = json.loads(function_arguments)
                else:
                    arguments = function_arguments

                self._add_history(
                    history_type="action",
                    message=f"Using tool `{tool_name}` in server `{server_name}`"
                )
                self._add_history(
                    history_type="action input",
                    message=str(arguments)
                )

                # Execute the tool
                tool_result = await self.call_tool(
                    {
                        "server": server_name,
                        "tool": tool_name,
                        "arguments": arguments
                    },
                    tracer=tracer,
                    callbacks=callbacks
                )

                tool_content = tool_result.content[0]
                if not isinstance(tool_content, TextContent):
                    raise ValueError("Tool output is not a text")

                result_text = tool_content.text.strip()
                if self._config.summarize_tool_response:
                    context = json.dumps({
                        "server": server_name,
                        "tool": tool_name,
                        "arguments": arguments
                    }, indent=2)
                    result_text = self.summarize_tool_response(
                        result_text,
                        context=context,
                        tracer=tracer
                    )

                self._add_history(history_type="result", message=result_text)

                # Add tool result to conversation
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text.strip()
                }
                messages.append(tool_message)

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
                            ])
                        }
                    )
                )

            except Exception as e:
                error_msg = str(e)[:300]
                self._add_history(history_type="result", message=error_msg)

                # Add error result to conversation
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: {error_msg}"
                }
                messages.append(tool_message)

        return None

    def get_history(self) -> str:
        """
        Retrieve the agent's conversation history.

        Returns:
            str: A string representation of the agent's conversation history.
        """
        return "\n".join(self._history)

    def clear_history(self):
        """
        Clear the agent's conversation history.
        """
        self._history = []

    def reset(self):
        """Reset the agent."""
        self.clear_history()
