import os
import unittest
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.agent.utils import (
    SUMMARIZE_PROMPT,
    build_system_prompt,
    get_iteration_prompt_content,
    get_tools_description,
)


class TestAgentUtils(unittest.IsolatedAsyncioTestCase):

    async def test_get_tools_description(self):
        manager = MCPManager()
        client = await manager.build_client(server_name="weather", transport="stdio")
        tools = await client.list_tools()
        description = get_tools_description({"weather": tools})
        print(description)
        await client.cleanup()

    async def test_build_system_prompt(self):
        manager = MCPManager()
        folder = os.path.dirname(os.path.realpath(__file__))
        system_prompt_template = os.path.join(folder, "../../mcpuniverse/agent/configs/system_prompt.j2")
        tools_prompt_template = os.path.join(folder, "../../mcpuniverse/agent/configs/tools_prompt.j2")
        client = await manager.build_client(server_name="weather", transport="stdio")
        tools = await client.list_tools()
        system_prompt = build_system_prompt(
            system_prompt_template, tools_prompt_template, {"weather": tools},
            INSTRUCTION="You are a weather agent"
        )
        print(system_prompt)
        await client.cleanup()

    async def test_get_iteration_prompt_content_can_disable_countdown(self):
        self.assertIsNone(
            get_iteration_prompt_content(
                iter_num=1,
                max_iterations=5,
                scheduler_mode=None,
                append_iteration_user_message=False,
            )
        )
        self.assertEqual(
            get_iteration_prompt_content(
                iter_num=4,
                max_iterations=5,
                scheduler_mode=None,
                append_iteration_user_message=False,
            ),
            SUMMARIZE_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
