# Copyright 2024 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Kubeflow MCP server setup."""

from unittest.mock import patch

from kubeflow.mcp.server import _load_instructions, create_server


class TestLoadInstructions:
    """Tests for the _load_instructions helper."""

    def test_instructions_file_exists_and_loads(self):
        """Test that the instructions.md file exists and can be loaded."""
        instructions = _load_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_instructions_contains_expected_content(self):
        """Test that the instructions contain key workflow descriptions."""
        instructions = _load_instructions()
        assert "Kubeflow MCP Server" in instructions
        assert "list_training_jobs" in instructions
        assert "get_training_job" in instructions
        assert "get_training_logs" in instructions


class TestCreateServer:
    """Tests for the create_server factory function."""

    @patch("kubeflow.mcp.server.FastMCP")
    def test_creates_server_with_default_name(self, mock_fastmcp):
        """Test that create_server creates a FastMCP server with default name."""
        create_server()
        mock_fastmcp.assert_called_once()
        args, kwargs = mock_fastmcp.call_args
        assert args[0] == "kubeflow-mcp"
        assert "instructions" in kwargs

    @patch("kubeflow.mcp.server.FastMCP")
    def test_creates_server_with_custom_name(self, mock_fastmcp):
        """Test that create_server accepts a custom server name."""
        create_server(name="my-custom-server")
        args, kwargs = mock_fastmcp.call_args
        assert args[0] == "my-custom-server"

    @patch("kubeflow.mcp.server.FastMCP")
    def test_server_loads_instructions(self, mock_fastmcp):
        """Test that the server is created with file-backed instructions."""
        create_server()
        _, kwargs = mock_fastmcp.call_args
        instructions = kwargs["instructions"]
        assert "Kubeflow MCP Server" in instructions

    @patch("kubeflow.mcp.server.FastMCP")
    def test_trainer_tools_are_registered(self, mock_fastmcp):
        """Test that trainer tools are registered on the server."""
        mock_server = mock_fastmcp.return_value
        create_server()
        # The tool() decorator should have been called 5 times
        # (one for each trainer discovery tool)
        assert mock_server.tool.call_count == 5
