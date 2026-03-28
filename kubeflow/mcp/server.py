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

"""Kubeflow MCP Server - AI Model Training on Kubernetes.

This module provides a Model Context Protocol (MCP) server that wraps the
Kubeflow SDK, enabling LLM agents to interact with Kubeflow training
infrastructure through standardized tool interfaces.

Related KEP: https://github.com/kubeflow/community/pull/937
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from kubeflow.mcp.tools.trainer import register_trainer_tools

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_instructions() -> str:
    """Load system instructions from the prompts directory."""
    instructions_path = _PROMPTS_DIR / "instructions.md"
    return instructions_path.read_text(encoding="utf-8")


def create_server(name: str = "kubeflow-mcp") -> FastMCP:
    """Create and configure the Kubeflow MCP server.

    Sets up a FastMCP server with trainer discovery tools and file-backed
    system instructions.

    Args:
        name: The name for the MCP server instance.

    Returns:
        A configured FastMCP server ready to run.
    """
    instructions = _load_instructions()

    mcp = FastMCP(name, instructions=instructions)

    # Register trainer discovery tools
    register_trainer_tools(mcp)

    logger.info("Kubeflow MCP server '%s' created with trainer tools registered.", name)
    return mcp


# Allow running the server directly: python -m kubeflow.mcp.server
if __name__ == "__main__":
    server = create_server()
    server.run()
