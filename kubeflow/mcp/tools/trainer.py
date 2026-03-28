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

"""MCP tools for the Kubeflow Trainer client.

Provides read-only discovery tools that wrap TrainerClient methods, enabling
LLM agents to inspect training jobs, runtimes, logs, and events.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import Enum
import logging

from mcp.server.fastmcp import FastMCP

from kubeflow.trainer import TrainerClient

logger = logging.getLogger(__name__)


def _dataclass_to_dict(obj: object) -> dict | list | object:
    """Recursively convert dataclass instances to JSON-serializable dicts.

    MCP tools must return JSON-serializable data. The Kubeflow SDK returns
    dataclass objects (TrainJob, Runtime, Event, Step), so we need to convert
    them before returning from a tool.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            try:
                value = getattr(obj, f.name)
            except AttributeError:
                # Skip fields that use Python name mangling (e.g. __command)
                # or haven't been initialized (init=False without default).
                continue
            result[f.name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj


def register_trainer_tools(mcp: FastMCP) -> None:
    """Register trainer discovery tools on the given MCP server.

    Args:
        mcp: The FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def list_training_jobs() -> dict:
        """List all TrainJobs in the Kubeflow cluster.

        Returns a list of training jobs with their names, statuses, runtimes,
        and creation timestamps.
        """
        try:
            client = TrainerClient()
            jobs = client.list_jobs()
            return {
                "success": True,
                "count": len(jobs),
                "jobs": [_dataclass_to_dict(job) for job in jobs],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def get_training_job(name: str) -> dict:
        """Get detailed status of a specific TrainJob.

        Args:
            name: The name of the TrainJob to retrieve.

        Returns the job's status, runtime configuration, steps, and metadata.
        """
        try:
            client = TrainerClient()
            job = client.get_job(name=name)
            return {"success": True, "job": _dataclass_to_dict(job)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def get_training_logs(
        name: str,
        step: str = "node-0",
        follow: bool = False,
    ) -> dict:
        """Get logs from a specific step of a TrainJob.

        Args:
            name: The name of the TrainJob.
            step: The step to collect logs from (e.g., 'node-0',
                  'dataset-initializer'). Defaults to 'node-0'.
            follow: Whether to include the latest logs. Defaults to False.

        Returns the log lines from the specified training step.
        """
        try:
            client = TrainerClient()
            log_lines = list(client.get_job_logs(name=name, step=step, follow=follow))
            return {
                "success": True,
                "job_name": name,
                "step": step,
                "line_count": len(log_lines),
                "logs": log_lines,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def get_training_events(name: str) -> dict:
        """Get Kubernetes events for a TrainJob.

        Useful for debugging when logs are empty or the pod hasn't started.
        Shows pod state changes, errors, and other significant occurrences.

        Args:
            name: The name of the TrainJob.

        Returns a list of events with timestamps, reasons, and messages.
        """
        try:
            client = TrainerClient()
            events = client.get_job_events(name=name)
            return {
                "success": True,
                "job_name": name,
                "count": len(events),
                "events": [_dataclass_to_dict(event) for event in events],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    def list_runtimes() -> dict:
        """List available training runtimes in the Kubeflow cluster.

        Returns the available runtimes with their trainer configurations,
        frameworks, and supported features.
        """
        try:
            client = TrainerClient()
            runtimes = client.list_runtimes()
            return {
                "success": True,
                "count": len(runtimes),
                "runtimes": [_dataclass_to_dict(rt) for rt in runtimes],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    logger.info("Registered 5 trainer discovery tools on MCP server.")
