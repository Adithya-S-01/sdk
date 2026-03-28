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

"""Tests for the MCP trainer discovery tools."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from kubeflow.mcp.tools.trainer import _dataclass_to_dict, register_trainer_tools
from kubeflow.trainer.types import types

# --- Helper fixtures ---


def _make_step(name="node-0", status="Running", pod_name="pod-abc"):
    return types.Step(name=name, status=status, pod_name=pod_name)


def _make_runtime(name="torch-distributed"):
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        image="pytorch/pytorch:2.0",
    )
    return types.Runtime(name=name, trainer=trainer)


def _make_train_job(name="test-job", status="Running"):
    return types.TrainJob(
        name=name,
        runtime=_make_runtime(),
        steps=[_make_step()],
        num_nodes=1,
        creation_timestamp=datetime(2026, 3, 28, 12, 0, 0),
        status=status,
    )


def _make_event(
    kind="TrainJob",
    obj_name="test-job",
    message="Job started",
    reason="Started",
):
    return types.Event(
        involved_object_kind=kind,
        involved_object_name=obj_name,
        message=message,
        reason=reason,
        event_time=datetime(2026, 3, 28, 12, 0, 0),
    )


# --- Tests for _dataclass_to_dict ---


class TestDataclassToDict:
    """Tests for the _dataclass_to_dict serialization helper."""

    def test_converts_simple_dataclass(self):
        step = _make_step()
        result = _dataclass_to_dict(step)
        assert isinstance(result, dict)
        assert result["name"] == "node-0"
        assert result["status"] == "Running"
        assert result["pod_name"] == "pod-abc"

    def test_converts_nested_dataclass(self):
        job = _make_train_job()
        result = _dataclass_to_dict(job)
        assert isinstance(result, dict)
        assert result["name"] == "test-job"
        assert isinstance(result["runtime"], dict)
        assert result["runtime"]["name"] == "torch-distributed"
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) == 1

    def test_converts_datetime_to_iso(self):
        job = _make_train_job()
        result = _dataclass_to_dict(job)
        assert result["creation_timestamp"] == "2026-03-28T12:00:00"

    def test_converts_list_of_dataclasses(self):
        jobs = [_make_train_job("job-1"), _make_train_job("job-2")]
        result = _dataclass_to_dict(jobs)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "job-1"

    def test_handles_none_values(self):
        runtime = _make_runtime()
        result = _dataclass_to_dict(runtime)
        assert result["pretrained_model"] is None

    def test_passes_through_primitives(self):
        assert _dataclass_to_dict("hello") == "hello"
        assert _dataclass_to_dict(42) == 42
        assert _dataclass_to_dict(True) is True


# --- Tests for trainer tools ---


class TestListTrainingJobs:
    """Tests for the list_training_jobs MCP tool."""

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_jobs_list(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [
            _make_train_job("job-1"),
            _make_train_job("job-2"),
        ]
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["list_training_jobs"]()
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["jobs"]) == 2
        assert result["jobs"][0]["name"] == "job-1"

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_empty_list(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = []
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["list_training_jobs"]()
        assert result["success"] is True
        assert result["count"] == 0

    @patch(
        "kubeflow.mcp.tools.trainer.TrainerClient",
        side_effect=RuntimeError("Connection failed"),
    )
    def test_handles_error(self, mock_client_cls):
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["list_training_jobs"]()
        assert result["success"] is False
        assert "Connection failed" in result["error"]


class TestGetTrainingJob:
    """Tests for the get_training_job MCP tool."""

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_job_details(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_job.return_value = _make_train_job("my-job", "Complete")
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["get_training_job"](name="my-job")
        assert result["success"] is True
        assert result["job"]["name"] == "my-job"
        assert result["job"]["status"] == "Complete"

    @patch(
        "kubeflow.mcp.tools.trainer.TrainerClient",
        side_effect=RuntimeError("Job not found"),
    )
    def test_handles_not_found(self, mock_client_cls):
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["get_training_job"](name="nonexistent")
        assert result["success"] is False
        assert "Job not found" in result["error"]


class TestGetTrainingLogs:
    """Tests for the get_training_logs MCP tool."""

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_logs(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter(["Epoch 1/3", "Loss: 0.5", "Epoch 2/3"])
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["get_training_logs"](name="my-job")
        assert result["success"] is True
        assert result["line_count"] == 3
        assert "Epoch 1/3" in result["logs"]


class TestGetTrainingEvents:
    """Tests for the get_training_events MCP tool."""

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_events(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_job_events.return_value = [
            _make_event(message="Job created"),
            _make_event(message="Pod scheduled"),
        ]
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["get_training_events"](name="my-job")
        assert result["success"] is True
        assert result["count"] == 2
        assert result["events"][0]["message"] == "Job created"


class TestListRuntimes:
    """Tests for the list_runtimes MCP tool."""

    @patch("kubeflow.mcp.tools.trainer.TrainerClient")
    def test_returns_runtimes(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_runtimes.return_value = [
            _make_runtime("torch-distributed"),
            _make_runtime("mpi-distributed"),
        ]
        mock_client_cls.return_value = mock_client

        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["list_runtimes"]()
        assert result["success"] is True
        assert result["count"] == 2
        assert result["runtimes"][0]["name"] == "torch-distributed"

    @patch(
        "kubeflow.mcp.tools.trainer.TrainerClient",
        side_effect=TimeoutError("Cluster unreachable"),
    )
    def test_handles_timeout(self, mock_client_cls):
        mcp = MagicMock()
        tools = {}

        def capture_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_trainer_tools(mcp)

        result = tools["list_runtimes"]()
        assert result["success"] is False
        assert "Cluster unreachable" in result["error"]
