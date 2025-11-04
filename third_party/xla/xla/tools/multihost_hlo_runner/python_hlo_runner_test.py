# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for running HLO files."""

import os
import pathlib

from absl.testing import absltest
from transformer_engine import transformer_engine_jax

from xla.tools.multihost_hlo_runner import py_hlo_multihost_runner


def _register_transformer_engine_custom_calls():
  for name, value in transformer_engine_jax.registrations().items():
    py_hlo_multihost_runner.register_custom_call_target(
        name, value, platform="CUDA", api_version=1
    )


def _get_test_hlo_path(file_name: str) -> str:
  """Returns the path to a HLO file in the data directory."""
  test_srcdir = pathlib.Path(os.environ["TEST_SRCDIR"])
  test_workspace = os.environ["TEST_WORKSPACE"]
  test_binary = os.environ["TEST_BINARY"]
  return os.path.join(
      os.path.dirname(test_srcdir / test_workspace / test_binary),
      "data",
      file_name,
  )


class RunTEHloTest(absltest.TestCase):
  """Tests for running custom calls from Transformer Engine."""

  def setUp(self):
    super().setUp()
    _register_transformer_engine_custom_calls()
    self.config = py_hlo_multihost_runner.PyHloRunnerConfig()
    self.config.input_format = py_hlo_multihost_runner.InputFormat.Text
    self.config.hlo_argument_mode = (
        py_hlo_multihost_runner.ModuleArgumentMode.Uninitialized
    )

  def test_run_custom_call_hlo(self):
    hlo_file = _get_test_hlo_path("transformer_engine_softmax.hlo")
    py_hlo_multihost_runner.RunHloFiles([hlo_file], self.config)


def main():
  absltest.main()


if __name__ == "__main__":
  main()
