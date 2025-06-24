# Copyright 2025 The OpenXLA Authors.
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
# ==============================================================================

import io
import os

from absl.testing import absltest

from xla.hlo.tools import generate_hlo_test_checks


class GenerateHloTestChecksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._data_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(self._data_dir, os.pardir))
    self._optimizer_path = os.path.join(parent_dir, "hlo-opt")
    self._input_file_path = os.path.join(
        self._data_dir, "generate_hlo_test_checks_test_input.hlo"
    )
    self._output_file_path = os.path.join(
        self._data_dir, "generate_hlo_test_checks_test_output.hlo"
    )

  def test_parallel_mode_generates_expected_output_file(self):
    actual_output = io.StringIO()

    # The `worker_count` argument is normally inferred, but we specify it here
    # in order to ensure that we're testing the parallel code path even in the
    # unlikely event that `os.process_cpu_count() == 1` on the test machine.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "{}",
            "--passes=logistic-expander,reshape-mover",
        ],
        worker_count=8,
    ) as writer:
      writer.transform_and_print_file(
          self._input_file_path,
          output_stream=actual_output,
      )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_sequential_mode_generates_expected_output_file(self):
    actual_output = io.StringIO()

    writer = generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "{}",
            "--passes=logistic-expander,reshape-mover",
        ],
    )

    writer.transform_and_print_file(
        self._input_file_path,
        output_stream=actual_output,
    )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_custom_input_file_placeholder_string(self):
    actual_output = io.StringIO()

    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=[
            "%s",
            "--passes=logistic-expander,reshape-mover",
        ],
        expand_to_input="%s",
    ) as writer:
      writer.transform_and_print_file(
          self._input_file_path,
          output_stream=actual_output,
      )

    actual_output.seek(0)

    with open(self._output_file_path, mode="r") as expected_output:
      self.assertEqual(actual_output.read(), expected_output.read())

  def test_argument_parsing(self):
    args = generate_hlo_test_checks.parse_args(
        "/path/to/test_file.hlo -- /path/to/hlo-opt {} --passes=foo,bar".split()
    )

    self.assertEqual(args.test_file, "/path/to/test_file.hlo")
    self.assertEqual(args.in_place, False)
    self.assertEqual(args.expand_to_input, "{}")
    self.assertEqual(args.opt_cmd, "/path/to/hlo-opt")
    self.assertEqual(args.opt_args, ["{}", "--passes=foo,bar"])

    args = generate_hlo_test_checks.parse_args(
        "test_file.hlo -i -I%s -- hlo-opt %s --passes=foo,bar".split()
    )

    self.assertEqual(args.test_file, "test_file.hlo")
    self.assertEqual(args.in_place, True)
    self.assertEqual(args.expand_to_input, "%s")
    self.assertEqual(args.opt_cmd, "hlo-opt")
    self.assertEqual(args.opt_args, ["%s", "--passes=foo,bar"])

  def test_conditional_parallelism(self):
    # A writer with `worker_count > 1` should have a worker pool.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=2,
    ) as parallel_writer:
      self.assertIsNotNone(parallel_writer._worker_pool)

    # The worker pool should be destroyed when exiting the context manager.
    self.assertIsNone(parallel_writer._worker_pool)

    # A writer with `worker_count == 1` should not have a worker pool.
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=1,
    ) as sequential_writer:
      self.assertIsNone(sequential_writer._worker_pool)

    # Attempting to construct a writer with `worker_count < 1` should result in
    # a `ValueError`.
    with self.assertRaises(ValueError):
      with generate_hlo_test_checks.TestCheckWriter(
          optimizer_path=self._optimizer_path,
          optimizer_args=["{}"],
          worker_count=0,
      ):
        pass

  def test_unscoped_writer(self):
    # A writer with no active context manager shouldn't have a worker pool even
    # if `worker_count > 1`. This is because the worker pool is RAII-managed by
    # the context manager (i.e. the `with` block). In this situation, the writer
    # will operate sequentially as though `worker_count` were 1.
    writer = generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
        worker_count=4,
    )
    self.assertIsNone(writer._worker_pool)

    # However, entering a context manager for a previously constructed writer
    # with `worker_count > 1` should initialize a worker pool that lasts for the
    # duration of the context manager.
    with writer:
      self.assertIsNotNone(writer._worker_pool)

    # The worker pool should be destroyed when exiting the context manager.
    self.assertIsNone(writer._worker_pool)

  def test_prevent_conflicting_context_managers(self):
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
    ) as writer:
      # Creating multiple overlapping context managers for the same
      # `TestCheckWriter` instance should result in a `RuntimeError`.
      with self.assertRaises(RuntimeError):
        with writer:
          pass

  def test_allow_non_conflicting_context_managers(self):
    with generate_hlo_test_checks.TestCheckWriter(
        optimizer_path=self._optimizer_path,
        optimizer_args=["{}"],
    ) as writer:
      # Context managers for *separate* `TestCheckWriter` instances are allowed
      # to coexist since they're managing separate resources.
      with generate_hlo_test_checks.TestCheckWriter(
          optimizer_path=self._optimizer_path,
          optimizer_args=["{}"],
      ):
        pass

    # A given `TestCheckWriter` instance may be reused in multiple context
    # managers as long as their scopes never overlap with each other (i.e. the
    # resources are released before they're re-acquired).
    with writer:
      pass


if __name__ == "__main__":
  absltest.main()
