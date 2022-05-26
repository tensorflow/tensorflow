# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Python xla::Status casters."""

from absl.testing import absltest

from tensorflow.compiler.xla.python import status_casters_example
from tensorflow.compiler.xla.python import xla_client


class StatusCastersTest(absltest.TestCase):

  def test_xla_runtime_error(self):
    with self.assertRaises(xla_client.XlaRuntimeError):
      status_casters_example.raise_xla_status()

    with self.assertRaises(xla_client.XlaRuntimeError):
      status_casters_example.raise_xla_status_or()

  def test_xla_test_error(self):
    with self.assertRaises(status_casters_example.XlaTestError):
      status_casters_example.raise_xla_status_with_xla_test_error()

    with self.assertRaises(status_casters_example.XlaTestError):
      status_casters_example.raise_xla_status_or_with_xla_test_error()


if __name__ == '__main__':
  absltest.main()
