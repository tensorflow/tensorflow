# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests and examples for status_casters.h."""

from absl.testing import absltest

from tensorflow.compiler.xla.python import status_casters_ext


class StatusCastersTest(absltest.TestCase):

  def test_status_wrappers(self):
    self.assertIsNone(status_casters_ext.my_lambda())
    self.assertIsNone(status_casters_ext.my_lambda2())

    self.assertIsNone(status_casters_ext.MyClass().my_method(1, 2))
    self.assertIsNone(status_casters_ext.MyClass().my_method_const(1, 2))

  def test_status_or_wrappers(self):
    self.assertEqual(status_casters_ext.my_lambda_statusor(), 1)
    self.assertEqual(status_casters_ext.status_or_identity(2), 2)

    self.assertEqual(status_casters_ext.MyClass().my_method_status_or(1, 2), 3)
    self.assertEqual(
        status_casters_ext.MyClass().my_method_status_or_const(1, 2), 3
    )


if __name__ == "__main__":
  absltest.main()
