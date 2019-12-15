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
"""Tests for the IgnoreErrors input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class IgnoreErrorsSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_ds(self):
    return dataset_ops.Dataset.range(5).map(
        array_ops.ones).map(lambda x: array_ops.gather(x, [0])).apply(
            error_ops.ignore_errors())

  @combinations.generate(test_base.default_test_combinations())
  def testIgnoreErrorsCore(self):
    num_outputs = 4
    self.run_core_tests(self._build_ds, num_outputs)


if __name__ == "__main__":
  test.main()
