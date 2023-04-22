# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.DatasetSpec`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test


class DatasetSpecTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testInputSignature(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        np.arange(10).astype(np.int32)).batch(5)

    @def_function.function(input_signature=[
        dataset_ops.DatasetSpec(
            tensor_spec.TensorSpec(
                shape=(None,), dtype=dtypes.int32, name=None),
            tensor_shape.TensorShape([]))
    ])
    def fn(_):
      pass

    fn(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecInnerSpec(self):
    inner_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    ds_spec = dataset_ops.DatasetSpec(inner_spec)
    self.assertEqual(ds_spec.element_spec, inner_spec)


if __name__ == "__main__":
  test.main()
