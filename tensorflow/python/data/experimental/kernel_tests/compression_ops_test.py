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
"""Tests for compression ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


def _test_objects():
  return [
      combinations.NamedObject("int", 1),
      combinations.NamedObject("string", "dog"),
      combinations.NamedObject("tuple", (1, 1)),
      combinations.NamedObject("int_string_tuple", (1, "dog")),
      combinations.NamedObject(
          "sparse",
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])),
      combinations.NamedObject(
          "sparse_structured", {
              "a":
                  sparse_tensor.SparseTensorValue(
                      indices=[[0, 0], [1, 2]],
                      values=[1, 2],
                      dense_shape=[3, 4]),
              "b": (1, 2, "dog")
          })
  ]


class CompressionOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(element=_test_objects())))
  def testCompression(self, element):
    element = element._obj

    compressed = compression_ops.compress(element)
    uncompressed = compression_ops.uncompress(
        compressed, structure.type_spec_from_value(element))
    self.assertValuesEqual(element, self.evaluate(uncompressed))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(element=_test_objects())))
  def testDatasetCompression(self, element):
    element = element._obj

    dataset = dataset_ops.Dataset.from_tensors(element)
    element_spec = dataset.element_spec

    dataset = dataset.map(lambda *x: compression_ops.compress(x))
    dataset = dataset.map(lambda x: compression_ops.uncompress(x, element_spec))
    self.assertDatasetProduces(dataset, [element])


if __name__ == "__main__":
  test.main()
