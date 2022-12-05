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
"""Tests for `tf.data.experimental.from_list()."""
import collections
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import from_list
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class FromListTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(elements=[
              combinations.NamedObject("empty_input", []),
              combinations.NamedObject("non-list_input", (1, 2, 3)),
          ])))
  def testInvalidInputs(self, elements):
    with self.assertRaises(ValueError):
      from_list.from_list(elements._obj)

  def testLargeNInputs(self):
    elements = list(range(1000))
    dataset = dataset_ops.Dataset.from_tensor_slices(elements)
    self.assertDatasetProduces(dataset, expected_output=elements)

  @combinations.generate(test_base.default_test_combinations())
  def testTupleInputs(self):
    elements = [(1, 2), (3, 4)]
    dataset = from_list.from_list(elements)
    self.assertEqual(
        [np.shape(c) for c in elements[0]],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    self.assertDatasetProduces(dataset, expected_output=elements)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(map_fn=[
              combinations.NamedObject("simple", lambda x: x),
              combinations.NamedObject(
                  "ordered_dicts",
                  lambda x: collections.OrderedDict([("x", x)]))
          ])))
  def testDatasetInputs(self, map_fn):
    dss = [dataset_ops.Dataset.range(10).map(map_fn) for _ in range(10)]
    ds = dataset_ops.Dataset.from_tensor_slices(dss)
    ds = ds.flat_map(lambda x: x)
    self.assertDatasetProduces(
        ds, expected_output=[map_fn(x) for x in list(range(10)) * 10])

  @combinations.generate(test_base.default_test_combinations())
  def testNonRectangularInputs(self):
    elements = [[[1]], [[2, 3]], [[4, 5, 6]]]
    dataset = from_list.from_list(elements)
    self.assertEqual(
        tensor_shape.Dimension(1),
        dataset_ops.get_legacy_output_shapes(dataset)[0])
    self.assertDatasetProduces(dataset, expected_output=elements)

  @combinations.generate(test_base.default_test_combinations())
  def testDictInputs(self):
    elements = [{
        "foo": [1, 2, 3],
        "bar": [[4.0], [5.0], [6.0]]
    }, {
        "foo": [4, 5, 6],
        "bar": [[7.0], [8.0], [9.0]]
    }]
    dataset = from_list.from_list(elements)
    self.assertEqual(dtypes.int32,
                     dataset_ops.get_legacy_output_types(dataset)["foo"])
    self.assertEqual(dtypes.float32,
                     dataset_ops.get_legacy_output_types(dataset)["bar"])
    self.assertEqual((3,), dataset_ops.get_legacy_output_shapes(dataset)["foo"])
    self.assertEqual((3, 1),
                     dataset_ops.get_legacy_output_shapes(dataset)["bar"])
    self.assertDatasetProduces(dataset, expected_output=elements)

  @combinations.generate(test_base.default_test_combinations())
  def testUintInputs(self):
    elements = [(np.tile(np.array([[0], [1]], dtype=np.uint8), 2),
                 np.tile(np.array([[2], [256]], dtype=np.uint16), 2),
                 np.tile(np.array([[4], [65536]], dtype=np.uint32), 2),
                 np.tile(np.array([[8], [4294967296]], dtype=np.uint64), 2))]
    dataset = from_list.from_list(elements)
    self.assertEqual(
        (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64),
        dataset_ops.get_legacy_output_types(dataset))
    self.assertDatasetProduces(dataset, elements)


class FromListRandomAccessTest(test_base.DatasetTestBase,
                               parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidIndex(self):
    dataset = from_list.from_list([1, 2, 3])
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, -1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(random_access.at(dataset, 3))

  @combinations.generate(test_base.default_test_combinations())
  def testOneDimensionalArray(self):
    tensor = [1, 2, 3]
    dataset = from_list.from_list(tensor)
    for i in range(len(tensor)):
      results = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(tensor[i], results)

  @combinations.generate(test_base.default_test_combinations())
  def testTwoDimensionalArray(self):
    tensor = [[1, 2], [3, 4]]
    dataset = from_list.from_list(tensor)
    for i in range(2):
      results = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(tensor[i], results)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleElements(self):
    dataset = from_list.from_list([[1, 2], [3, 4], [5, 6]])
    self.assertEqual(1, self.evaluate(random_access.at(dataset, 0))[0])
    self.assertEqual(2, self.evaluate(random_access.at(dataset, 0))[1])
    self.assertEqual(3, self.evaluate(random_access.at(dataset, 1))[0])
    self.assertEqual(4, self.evaluate(random_access.at(dataset, 1))[1])

  @combinations.generate(test_base.default_test_combinations())
  def testDictionary(self):
    dataset = from_list.from_list([{"a": 1, "b": 3}, {"a": 2, "b": 4}])
    self.assertEqual({
        "a": 1,
        "b": 3
    }, self.evaluate(random_access.at(dataset, 0)))
    self.assertEqual({
        "a": 2,
        "b": 4
    }, self.evaluate(random_access.at(dataset, 1)))

  @combinations.generate(test_base.default_test_combinations())
  def testNumpy(self):
    elements = [
        np.tile(np.array([[0], [1]], dtype=np.uint64), 2),
        np.tile(np.array([[2], [256]], dtype=np.uint64), 2),
        np.tile(np.array([[4], [65536]], dtype=np.uint64), 2),
        np.tile(np.array([[8], [4294967296]], dtype=np.uint64), 2),
    ]
    dataset = from_list.from_list(elements)
    for i in range(len(elements)):
      result = self.evaluate(random_access.at(dataset, i))
      self.assertAllEqual(elements[i], result)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = from_list.from_list([42], name="from_list")
    self.assertDatasetProduces(dataset, [42])


class FromListCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                             parameterized.TestCase):

  def _build_list_dataset(self, elements, options=None):
    dataset = from_list.from_list(elements)
    if options:
      dataset = dataset.with_options(options)
    return dataset

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # Equal length elements
    elements = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array([37, 38, 39, 40])
    ]
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    verify_fn(
        self,
        lambda: self._build_list_dataset(elements, options),
        num_outputs=3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         checkpoint_test_base.default_test_combinations()))
  def testDict(self, verify_fn):
    dict_elements = [{
        "foo": 1,
        "bar": 4.0
    }, {
        "foo": 2,
        "bar": 5.0
    }, {
        "foo": 3,
        "bar": 6.0
    }]
    verify_fn(
        self, lambda: self._build_list_dataset(dict_elements), num_outputs=3)


if __name__ == "__main__":
  test.main()
