# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `experimental_slack` option."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class PrefetchWithSlackTest(test_base.DatasetTestBase, parameterized.TestCase):

  # TODO(b/121264236)
  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testPrefetchWithSlackOption(self):
    """Determines slack_period based on num devices attached to iterator."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = dataset_ops.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])
    dataset = multi_device_iterator._dataset  # pylint: disable=protected-access
    self.assertIn("slack", dataset.options()._graph_rewrites())
    self.assertIn("slack:slack_period:2",
                  dataset.options()._graph_rewrite_configs())

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testPrefetchWithSlackOptionWithoutIterator(self):
    """Defaults to slack period of 1 without iterator."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    options = dataset_ops.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertIn("slack", dataset.options()._graph_rewrites())
    self.assertIn("slack:slack_period:1",
                  dataset.options()._graph_rewrite_configs())
    self.assertDatasetProduces(dataset, range(10))

  @combinations.generate(test_base.default_test_combinations())
  def testWithPassthroughDataset(self):
    """Should still work with a passthrough dataset after prefetch()."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.map(lambda x: x + 1)
    options = dataset_ops.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, range(1, 11))

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithoutPrefetch(self):
    """The rewrite fails if there is no prefetch() in the pipeline."""
    dataset = dataset_ops.Dataset.range(10)
    options = dataset_ops.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    with self.assertRaises(errors.InvalidArgumentError):
      get_next = self.getNext(dataset)
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testErrorWithInvalidDataset(self):
    """With a nested dataset op after prefetch, the rewrite should fail."""
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.prefetch(1)
    dataset = dataset.flat_map(dataset_ops.Dataset.from_tensors)
    options = dataset_ops.Options()
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    with self.assertRaises(errors.InvalidArgumentError):
      get_next = self.getNext(dataset)
      self.evaluate(get_next())


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3}))
  test.main()
