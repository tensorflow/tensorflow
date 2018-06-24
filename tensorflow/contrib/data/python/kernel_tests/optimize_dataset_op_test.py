# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import optimization
from tensorflow.core.framework import graph_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class OptimizeDatasetTest(test.TestCase):

  def testDefaultOptimizations(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x).batch(
        10).apply(optimization.optimize())
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      graph = graph_pb2.GraphDef().FromString(
          sess.run(dataset._as_serialized_graph()))
      self.assertTrue(
          all([node.op != "MapAndBatchDatasetV2" for node in graph.node]))
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testEmptyOptimizations(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x).batch(
        10).apply(optimization.optimize([]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      graph = graph_pb2.GraphDef().FromString(
          sess.run(dataset._as_serialized_graph()))
      self.assertTrue(
          all([node.op != "MapAndBatchDatasetV2" for node in graph.node]))
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testOptimization(self):
    dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x).batch(
        10).apply(optimization.optimize(["map_and_batch_fusion"]))
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.test_session() as sess:
      graph = graph_pb2.GraphDef().FromString(
          sess.run(dataset._as_serialized_graph()))
      self.assertTrue(
          any([node.op == "MapAndBatchDatasetV2" for node in graph.node]))
      self.assertAllEqual([x * x for x in range(10)], sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)


class OptimizeDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testCore(self):

    def build_dataset(num_elements, batch_size):
      return dataset_ops.Dataset.range(num_elements).map(lambda x: x * x).batch(
          batch_size).apply(optimization.optimize(["map_and_batch_fusion"]))

    self.run_core_tests(lambda: build_dataset(200, 10), None, 20)


if __name__ == "__main__":
  test.main()
