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
"""Tests for the datasets shape inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.grappler import item
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GrapplerTest(test.TestCase):

  def testFromTensors(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromTensorSlices(self):
    test_cases = [{
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[[1, 2, 3]]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensor_slices(test_case['tensor'])
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromGenerator(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:

      def make_generator(tensor):

        def generator():
          yield tensor

        return generator

      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_generator(
            make_generator(test_case['tensor']),
            dtypes.int64,
            output_shapes=test_case['shape'])
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testRange(self):
    with ops.Graph().as_default() as g:
      dataset = dataset_ops.Dataset.range(42)
      iterator = dataset.make_one_shot_iterator()
      get_next = iterator.get_next()
      train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
      train_op.append(get_next)
      mg = meta_graph.create_meta_graph_def(graph=g)
      grappler_item = item.Item(mg)
      op_properties = grappler_item.GetOpProperties()
      self.assertEqual(tensor_shape.scalar(),
                       op_properties['IteratorGetNext'][0].shape)

  def _testTransformation(self, fn):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape({})
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = fn(dataset, test_case['tensor'], test_case['shape'])
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testConcatenate(self):

    def fn(dataset, tensor, shape):
      del shape
      return dataset.concatenate(dataset_ops.Dataset.from_tensors(tensor))

    self._testTransformation(fn)

  def testPrefetch(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.prefetch(42)

    self._testTransformation(fn)

  def testRepeat(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.repeat(42)

    self._testTransformation(fn)

  def testShuffle(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.shuffle(42)

    self._testTransformation(fn)

  def testCache(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.cache()

    self._testTransformation(fn)

  def testTake(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.take(42)

    self._testTransformation(fn)

  def testSkip(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.skip(42)

    self._testTransformation(fn)

  def testShard(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.shard(42, 0)

    self._testTransformation(fn)

  def testFilter(self):

    def fn(dataset, tensor, shape):
      del tensor, shape
      return dataset.filter(lambda x: True)

    self._testTransformation(fn)

  def as_tensor_shape(self, proto_with_symbolic_values):
    for i in range(len(proto_with_symbolic_values.dim)):
      if proto_with_symbolic_values.dim[i].size < -1:
        proto_with_symbolic_values.dim[i].size = -1
    return tensor_shape.TensorShape(proto_with_symbolic_values)

  def testBatch(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([None])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([None, 3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([None, 1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.batch(42)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        inferred_shape = self.as_tensor_shape(
            op_properties['IteratorGetNext'][0].shape)
        self.assertTrue(test_case['shape'][0].is_compatible_with(
            inferred_shape[0]))
        self.assertEqual(test_case['shape'][1:], inferred_shape[1:])

  def testPaddedBatch(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([None])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([None, 4])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([None, 2, 4])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.padded_batch(42, padded_shapes=test_case['shape'][1:])
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        inferred_shape = self.as_tensor_shape(
            op_properties['IteratorGetNext'][0].shape)
        self.assertTrue(test_case['shape'][0].is_compatible_with(
            inferred_shape[0]))
        self.assertEqual(test_case['shape'][1:], inferred_shape[1:])

  def testFlatMap(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.range(42)

        def make_dataset(tensor):

          def dataset_fn(n):
            return dataset_ops.Dataset.from_tensors(tensor).repeat(n)

          return dataset_fn

        dataset = dataset.flat_map(make_dataset(test_case['tensor']))
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testInterleave(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([1, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.range(42)

        def make_dataset(tensor):

          def dataset_fn(n):
            return dataset_ops.Dataset.from_tensors(tensor).repeat(n)

          return dataset_fn

        dataset = dataset.interleave(
            make_dataset(test_case['tensor']), cycle_length=42)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testMap(self):
    test_cases = [{
        'tensor': 0,
        'shape': tensor_shape.TensorShape([])
    }, {
        'tensor': np.array([1, 2, 3]),
        'shape': tensor_shape.TensorShape([3])
    }, {
        'tensor': np.array([[1, 2, 3]]),
        'shape': tensor_shape.TensorShape([3, 1])
    }, {
        'tensor': np.array([[[1, 2, 3], [4, 5, 6]]]),
        'shape': tensor_shape.TensorShape([3, 2, 1])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        dataset = dataset_ops.Dataset.from_tensors(test_case['tensor'])
        dataset = dataset.map(array_ops.transpose)
        iterator = dataset.make_one_shot_iterator()
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromStructure(self):
    test_cases = [{
        'shape': tensor_shape.TensorShape([])
    }, {
        'shape': tensor_shape.TensorShape([3])
    }, {
        'shape': tensor_shape.TensorShape([1, 2])
    }, {
        'shape': tensor_shape.TensorShape([1, 2, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        iterator = iterator_ops.Iterator.from_structure(
            dtypes.int64, output_shapes=test_case['shape'])
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)

  def testFromStringHandle(self):
    test_cases = [{
        'shape': tensor_shape.TensorShape([])
    }, {
        'shape': tensor_shape.TensorShape([3])
    }, {
        'shape': tensor_shape.TensorShape([1, 2])
    }, {
        'shape': tensor_shape.TensorShape([1, 2, 3])
    }]

    for test_case in test_cases:
      with ops.Graph().as_default() as g:
        iterator = iterator_ops.Iterator.from_structure(dtypes.int64)
        handle = iterator.string_handle()
        iterator = iterator_ops.Iterator.from_string_handle(
            handle, dtypes.int64, output_shapes=test_case['shape'])
        get_next = iterator.get_next()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(get_next)
        mg = meta_graph.create_meta_graph_def(graph=g)
        grappler_item = item.Item(mg)
        op_properties = grappler_item.GetOpProperties()
        self.assertEqual(test_case['shape'],
                         op_properties['IteratorGetNext'][0].shape)


if __name__ == '__main__':
  test.main()
