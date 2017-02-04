# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.tensor_forest.ops.tree_predictions_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class TreePredictionsDenseTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []
    spec_proto = data_ops.TensorForestDataSpec()
    f1 = spec_proto.dense.add()
    f1.name = 'f1'
    f1.original_type = data_ops.DATA_FLOAT
    f1.size = 1

    f2 = spec_proto.dense.add()
    f2.name = 'f2'
    f2.original_type = data_ops.DATA_FLOAT
    f2.size = 1
    spec_proto.dense_features_size = 2
    self.data_spec = spec_proto.SerializeToString()

  def testSimple(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          tree,
          tree_thresholds,
          node_pcw,
          input_spec=self.data_spec,
          valid_leaf_threshold=1)

      self.assertAllClose([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
                          predictions.eval())

  def testBackoffToParent(self):
    input_data = [
        [-1., 0.],
        [-1., 2.],  # node 1
        [1., 0.],
        [1., -2.]
    ]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[15.0, 3.0, 9.0, 3.0], [5.0, 1.0, 1.0, 3.0],
                [25.0, 5.0, 20.0, 0.0]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=10,
          input_spec=self.data_spec)

      # Node 2 has enough data, but Node 1 needs to combine with the parent
      # counts.
      self.assertAllClose([[0.2, 0.4, 0.4], [0.2, 0.4, 0.4], [0.2, 0.8, 0.0],
                           [0.2, 0.8, 0.0]], predictions.eval())

  def testNoInput(self):
    input_data = []

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=10,
          input_spec=self.data_spec)

      self.assertEquals((0, 3), predictions.eval().shape)

  def testBadInput(self):
    input_data = [
        [-1., 0.],
        [-1., 2.],  # node 1
        [1., 0.],
        [1., -2.]
    ]  # node 2

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0.]  # not enough nodes.
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      with self.assertRaisesOpError(
          'Number of nodes should be the same in tree, tree_thresholds '
          'and node_pcw.'):
        predictions = tensor_forest_ops.tree_predictions(
            input_data,
            self.nothing,
            self.nothing,
            self.nothing,
            tree,
            tree_thresholds,
            node_pcw,
            valid_leaf_threshold=10,
            input_spec=self.data_spec)

        self.assertEquals((0, 3), predictions.eval().shape)


class TreePredictionsSparseTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []
    spec_proto = data_ops.TensorForestDataSpec()
    f1 = spec_proto.sparse.add()
    f1.name = 'f1'
    f1.original_type = data_ops.DATA_FLOAT
    f1.size = 1

    f2 = spec_proto.sparse.add()
    f2.name = 'f2'
    f2.original_type = data_ops.DATA_FLOAT
    f2.size = 9
    spec_proto.dense_features_size = 0
    self.data_spec = spec_proto.SerializeToString()

  def testSparseInput(self):
    sparse_shape = [3, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0]

    tree = [[1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          self.nothing,
          sparse_indices,
          sparse_values,
          sparse_shape,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1,
          input_spec=self.data_spec)

      self.assertAllClose([[0.5, 0.25, 0.25],
                           [0.5, 0.25, 0.25],
                           [0.1, 0.1, 0.8]],
                          predictions.eval())

  def testSparseInputDefaultIsZero(self):
    sparse_shape = [3, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0]

    tree = [[1, 7], [-1, 0], [-1, 0]]
    tree_thresholds = [3.0, 0., 0.]
    node_pcw = [[1.0, 0.3, 0.4, 0.3], [1.0, 0.1, 0.1, 0.8],
                [1.0, 0.5, 0.25, 0.25]]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          self.nothing,
          sparse_indices,
          sparse_values,
          sparse_shape,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1,
          input_spec=self.data_spec)

      self.assertAllClose([[0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25],
                           [0.1, 0.1, 0.8]],
                          predictions.eval())


class TreePredictionsMixedTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []
    spec_proto = data_ops.TensorForestDataSpec()
    f1 = spec_proto.dense.add()
    f1.name = 'f1'
    f1.original_type = data_ops.DATA_FLOAT
    f1.size = 2

    f2 = spec_proto.dense.add()
    f2.name = 'f2'
    f2.original_type = data_ops.DATA_CATEGORICAL
    f2.size = 1

    f3 = spec_proto.sparse.add()
    f3.name = 'f3'
    f3.original_type = data_ops.DATA_FLOAT
    f3.size = -1
    spec_proto.dense_features_size = 3
    self.data_spec = spec_proto.SerializeToString()

  def testSimpleMixed(self):
    #        0       1       2       3        4        5        6
    tree = [[1, 0], [3, 2], [5, 5], [-1, 0], [-1, 0], [-1, 0], [-1, 0]]
    tree_thresholds = [0., 15., 1., 0., 0., 0., 0.]
    node_pcw = [[1.0, 0., 1.0, 0.4, 0.3], [1.0, 0., 0.1, 0.1, 0.8],
                [1.0, 0., 0.5, 0.25, 0.25], [1.0, 1., 0., 0., 0.],
                [1.0, 0., 1., 0., 0.], [1.0, 0., 0., 1., 0.],
                [1.0, 0., 0., 0., 1.]]

    input_data = [
        [-1., 0., 15.],  # node 3
        [-1., 2., 11.],  # node 4
        [1., 0., 11.],
        [1., -2., 30.]
    ]

    sparse_shape = [4, 5]
    sparse_indices = [
        [0, 0],
        [0, 1],
        [0, 4],
        [1, 0],
        [1, 2],
        [2, 1],  # node 5
        [3, 2]
    ]  # node 6
    sparse_values = [3.0, -1.0, 0.5, 1.5, 6.0, -2.0, 2.0]

    with self.test_session():
      predictions = tensor_forest_ops.tree_predictions(
          input_data,
          sparse_indices,
          sparse_values,
          sparse_shape,
          tree,
          tree_thresholds,
          node_pcw,
          valid_leaf_threshold=1,
          input_spec=self.data_spec)

      self.assertAllClose([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                           [0., 0., 0., 1.]], predictions.eval())


if __name__ == '__main__':
  googletest.main()
