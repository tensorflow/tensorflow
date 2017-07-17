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
"""Tests for tf.contrib.tensor_forest.ops.tree_predictions_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import descriptor_pool
from google.protobuf import text_format

from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.python import tensor_forest_v4
from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensorflow.contrib.tensor_forest.python.ops import model_ops

from tensorflow.python.framework import test_util
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


TREE_PROTO_DENSE = """
decision_tree {
  nodes {
    binary_node {
      left_child_id {
        value: 1
      }
      right_child_id {
        value: 2
      }
      inequality_left_child_test {
        feature_id {
          id {
            value: "0"
          }
        }
        threshold {
          float_value: 0
        }
      }
    }
  }
  nodes {
    node_id {
      value: 1
    }
    leaf {
      vector {
        value {
          float_value: 10
        }
        value {
          float_value: 10
        }
        value {
          float_value: 80
        }
      }
    }
  }
  nodes {
    node_id {
      value: 2
    }
    leaf {
      vector {
        value {
          float_value: 50
        }
        value {
          float_value: 25
        }
        value {
          float_value: 25
        }
      }
    }
  }
}
"""

# TODO(gilberth): This currently has to load inequality_left_child_test
# in the tree because of a proto parsing error (MatchingValuesTest not
# found in the descriptor pool).
TREE_PROTO_SPARSE = """
decision_tree {
  nodes {
    binary_node {
      left_child_id {
        value: 1
      }
      right_child_id {
        value: 2
      }
      inequality_left_child_test {
        feature_id {
          id {
            value: "1"
          }
        }
        threshold {
          float_value: 10
        }
      }
    }
  }
  nodes {
    node_id {
      value: 1
    }
    leaf {
      vector {
        value {
          float_value: 10
        }
        value {
          float_value: 10
        }
        value {
          float_value: 80
        }
      }
    }
  }
  nodes {
    node_id {
      value: 2
    }
    leaf {
      vector {
        value {
          float_value: 50
        }
        value {
          float_value: 25
        }
        value {
          float_value: 25
        }
      }
    }
  }
}
"""

FEATURE_COUNT_TREE = """
decision_tree {
  nodes {
    binary_node {
      left_child_id {
        value: 1
      }
      right_child_id {
        value: 2
      }
      inequality_left_child_test {
        feature_id {
          id {
            value: "0"
          }
        }
        threshold {
          float_value: 0
        }
      }
    }
  }
  nodes {
    node_id {
      value: 1
    }
    binary_node {
      left_child_id {
        value: 3
      }
      right_child_id {
        value: 4
      }
      inequality_left_child_test {
        feature_id {
          id {
            value: "1"
          }
        }
        threshold {
          float_value: 0
        }
      }
    }
  }
  nodes {
    node_id {
      value: 2
    }
    binary_node {
      left_child_id {
        value: 5
      }
      right_child_id {
        value: 6
      }
      inequality_left_child_test {
        feature_id {
          id {
            value: "0"
          }
        }
        threshold {
          float_value: 10
        }
      }
    }
  }
  nodes {
    node_id {
      value: 3
    }
    leaf {
      vector {
        value {
          float_value: 10
        }
        value {
          float_value: 10
        }
        value {
          float_value: 80
        }
      }
    }
  }
  nodes {
    node_id {
      value: 4
    }
    leaf {
      vector {
        value {
          float_value: 50
        }
        value {
          float_value: 25
        }
        value {
          float_value: 25
        }
      }
    }
  }
  nodes {
    node_id {
      value: 5
    }
    leaf {
      vector {
        value {
          float_value: 50
        }
        value {
          float_value: 25
        }
        value {
          float_value: 25
        }
      }
    }
  }
  nodes {
    node_id {
      value: 6
    }
    leaf {
      vector {
        value {
          float_value: 50
        }
        value {
          float_value: 25
        }
        value {
          float_value: 25
        }
      }
    }
  }
}
"""


def get_v4_params(num_classes, num_features, regression):
  params = tensor_forest.ForestHParams()
  params.num_classes = num_classes
  params.num_features = num_features
  params.regression = regression
  v4_params = tensor_forest_v4.V4ForestHParams(params)
  v4_params.params_proto = tensor_forest_v4.build_params_proto(v4_params)
  v4_params.serialized_params_proto = (
      v4_params.params_proto.SerializeToString())
  return v4_params


def get_dense_data_spec():
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
  return spec_proto.SerializeToString()


def get_sparse_data_spec():
  spec_proto = data_ops.TensorForestDataSpec()
  f1 = spec_proto.dense.add()
  f1.name = 'f1'
  f1.original_type = data_ops.DATA_CATEGORICAL
  f1.size = 10
  spec_proto.dense_features_size = 0

  return spec_proto.SerializeToString()


class ModelOpsPredictionsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []

  def testDense(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2

    v4_params = get_v4_params(3, 2, False)

    tree_proto = generic_tree_model_pb2.Model()
    text_format.Merge(TREE_PROTO_DENSE, tree_proto,
                      descriptor_pool=descriptor_pool.Default())

    with self.test_session():
      tree = model_ops.tree_variable(
          v4_params, tree_proto.SerializeToString(), None, 'tree-0')
      resources.initialize_resources(resources.shared_resources()).run()

      predictions = model_ops.tree_predictions_v4(
          tree,
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          input_spec=get_dense_data_spec(),
          params=v4_params.serialized_params_proto)

      leaves = model_ops.traverse_tree_v4(
          tree,
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          input_spec=get_dense_data_spec(),
          params=v4_params.serialized_params_proto)

      self.assertAllClose([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]],
                          predictions.eval())

      self.assertAllEqual([1, 1, 2, 2], leaves.eval())

  def testSparseInput(self):
    sparse_shape = [4, 10]
    sparse_indices = [[0, 0], [0, 1], [0, 9],
                      [1, 0], [1, 1],
                      [2, 1],
                      [3, 6]]
    sparse_values = [3.0, 5.0, 0.5,
                     15.0, 10.0,
                     20.0,
                     20.0]

    v4_params = get_v4_params(3, 10, False)

    tree_proto = generic_tree_model_pb2.Model()
    text_format.Merge(TREE_PROTO_SPARSE, tree_proto,
                      descriptor_pool=descriptor_pool.Default())

    with self.test_session():
      tree = model_ops.tree_variable(
          v4_params, tree_proto.SerializeToString(), None, 'tree-0')
      resources.initialize_resources(resources.shared_resources()).run()

      predictions = model_ops.tree_predictions_v4(
          tree,
          self.nothing,
          sparse_indices,
          sparse_values,
          sparse_shape,
          input_spec=get_sparse_data_spec(),
          params=v4_params.serialized_params_proto)

      leaves = model_ops.traverse_tree_v4(
          tree,
          self.nothing,
          sparse_indices,
          sparse_values,
          sparse_shape,
          input_spec=get_sparse_data_spec(),
          params=v4_params.serialized_params_proto)

      self.assertAllClose([[0.1, 0.1, 0.8], [0.1, 0.1, 0.8],
                           [0.5, 0.25, 0.25], [0.1, 0.1, 0.8]],
                          predictions.eval())

      self.assertAllEqual([1, 1, 2, 1], leaves.eval())

  def testNoInput(self):
    v4_params = get_v4_params(3, 4, False)

    tree_proto = generic_tree_model_pb2.Model()
    text_format.Merge(TREE_PROTO_DENSE, tree_proto,
                      descriptor_pool=descriptor_pool.Default())

    with self.test_session():
      tree = model_ops.tree_variable(
          v4_params, tree_proto.SerializeToString(), None, 'tree-0')
      resources.initialize_resources(resources.shared_resources()).run()

      predictions = model_ops.tree_predictions_v4(
          tree,
          self.nothing,
          self.nothing,
          self.nothing,
          self.nothing,
          input_spec=get_dense_data_spec(),
          params=v4_params.serialized_params_proto)

      self.assertEqual((0, 3), predictions.eval().shape)


class ModelOpsUpdateTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []

  def testBasic(self):
    input_labels = [0, 1, 2, 1]
    leaves = [1, 2, 1, 2]
    weights = [10, 10, 10, 10]

    # For checking the resulting leaves after update.
    input_data = [[-1., 0.],  # node 1
                  [1., 0.]]   # node 2

    v4_params = get_v4_params(3, 2, False)

    tree_proto = generic_tree_model_pb2.Model()
    text_format.Merge(TREE_PROTO_DENSE, tree_proto,
                      descriptor_pool=descriptor_pool.Default())

    with self.test_session():
      tree = model_ops.tree_variable(
          v4_params, tree_proto.SerializeToString(), None, 'tree-0')
      resources.initialize_resources(resources.shared_resources()).run()

      model_ops.update_model_v4(
          tree,
          leaves,
          input_labels,
          weights,
          params=v4_params.serialized_params_proto).run()

      predictions = model_ops.tree_predictions_v4(
          tree,
          input_data,
          self.nothing,
          self.nothing,
          self.nothing,
          input_spec=get_dense_data_spec(),
          params=v4_params.serialized_params_proto)

      self.assertAllClose([[0.16666, 0.08333, 0.75],
                           [0.416666, 0.375, 0.208333]],
                          predictions.eval(),
                          rtol=0.0001, atol=0.0001)


class ModelOpsFeatureCountsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.nothing = []

  def testBasic(self):
    v4_params = get_v4_params(3, 4, False)

    tree_proto = generic_tree_model_pb2.Model()
    text_format.Merge(FEATURE_COUNT_TREE, tree_proto,
                      descriptor_pool=descriptor_pool.Default())

    with self.test_session():
      tree = model_ops.tree_variable(
          v4_params, tree_proto.SerializeToString(), None, 'tree-0')
      resources.initialize_resources(resources.shared_resources()).run()

      feature_counts = model_ops.feature_usage_counts(
          tree,
          params=v4_params.serialized_params_proto)

      self.assertAllEqual([2, 1, 0, 0], feature_counts.eval())


if __name__ == '__main__':
  googletest.main()
