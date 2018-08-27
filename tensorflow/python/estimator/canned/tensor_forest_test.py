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
"""Tests for tensor_forest estimator and model_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf.json_format import ParseDict
from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.estimator.canned import tensor_forest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class TensorForestTest(test_util.TensorFlowTestCase):

  def testInfrenceFromRestoredModel(self):
    input_data = [[-1., 0.], [-1., 2.],  # node 1
                  [1., 0.], [1., -2.]]  # node 2
    expected_prediction = [[0.0, 1.0], [0.0, 1.0],
                           [0.0, 1.0], [0.0, 1.0]]
    hparams = tensor_forest._ForestHParams(
        logits_dimension=2,
        n_trees=1,
        max_nodes=1000,
        num_splits_to_consider=250,
        split_node_after_samples=25,
        is_regression=False)
    tree_weight = {'nodes':
                   [{'dense_split':
                     {'right_id': 2,
                      'left_id': 1,
                      'feature_id': 0,
                      'threshold': 0.0}},
                    {'leaf': {'vector':
                              {'value': [0.0, 1.0]}}},
                    {'leaf': {'vector':
                              {'value': [0.0, 1.0]}}},
                    ]}
    restored_tree_param = ParseDict(tree_weight,
                                    boosted_trees_pb2.Tree()).SerializeToString()
    graph_builder = tensor_forest.RandomForestGraphs(hparams, None,
                                                     [restored_tree_param])
    probs, var = graph_builder.inference_graph(
        input_data)
    self.assertTrue(isinstance(probs, ops.Tensor))
    self.assertTrue(isinstance(paths, ops.Tensor))
    self.assertTrue(isinstance(var, ops.Tensor))
    with self.test_session():
      variables.global_variables_initializer().run()
      resources.initialize_resources(
          resources.shared_resources()).run()
      self.assertEquals(probs.eval().shape, (4, 2))
      self.assertEquals(probs.eval().tolist(),
                        expected_prediction)


if __name__ == '__main__':
  googletest.main()
