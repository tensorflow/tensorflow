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
"""Tests for tf.contrib.tensor_forest.client.eval_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensor_forest.client import eval_metrics
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class EvalMetricsTest(test_util.TensorFlowTestCase):

  def testTop2(self):
    top_2_fn = eval_metrics._top_k_generator(2)
    probabilities = constant_op.constant([[0.1, 0.2, 0.3], [0.4, 0.7, 0.5],
                                          [0.9, 0.8, 0.2], [0.6, 0.4, 0.8]])
    targets = constant_op.constant([[0], [2], [1], [1]])
    in_top_2_op, update_op = top_2_fn(probabilities, targets)
    with self.test_session():
      # initializes internal accuracy vars
      variables.local_variables_initializer().run()
      # need to call in order to run the in_top_2_op internal operations because
      # it is a streaming function
      update_op.eval()
      self.assertNear(0.5, in_top_2_op.eval(), 0.0001)

  def testTop3(self):
    top_3_fn = eval_metrics._top_k_generator(3)
    probabilities = constant_op.constant([[0.1, 0.2, 0.6, 0.3, 0.5, 0.5],
                                          [0.1, 0.4, 0.7, 0.3, 0.5, 0.2],
                                          [0.1, 0.3, 0.8, 0.7, 0.4, 0.9],
                                          [0.9, 0.8, 0.1, 0.8, 0.2, 0.7],
                                          [0.3, 0.6, 0.9, 0.4, 0.8, 0.6]])
    targets = constant_op.constant([3, 0, 2, 5, 1])
    in_top_3_op, update_op = top_3_fn(probabilities, targets)
    with self.test_session():
      # initializes internal accuracy vars
      variables.local_variables_initializer().run()
      # need to call in order to run the in_top_3_op internal operations because
      # it is a streaming function
      update_op.eval()
      self.assertNear(0.4, in_top_3_op.eval(), 0.0001)

  def testAccuracy(self):
    predictions = constant_op.constant([0, 1, 3, 6, 5, 2, 7, 6, 4, 9])
    targets = constant_op.constant([0, 1, 4, 6, 5, 1, 7, 5, 4, 8])
    accuracy_op, update_op = eval_metrics._accuracy(predictions, targets)
    with self.test_session():
      variables.local_variables_initializer().run()
      # need to call in order to run the accuracy_op internal operations because
      # it is a streaming function
      update_op.eval()
      self.assertNear(0.6, accuracy_op.eval(), 0.0001)

  def testR2(self):
    scores = constant_op.constant(
        [1.2, 3.9, 2.1, 0.9, 2.2, 0.1, 6.0, 4.0, 0.9])
    targets = constant_op.constant(
        [1.0, 4.3, 2.6, 0.5, 1.1, 0.7, 5.1, 3.4, 1.8])
    r2_op, update_op = eval_metrics._r2(scores, targets)
    with self.test_session():
      # initializes internal accuracy vars
      variables.local_variables_initializer().run()
      # need to call in order to run the r2_op internal operations because
      # it is a streaming function
      update_op.eval()
      self.assertNear(0.813583, r2_op.eval(), 0.0001)


if __name__ == '__main__':
  googletest.main()
