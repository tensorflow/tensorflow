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
"""Tests for tensorflow.metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util

class SpecificityTest(test_util.TensorFlowTestCase):

  def __init__(self, method_name="runTest"):
    super(SpecificityTest, self).__init__(method_name)

  def testSpecificity(self):
    with self.test_session(use_gpu=True) as sess:
      labels = array_ops.placeholder(dtypes.float32, shape=(None,))
      predictions = array_ops.placeholder(dtypes.float32, shape=(None,))
      specificity, update_specificity = metrics.specificity(
          labels, predictions)
      sess.run(variables.local_variables_initializer())

      sess.run(update_specificity, feed_dict={
          labels: [1.0, 1.0],
          predictions: [1.0, 1.0]
      })
      s = specificity.eval()
      self.assertEqual(s, 0)

      sess.run(update_specificity, feed_dict={
          labels: [0.0, 0.0],
          predictions: [0.0, 1.0]
      })
      s = specificity.eval()
      self.assertEqual(s, 0.5)

      sess.run(update_specificity, feed_dict={
          labels: [0.0, 0.0],
          predictions: [0.0, 0.0]
      })
      s = specificity.eval()
      self.assertEqual(s, 0.75)

if __name__ == "__main__":
  googletest.main()
