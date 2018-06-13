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
"""Large tests for metric_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class StreamingPrecisionRecallAtEqualThresholdsLargeTest(test.TestCase):

  def setUp(self):
    np.random.seed(1)
    ops.reset_default_graph()

  def testLargeCase(self):
    shape = [32, 512, 256, 1]
    predictions = random_ops.random_uniform(
        shape, 0.0, 1.0, dtype=dtypes_lib.float32)
    labels = math_ops.greater(random_ops.random_uniform(shape, 0.0, 1.0), 0.5)

    result, update_op = metric_ops.precision_recall_at_equal_thresholds(
        labels=labels, predictions=predictions, num_thresholds=201)
    # Run many updates, enough to cause highly inaccurate values if the
    # code used float32 for accumulation.
    num_updates = 71

    with self.test_session() as sess:
      sess.run(variables.local_variables_initializer())
      for _ in xrange(num_updates):
        sess.run(update_op)

      prdata = sess.run(result)

      # Since we use random values, we won't know the tp/fp/tn/fn values, but
      # tp and fp at threshold 0 should be the total number of positive and
      # negative labels, hence their sum should be total number of pixels.
      expected_value = 1.0 * np.product(shape) * num_updates
      got_value = prdata.tp[0] + prdata.fp[0]
      # They should be at least within 1.
      self.assertNear(got_value, expected_value, 1.0)

if __name__ == '__main__':
  test.main()
