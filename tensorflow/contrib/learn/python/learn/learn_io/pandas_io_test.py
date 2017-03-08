# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for pandas_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.learn_io import pandas_io
from tensorflow.python.framework import errors

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


class PandasIoTest(tf.test.TestCase):

  def testPandasInputFn(self):
    if not HAS_PANDAS:
      return
    index = np.arange(100, 104)
    a = np.arange(4)
    b = np.arange(32, 36)
    x = pd.DataFrame({'a': a, 'b': b}, index=index)
    y_noindex = pd.Series(np.arange(-32, -28))
    y = pd.Series(np.arange(-32, -28), index=index)
    with self.test_session() as session:
      with self.assertRaises(ValueError):
        failing_input_fn = pandas_io.pandas_input_fn(
            x, y_noindex, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['index'], [100, 101])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      session.run([features, target])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  tf.test.main()
