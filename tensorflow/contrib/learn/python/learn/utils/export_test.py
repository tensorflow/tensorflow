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

"""Tests for export tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn


class ExportTest(tf.test.TestCase):

  def testExportMonitor(self):
    random.seed(42)
    x = np.random.rand(1000)
    y = 2 * x + 3
    regressor = learn.LinearRegressor()
    export_dir = tempfile.mkdtemp() + 'export/'
    export_monitor = learn.monitors.ExportMonitor(every_n_steps=1,
                                                  export_dir=export_dir,
                                                  exports_to_keep=1)
    regressor.fit(x, y, steps=10,
                  monitors=[export_monitor])
    self.assertTrue(tf.gfile.Exists(export_dir))
    self.assertFalse(tf.gfile.Exists(export_dir + '00000000/export'))
    self.assertTrue(tf.gfile.Exists(export_dir + '00000010/export'))


if __name__ == '__main__':
  tf.test.main()
