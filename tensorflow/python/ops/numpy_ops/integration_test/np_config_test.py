# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that an error is raised when numpy functions are called."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.python.ops.numpy_ops import np_config


class ConfigTest(tf.test.TestCase):

  def testMethods(self):
    a = tf.constant(1.)

    for name in {'T', 'astype', 'ravel', 'transpose', 'reshape', 'clip', 'size',
                 'tolist'}:
      with self.assertRaisesRegex(AttributeError, 'enable_numpy_behavior'):
        getattr(a, name)

    np_config.enable_numpy_behavior()

    for name in {'T', 'astype', 'ravel', 'transpose', 'reshape', 'clip', 'size',
                 'tolist'}:
      _ = getattr(a, name)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
