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
"""Test that the contrib module shows up properly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class ContribTest(test.TestCase):

  def testContrib(self):
    # pylint: disable=g-import-not-at-top
    import tensorflow as tf
    _ = tf.contrib.layers  # `tf.contrib` is loaded lazily on first use.
    assert tf_inspect.ismodule(tf.contrib)

  def testLayers(self):
    # pylint: disable=g-import-not-at-top
    import tensorflow as tf
    assert tf_inspect.ismodule(tf.contrib.layers)

  def testLinearOptimizer(self):
    # pylint: disable=g-import-not-at-top
    import tensorflow as tf
    assert tf_inspect.ismodule(tf.contrib.linear_optimizer)


if __name__ == '__main__':
  test.main()
