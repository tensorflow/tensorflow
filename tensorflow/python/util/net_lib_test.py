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

"""Tests for the SWIG-wrapped test lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.util import net_lib


class TestLibTest(tf.test.TestCase):

  def testPickUnusedPortOrDie(self):
    port0 = net_lib.pick_unused_port_or_die()
    port1 = net_lib.pick_unused_port_or_die()
    self.assertGreater(port0, 0)
    self.assertLess(port0, 65536)
    self.assertGreater(port1, 0)
    self.assertLess(port1, 65536)
    self.assertNotEqual(port0, port1)


if __name__ == "__main__":
  tf.test.main()
