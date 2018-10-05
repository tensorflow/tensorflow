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
"""Tests for losses util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import test


class LossesUtilTest(test.TestCase):

  def testGetRegularizationLoss(self):
    # Empty regularization collection should evaluate to 0.0.
    with self.cached_session():
      self.assertEqual(0.0, util.get_regularization_loss().eval())

    # Loss should sum.
    ops.add_to_collection(
        ops.GraphKeys.REGULARIZATION_LOSSES, constant_op.constant(2.0))
    ops.add_to_collection(
        ops.GraphKeys.REGULARIZATION_LOSSES, constant_op.constant(3.0))
    with self.cached_session():
      self.assertEqual(5.0, util.get_regularization_loss().eval())

    # Check scope capture mechanism.
    with ops.name_scope('scope1'):
      ops.add_to_collection(
          ops.GraphKeys.REGULARIZATION_LOSSES, constant_op.constant(-1.0))
    with self.cached_session():
      self.assertEqual(-1.0, util.get_regularization_loss('scope1').eval())


if __name__ == '__main__':
  test.main()
