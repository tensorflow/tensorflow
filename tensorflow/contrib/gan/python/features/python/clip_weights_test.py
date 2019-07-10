# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for features.clip_weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.gan.python.features.python import clip_weights_impl as clip_weights

from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import training


class ClipWeightsTest(test.TestCase):
  """Tests for `discriminator_weight_clip`."""

  def setUp(self):
    super(ClipWeightsTest, self).setUp()
    self.variables = [variables.Variable(2.0)]
    self.tuple = collections.namedtuple(
        'VarTuple', ['discriminator_variables'])(self.variables)

  def _test_weight_clipping_helper(self, use_tuple):
    loss = self.variables[0]
    opt = training.GradientDescentOptimizer(1.0)
    if use_tuple:
      opt_clip = clip_weights.clip_variables(opt, self.variables, 0.1)
    else:
      opt_clip = clip_weights.clip_discriminator_weights(opt, self.tuple, 0.1)

    train_op1 = opt.minimize(loss, var_list=self.variables)
    train_op2 = opt_clip.minimize(loss, var_list=self.variables)

    with self.cached_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      self.assertEqual(2.0, self.variables[0].eval())
      sess.run(train_op1)
      self.assertLess(0.1, self.variables[0].eval())

    with self.cached_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      self.assertEqual(2.0, self.variables[0].eval())
      sess.run(train_op2)
      self.assertNear(0.1, self.variables[0].eval(), 1e-7)

  def test_weight_clipping_argsonly(self):
    self._test_weight_clipping_helper(False)

  def test_weight_clipping_ganmodel(self):
    self._test_weight_clipping_helper(True)

  def _test_incorrect_weight_clip_value_helper(self, use_tuple):
    opt = training.GradientDescentOptimizer(1.0)

    if use_tuple:
      with self.assertRaisesRegexp(ValueError, 'must be positive'):
        clip_weights.clip_discriminator_weights(opt, self.tuple, weight_clip=-1)
    else:
      with self.assertRaisesRegexp(ValueError, 'must be positive'):
        clip_weights.clip_variables(opt, self.variables, weight_clip=-1)

  def test_incorrect_weight_clip_value_argsonly(self):
    self._test_incorrect_weight_clip_value_helper(False)

  def test_incorrect_weight_clip_value_tuple(self):
    self._test_incorrect_weight_clip_value_helper(True)


if __name__ == '__main__':
  test.main()
