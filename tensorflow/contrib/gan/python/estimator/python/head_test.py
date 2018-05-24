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
"""Tests for TFGAN's head.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python.estimator.python import head

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import training


def dummy_loss(gan_model, add_summaries=True):  # pylint:disable=unused-argument
  return math_ops.reduce_sum(gan_model.discriminator_real_outputs -
                             gan_model.discriminator_gen_outputs)


def get_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with variable_scope.variable_scope('generator') as gen_scope:
    gen_var = variable_scope.get_variable('dummy_var', initializer=0.0)
  with variable_scope.variable_scope('discriminator') as dis_scope:
    dis_var = variable_scope.get_variable('dummy_var', initializer=0.0)
  return tfgan_tuples.GANModel(
      generator_inputs=None,
      generated_data=array_ops.ones([3, 4]),
      generator_variables=[gen_var],
      generator_scope=gen_scope,
      generator_fn=None,
      real_data=None,
      discriminator_real_outputs=array_ops.ones([1, 2, 3]) * dis_var,
      discriminator_gen_outputs=array_ops.ones([1, 2, 3]) * gen_var * dis_var,
      discriminator_variables=[dis_var],
      discriminator_scope=dis_scope,
      discriminator_fn=None)


class GANHeadTest(test.TestCase):

  def setUp(self):
    super(GANHeadTest, self).setUp()
    self.gan_head = head.gan_head(
        generator_loss_fn=dummy_loss,
        discriminator_loss_fn=dummy_loss,
        generator_optimizer=training.GradientDescentOptimizer(1.0),
        discriminator_optimizer=training.GradientDescentOptimizer(1.0),
        get_eval_metric_ops_fn=self.get_metrics)
    self.assertTrue(isinstance(self.gan_head, head.GANHead))

  def get_metrics(self, gan_model):
    self.assertTrue(isinstance(gan_model, tfgan_tuples.GANModel))
    return {}

  def _test_modes_helper(self, mode):
    return self.gan_head.create_estimator_spec(
        features=None,
        mode=mode,
        logits=get_gan_model())

  def test_modes_predict(self):
    spec = self._test_modes_helper(model_fn_lib.ModeKeys.PREDICT)
    self.assertItemsEqual(('predict',), spec.export_outputs.keys())

  def test_modes_eval(self):
    self._test_modes_helper(model_fn_lib.ModeKeys.EVAL)

  def test_modes_train(self):
    self._test_modes_helper(model_fn_lib.ModeKeys.TRAIN)


if __name__ == '__main__':
  test.main()
