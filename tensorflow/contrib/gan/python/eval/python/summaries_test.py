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
"""Tests for TFGAN summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.eval.python import summaries_impl as summaries
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary


def generator_model(inputs):
  return variable_scope.get_variable('dummy_g', initializer=2.0) * inputs


def discriminator_model(inputs, _):
  return variable_scope.get_variable('dummy_d', initializer=2.0) * inputs


def get_gan_model():
  # TODO(joelshor): Find a better way of creating a variable scope.
  with variable_scope.variable_scope('generator') as gen_scope:
    pass
  with variable_scope.variable_scope('discriminator') as dis_scope:
    pass
  return namedtuples.GANModel(
      generator_inputs=array_ops.zeros([4, 32, 32, 3]),
      generated_data=array_ops.zeros([4, 32, 32, 3]),
      generator_variables=[variables.Variable(0), variables.Variable(1)],
      generator_scope=gen_scope,
      generator_fn=generator_model,
      real_data=array_ops.ones([4, 32, 32, 3]),
      discriminator_real_outputs=array_ops.ones([1, 2, 3]),
      discriminator_gen_outputs=array_ops.ones([1, 2, 3]),
      discriminator_variables=[variables.Variable(0)],
      discriminator_scope=dis_scope,
      discriminator_fn=discriminator_model)


def get_cyclegan_model():
  with variable_scope.variable_scope('x2y'):
    model_x2y = get_gan_model()
  with variable_scope.variable_scope('y2x'):
    model_y2x = get_gan_model()
  return namedtuples.CycleGANModel(
      model_x2y=model_x2y,
      model_y2x=model_y2x,
      reconstructed_x=array_ops.zeros([3, 30, 35, 6]),
      reconstructed_y=array_ops.zeros([3, 30, 35, 6]))


class SummariesTest(test.TestCase):

  def _test_add_gan_model_image_summaries_impl(self, get_model_fn,
                                               expected_num_summary_ops):
    summaries.add_gan_model_image_summaries(get_model_fn(), grid_size=2)

    self.assertEquals(expected_num_summary_ops,
                      len(ops.get_collection(ops.GraphKeys.SUMMARIES)))
    with self.test_session(use_gpu=True):
      variables.global_variables_initializer().run()
      summary.merge_all().eval()

  def test_add_gan_model_image_summaries(self):
    self._test_add_gan_model_image_summaries_impl(get_gan_model, 5)

  def test_add_gan_model_image_summaries_for_cyclegan(self):
    self._test_add_gan_model_image_summaries_impl(get_cyclegan_model, 10)

  def _test_add_gan_model_summaries_impl(self, get_model_fn,
                                         expected_num_summary_ops):
    summaries.add_gan_model_summaries(get_model_fn())

    self.assertEquals(expected_num_summary_ops,
                      len(ops.get_collection(ops.GraphKeys.SUMMARIES)))
    with self.test_session(use_gpu=True):
      variables.global_variables_initializer().run()
      summary.merge_all().eval()

  def test_add_gan_model_summaries(self):
    self._test_add_gan_model_summaries_impl(get_gan_model, 3)

  def test_add_gan_model_summaries_for_cyclegan(self):
    self._test_add_gan_model_summaries_impl(get_cyclegan_model, 6)

  def _test_add_regularization_loss_summaries_impl(self, get_model_fn,
                                                   expected_num_summary_ops):
    summaries.add_regularization_loss_summaries(get_model_fn())

    self.assertEquals(expected_num_summary_ops,
                      len(ops.get_collection(ops.GraphKeys.SUMMARIES)))
    with self.test_session(use_gpu=True):
      summary.merge_all().eval()

  def test_add_regularization_loss_summaries(self):
    self._test_add_regularization_loss_summaries_impl(get_gan_model, 2)

  def test_add_regularization_loss_summaries_for_cyclegan(self):
    self._test_add_regularization_loss_summaries_impl(get_cyclegan_model, 4)

  # TODO(joelshor): Add correctness test.
  def _test_add_image_comparison_summaries_impl(self, get_model_fn,
                                                expected_num_summary_ops):
    summaries.add_image_comparison_summaries(get_model_fn(), display_diffs=True)

    self.assertEquals(expected_num_summary_ops,
                      len(ops.get_collection(ops.GraphKeys.SUMMARIES)))
    with self.test_session(use_gpu=True):
      summary.merge_all().eval()

  def test_add_image_comparison_summaries(self):
    self._test_add_image_comparison_summaries_impl(get_gan_model, 1)

  def test_add_image_comparison_summaries_for_cyclegan(self):
    self._test_add_image_comparison_summaries_impl(get_cyclegan_model, 2)


if __name__ == '__main__':
  test.main()
