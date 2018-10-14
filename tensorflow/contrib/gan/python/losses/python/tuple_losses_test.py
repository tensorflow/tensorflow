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
"""Tests for contrib.gan.python.losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.losses.python import losses_impl as tfgan_losses_impl
from tensorflow.contrib.gan.python.losses.python import tuple_losses_impl as tfgan_losses
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ArgsToGanModelTest(test.TestCase):

  def test_args_to_gan_model(self):
    """Test `_args_to_gan_model`."""
    tuple_type = collections.namedtuple('fake_type', ['arg1', 'arg3'])

    def args_loss(arg1, arg2, arg3=3, arg4=4):
      return arg1 + arg2 + arg3 + arg4

    gan_model_loss = tfgan_losses._args_to_gan_model(args_loss)

    # Value is correct.
    self.assertEqual(1 + 2 + 5 + 6,
                     gan_model_loss(tuple_type(1, 2), arg2=5, arg4=6))

    # Uses tuple argument with defaults.
    self.assertEqual(1 + 5 + 3 + 7,
                     gan_model_loss(tuple_type(1, None), arg2=5, arg4=7))

    # Uses non-tuple argument with defaults.
    self.assertEqual(1 + 5 + 2 + 4,
                     gan_model_loss(tuple_type(1, 2), arg2=5))

    # Requires non-tuple, non-default arguments.
    with self.assertRaisesRegexp(ValueError, '`arg2` must be supplied'):
      gan_model_loss(tuple_type(1, 2))

    # Can't pass tuple argument outside tuple.
    with self.assertRaisesRegexp(
        ValueError, 'present in both the tuple and keyword args'):
      gan_model_loss(tuple_type(1, 2), arg2=1, arg3=5)

  def test_args_to_gan_model_name(self):
    """Test that `_args_to_gan_model` produces correctly named functions."""
    def loss_fn(x):
      return x
    new_loss_fn = tfgan_losses._args_to_gan_model(loss_fn)
    self.assertEqual('loss_fn', new_loss_fn.__name__)
    self.assertTrue('The gan_model version of' in new_loss_fn.__docstring__)

  def test_tuple_respects_optional_args(self):
    """Test that optional args can be changed with tuple losses."""
    tuple_type = collections.namedtuple('fake_type', ['arg1', 'arg2'])
    def args_loss(arg1, arg2, arg3=3):
      return arg1 + 2 * arg2 + 3 * arg3

    loss_fn = tfgan_losses._args_to_gan_model(args_loss)
    loss = loss_fn(tuple_type(arg1=-1, arg2=2), arg3=4)

    # If `arg3` were not set properly, this value would be different.
    self.assertEqual(-1 + 2 * 2 + 3 * 4, loss)

  def test_works_with_child_classes(self):
    """`args_to_gan_model` should work with classes derived from namedtuple."""
    tuple_type = collections.namedtuple('fake_type', ['arg1', 'arg2'])

    class InheritedType(tuple_type):
      pass
    def args_loss(arg1, arg2, arg3=3):
      return arg1 + 2 * arg2 + 3 * arg3

    loss_fn = tfgan_losses._args_to_gan_model(args_loss)
    loss = loss_fn(InheritedType(arg1=-1, arg2=2), arg3=4)

    # If `arg3` were not set properly, this value would be different.
    self.assertEqual(-1 + 2 * 2 + 3 * 4, loss)


class ConsistentLossesTest(test.TestCase):

  pass


def _tuple_from_dict(args_dict):
  return collections.namedtuple('Tuple', args_dict.keys())(**args_dict)


def add_loss_consistency_test(test_class, loss_name_str, loss_args):
  tuple_loss = getattr(tfgan_losses, loss_name_str)
  arg_loss = getattr(tfgan_losses.losses_impl, loss_name_str)

  def consistency_test(self):
    self.assertEqual(arg_loss.__name__, tuple_loss.__name__)
    with self.cached_session():
      self.assertEqual(arg_loss(**loss_args).eval(),
                       tuple_loss(_tuple_from_dict(loss_args)).eval())

  test_name = 'test_loss_consistency_%s' %  loss_name_str
  setattr(test_class, test_name, consistency_test)


# A list of consistency tests which need to be manually written.
manual_tests = [
    'acgan_discriminator_loss',
    'acgan_generator_loss',
    'combine_adversarial_loss',
    'mutual_information_penalty',
    'wasserstein_gradient_penalty',
    'cycle_consistency_loss',
    'stargan_generator_loss_wrapper',
    'stargan_discriminator_loss_wrapper',
    'stargan_gradient_penalty_wrapper'
]

discriminator_keyword_args = {
    'discriminator_real_outputs': np.array([[3.4, 2.3, -2.3],
                                            [6.3, -2.1, 0.2]]),
    'discriminator_gen_outputs': np.array([[6.2, -1.5, 2.3],
                                           [-2.9, -5.1, 0.1]]),
}
generator_keyword_args = {
    'discriminator_gen_outputs': np.array([[6.2, -1.5, 2.3],
                                           [-2.9, -5.1, 0.1]]),
}


class CycleConsistencyLossTest(test.TestCase):

  def setUp(self):
    super(CycleConsistencyLossTest, self).setUp()

    def _partial_model(generator_inputs_np):
      model = namedtuples.GANModel(*[None] * 11)
      return model._replace(
          generator_inputs=constant_op.constant(
              generator_inputs_np, dtype=dtypes.float32))

    self._model_x2y = _partial_model([1, 2])
    self._model_y2x = _partial_model([5, 6])

  def test_model_type(self):
    """Test the input model type for `cycle_consistency_loss`."""
    with self.assertRaises(ValueError):
      tfgan_losses.cycle_consistency_loss(self._model_x2y)

  def test_correct_loss(self):
    """Test the output of `cycle_consistency_loss`."""
    loss = tfgan_losses.cycle_consistency_loss(
        namedtuples.CycleGANModel(
            model_x2y=self._model_x2y,
            model_y2x=self._model_y2x,
            reconstructed_x=constant_op.constant([9, 8], dtype=dtypes.float32),
            reconstructed_y=constant_op.constant([7, 2], dtype=dtypes.float32)))
    with self.test_session(use_gpu=True):
      variables.global_variables_initializer().run()
      self.assertNear(5.0, loss.eval(), 1e-5)


class StarGANLossWrapperTest(test.TestCase):

  def setUp(self):

    super(StarGANLossWrapperTest, self).setUp()

    self.input_data = array_ops.ones([1, 2, 2, 3])
    self.input_data_domain_label = constant_op.constant([[0, 1]])
    self.generated_data = array_ops.ones([1, 2, 2, 3])
    self.discriminator_input_data_source_predication = array_ops.ones([1])
    self.discriminator_generated_data_source_predication = array_ops.ones([1])

    def _discriminator_fn(inputs, num_domains):
      """Differentiable dummy discriminator for StarGAN."""
      hidden = layers.flatten(inputs)
      output_src = math_ops.reduce_mean(hidden, axis=1)
      output_cls = layers.fully_connected(
          inputs=hidden,
          num_outputs=num_domains,
          activation_fn=None,
          normalizer_fn=None,
          biases_initializer=None)
      return output_src, output_cls

    with variable_scope.variable_scope('discriminator') as dis_scope:
      pass

    self.model = namedtuples.StarGANModel(
        input_data=self.input_data,
        input_data_domain_label=self.input_data_domain_label,
        generated_data=self.generated_data,
        generated_data_domain_target=None,
        reconstructed_data=None,
        discriminator_input_data_source_predication=self.
        discriminator_input_data_source_predication,
        discriminator_generated_data_source_predication=self.
        discriminator_generated_data_source_predication,
        discriminator_input_data_domain_predication=None,
        discriminator_generated_data_domain_predication=None,
        generator_variables=None,
        generator_scope=None,
        generator_fn=None,
        discriminator_variables=None,
        discriminator_scope=dis_scope,
        discriminator_fn=_discriminator_fn)

    self.discriminator_fn = _discriminator_fn
    self.discriminator_scope = dis_scope

  def test_stargan_generator_loss_wrapper(self):
    """Test StarGAN generator loss wrapper."""
    loss_fn = tfgan_losses_impl.wasserstein_generator_loss
    wrapped_loss_fn = tfgan_losses.stargan_generator_loss_wrapper(loss_fn)

    loss_result_tensor = loss_fn(
        self.discriminator_generated_data_source_predication)
    wrapped_loss_result_tensor = wrapped_loss_fn(self.model)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      loss_result, wrapped_loss_result = sess.run(
          [loss_result_tensor, wrapped_loss_result_tensor])
      self.assertAlmostEqual(loss_result, wrapped_loss_result)

  def test_stargan_discriminator_loss_wrapper(self):
    """Test StarGAN discriminator loss wrapper."""
    loss_fn = tfgan_losses_impl.wasserstein_discriminator_loss
    wrapped_loss_fn = tfgan_losses.stargan_discriminator_loss_wrapper(loss_fn)

    loss_result_tensor = loss_fn(
        self.discriminator_generated_data_source_predication,
        self.discriminator_generated_data_source_predication)
    wrapped_loss_result_tensor = wrapped_loss_fn(self.model)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      loss_result, wrapped_loss_result = sess.run(
          [loss_result_tensor, wrapped_loss_result_tensor])
      self.assertAlmostEqual(loss_result, wrapped_loss_result)

  def test_stargan_gradient_penalty_wrapper(self):
    """Test StaGAN gradient penalty wrapper.

    Notes:
      The random interpolates are handled by given setting the reconstruction to
      be the same as the input.

    """
    loss_fn = tfgan_losses_impl.wasserstein_gradient_penalty
    wrapped_loss_fn = tfgan_losses.stargan_gradient_penalty_wrapper(loss_fn)

    loss_result_tensor = loss_fn(
        real_data=self.input_data,
        generated_data=self.generated_data,
        generator_inputs=self.input_data_domain_label.shape.as_list()[-1],
        discriminator_fn=self.discriminator_fn,
        discriminator_scope=self.discriminator_scope)
    wrapped_loss_result_tensor = wrapped_loss_fn(self.model)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      loss_result, wrapped_loss_result = sess.run(
          [loss_result_tensor, wrapped_loss_result_tensor])
      self.assertAlmostEqual(loss_result, wrapped_loss_result)


if __name__ == '__main__':
  for loss_name in tfgan_losses.__all__:
    if loss_name in manual_tests: continue
    keyword_args = (generator_keyword_args if 'generator' in loss_name else
                    discriminator_keyword_args)
    add_loss_consistency_test(ConsistentLossesTest, loss_name, keyword_args)

  test.main()
