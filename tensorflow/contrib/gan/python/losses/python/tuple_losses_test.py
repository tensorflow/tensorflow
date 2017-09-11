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

from tensorflow.contrib.gan.python.losses.python import tuple_losses_impl as tfgan_losses

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


class ConsistentLossesTest(test.TestCase):

  pass


def _tuple_from_dict(args_dict):
  return collections.namedtuple('Tuple', args_dict.keys())(**args_dict)


def add_loss_consistency_test(test_class, loss_name_str, loss_args):
  tuple_loss = getattr(tfgan_losses, loss_name_str)
  arg_loss = getattr(tfgan_losses.losses_impl, loss_name_str)

  def consistency_test(self):
    self.assertEqual(arg_loss.__name__, tuple_loss.__name__)
    with self.test_session():
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


if __name__ == '__main__':
  for loss_name in tfgan_losses.__all__:
    if loss_name in manual_tests: continue
    keyword_args = (generator_keyword_args if 'generator' in loss_name else
                    discriminator_keyword_args)
    add_loss_consistency_test(ConsistentLossesTest, loss_name, keyword_args)

  test.main()
