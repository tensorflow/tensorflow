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
"""Named tuples for TFGAN.

TFGAN training occurs in four steps, and each step communicates with the next
step via one of these named tuples. At each step, you can either use a TFGAN
helper function in `train.py`, or you can manually construct a tuple.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

__all__ = [
    'GANModel',
    'InfoGANModel',
    'ACGANModel',
    'CycleGANModel',
    'StarGANModel',
    'GANLoss',
    'CycleGANLoss',
    'GANTrainOps',
    'GANTrainSteps',
]


class GANModel(
    collections.namedtuple('GANModel', (
        'generator_inputs',
        'generated_data',
        'generator_variables',
        'generator_scope',
        'generator_fn',
        'real_data',
        'discriminator_real_outputs',
        'discriminator_gen_outputs',
        'discriminator_variables',
        'discriminator_scope',
        'discriminator_fn',
    ))):
  """A GANModel contains all the pieces needed for GAN training.

  Generative Adversarial Networks (https://arxiv.org/abs/1406.2661) attempt
  to create an implicit generative model of data by solving a two agent game.
  The generator generates candidate examples that are supposed to match the
  data distribution, and the discriminator aims to tell the real examples
  apart from the generated samples.

  Args:
    generator_inputs: The random noise source that acts as input to the
      generator.
    generated_data: The generated output data of the GAN.
    generator_variables: A list of all generator variables.
    generator_scope: Variable scope all generator variables live in.
    generator_fn: The generator function.
    real_data: A tensor or real data.
    discriminator_real_outputs: The discriminator's output on real data.
    discriminator_gen_outputs: The discriminator's output on generated data.
    discriminator_variables: A list of all discriminator variables.
    discriminator_scope: Variable scope all discriminator variables live in.
    discriminator_fn: The discriminator function.
  """


# TODO(joelshor): Have this class inherit from `GANModel`.
class InfoGANModel(
    collections.namedtuple('InfoGANModel', GANModel._fields + (
        'structured_generator_inputs',
        'predicted_distributions',
        'discriminator_and_aux_fn',
    ))):
  """An InfoGANModel contains all the pieces needed for InfoGAN training.

  See https://arxiv.org/abs/1606.03657 for more details.

  Args:
    structured_generator_inputs: A list of Tensors representing the random noise
      that must  have high mutual information with the generator output. List
      length should match `predicted_distributions`.
    predicted_distributions: A list of `tfp.distributions.Distribution`s.
      Predicted by the recognizer, and used to evaluate the likelihood of the
      structured noise. List length should match `structured_generator_inputs`.
    discriminator_and_aux_fn: The original discriminator function that returns
      a tuple of (logits, `predicted_distributions`).
  """


class ACGANModel(
    collections.namedtuple('ACGANModel', GANModel._fields +
                           ('one_hot_labels',
                            'discriminator_real_classification_logits',
                            'discriminator_gen_classification_logits',))):
  """An ACGANModel contains all the pieces needed for ACGAN training.

  See https://arxiv.org/abs/1610.09585 for more details.

  Args:
    one_hot_labels: A Tensor holding one-hot-labels for the batch.
    discriminator_real_classification_logits: Classification logits for real
      data.
    discriminator_gen_classification_logits: Classification logits for generated
      data.
  """


class CycleGANModel(
    collections.namedtuple(
        'CycleGANModel',
        ('model_x2y', 'model_y2x', 'reconstructed_x', 'reconstructed_y'))):
  """An CycleGANModel contains all the pieces needed for CycleGAN training.

  The model `model_x2y` generator F maps data set X to Y, while the model
  `model_y2x` generator G maps data set Y to X.

  See https://arxiv.org/abs/1703.10593 for more details.

  Args:
    model_x2y: A `GANModel` namedtuple whose generator maps data set X to Y.
    model_y2x: A `GANModel` namedtuple whose generator maps data set Y to X.
    reconstructed_x: A `Tensor` of reconstructed data X which is G(F(X)).
    reconstructed_y: A `Tensor` of reconstructed data Y which is F(G(Y)).
  """


class StarGANModel(
    collections.namedtuple('StarGANModel', (
        'input_data',
        'input_data_domain_label',
        'generated_data',
        'generated_data_domain_target',
        'reconstructed_data',
        'discriminator_input_data_source_predication',
        'discriminator_generated_data_source_predication',
        'discriminator_input_data_domain_predication',
        'discriminator_generated_data_domain_predication',
        'generator_variables',
        'generator_scope',
        'generator_fn',
        'discriminator_variables',
        'discriminator_scope',
        'discriminator_fn',
    ))):
  """A StarGANModel contains all the pieces needed for StarGAN training.

  Args:
    input_data: The real images that need to be transferred by the generator.
    input_data_domain_label: The real domain labels associated with the real
      images.
    generated_data: The generated images produced by the generator. It has the
      same shape as the input_data.
    generated_data_domain_target: The target domain that the generated images
      belong to. It has the same shape as the input_data_domain_label.
    reconstructed_data: The reconstructed images produced by the G(enerator).
      reconstructed_data = G(G(input_data, generated_data_domain_target),
      input_data_domain_label).
    discriminator_input_data_source: The discriminator's output for predicting
      the source (real/generated) of input_data.
    discriminator_generated_data_source: The discriminator's output for
      predicting the source (real/generated) of  generated_data.
    discriminator_input_data_domain_predication: The discriminator's output for
      predicting the domain_label for the input_data.
    discriminator_generated_data_domain_predication: The discriminatorr's output
      for predicting the domain_target for the generated_data.
    generator_variables: A list of all generator variables.
    generator_scope: Variable scope all generator variables live in.
    generator_fn: The generator function.
    discriminator_variables: A list of all discriminator variables.
    discriminator_scope: Variable scope all discriminator variables live in.
    discriminator_fn: The discriminator function.
  """


class GANLoss(
    collections.namedtuple('GANLoss', (
        'generator_loss',
        'discriminator_loss'
    ))):
  """GANLoss contains the generator and discriminator losses.

  Args:
    generator_loss: A tensor for the generator loss.
    discriminator_loss: A tensor for the discriminator loss.
  """


class CycleGANLoss(
    collections.namedtuple('CycleGANLoss', ('loss_x2y', 'loss_y2x'))):
  """CycleGANLoss contains the losses for `CycleGANModel`.

  See https://arxiv.org/abs/1703.10593 for more details.

  Args:
    loss_x2y: A `GANLoss` namedtuple representing the loss of `model_x2y`.
    loss_y2x: A `GANLoss` namedtuple representing the loss of `model_y2x`.
  """


class GANTrainOps(
    collections.namedtuple('GANTrainOps', (
        'generator_train_op',
        'discriminator_train_op',
        'global_step_inc_op'
    ))):
  """GANTrainOps contains the training ops.

  Args:
    generator_train_op: Op that performs a generator update step.
    discriminator_train_op: Op that performs a discriminator update step.
    global_step_inc_op: Op that increments the shared global step.
  """


class GANTrainSteps(
    collections.namedtuple('GANTrainSteps', (
        'generator_train_steps',
        'discriminator_train_steps'
    ))):
  """Contains configuration for the GAN Training.

  Args:
    generator_train_steps: Number of generator steps to take in each GAN step.
    discriminator_train_steps: Number of discriminator steps to take in each GAN
      step.
  """
