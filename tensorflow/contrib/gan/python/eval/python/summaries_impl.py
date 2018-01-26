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
"""Common TFGAN summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.eval.python import eval_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as loss_util
from tensorflow.python.summary import summary

__all__ = [
    'add_gan_model_image_summaries',
    'add_image_comparison_summaries',
    'add_gan_model_summaries',
    'add_regularization_loss_summaries',
]


def _assert_is_image(data):
  data.shape.assert_has_rank(4)
  data.shape[1:].assert_is_fully_defined()


def add_gan_model_image_summaries(gan_model, grid_size=4):
  """Adds image summaries for real and fake images.

  Args:
    gan_model: A GANModel tuple.
    grid_size: The size of an image grid.

  Raises:
    ValueError: If real and generated data aren't images.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    saved_params = locals()
    saved_params.pop('gan_model', None)
    with ops.name_scope('cyclegan_x2y_image_summaries'):
      add_gan_model_image_summaries(gan_model.model_x2y, **saved_params)
    with ops.name_scope('cyclegan_y2x_image_summaries'):
      add_gan_model_image_summaries(gan_model.model_y2x, **saved_params)
    return

  _assert_is_image(gan_model.real_data)
  _assert_is_image(gan_model.generated_data)

  num_images = grid_size ** 2
  real_image_shape = gan_model.real_data.shape.as_list()[1:3]
  generated_image_shape = gan_model.generated_data.shape.as_list()[1:3]
  real_channels = gan_model.real_data.shape.as_list()[3]
  generated_channels = gan_model.generated_data.shape.as_list()[3]

  summary.image(
      'real_data',
      eval_utils.image_grid(
          gan_model.real_data[:num_images],
          grid_shape=(grid_size, grid_size),
          image_shape=real_image_shape,
          num_channels=real_channels),
      max_outputs=1)
  summary.image(
      'generated_data',
      eval_utils.image_grid(
          gan_model.generated_data[:num_images],
          grid_shape=(grid_size, grid_size),
          image_shape=generated_image_shape,
          num_channels=generated_channels),
      max_outputs=1)
  add_gan_model_summaries(gan_model)


def add_image_comparison_summaries(gan_model, num_comparisons=2,
                                   display_diffs=False):
  """Adds image summaries to compare triplets of images.

  The first image is the generator input, the second is the generator output,
  and the third is the real data. This style of comparison is useful for
  image translation problems, where the generator input is a corrupted image,
  the generator output is the reconstruction, and the real data is the target.

  Args:
    gan_model: A GANModel tuple.
    num_comparisons: The number of image triplets to display.
    display_diffs: Also display the difference between generated and target.

  Raises:
    ValueError: If real data, generated data, and generator inputs aren't
      images.
    ValueError: If the generator input, real, and generated data aren't all the
      same size.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    saved_params = locals()
    saved_params.pop('gan_model', None)
    with ops.name_scope('cyclegan_x2y_image_comparison_summaries'):
      add_image_comparison_summaries(gan_model.model_x2y, **saved_params)
    with ops.name_scope('cyclegan_y2x_image_comparison_summaries'):
      add_image_comparison_summaries(gan_model.model_y2x, **saved_params)
    return

  _assert_is_image(gan_model.generator_inputs)
  _assert_is_image(gan_model.generated_data)
  _assert_is_image(gan_model.real_data)

  gan_model.generated_data.shape.assert_is_compatible_with(
      gan_model.generator_inputs.shape)
  gan_model.real_data.shape.assert_is_compatible_with(
      gan_model.generated_data.shape)

  image_list = []
  image_list.extend(
      array_ops.unstack(gan_model.generator_inputs[:num_comparisons]))
  image_list.extend(
      array_ops.unstack(gan_model.generated_data[:num_comparisons]))
  image_list.extend(array_ops.unstack(gan_model.real_data[:num_comparisons]))
  if display_diffs:
    generated_list = array_ops.unstack(
        gan_model.generated_data[:num_comparisons])
    real_list = array_ops.unstack(gan_model.real_data[:num_comparisons])
    diffs = [
        math_ops.abs(math_ops.to_float(generated) - math_ops.to_float(real)) for
        generated, real in zip(generated_list, real_list)]
    image_list.extend(diffs)

  # Reshape image and display.
  summary.image(
      'image_comparison',
      eval_utils.image_reshaper(image_list, num_cols=num_comparisons),
      max_outputs=1)


def add_gan_model_summaries(gan_model):
  """Adds typical GANModel summaries.

  Args:
    gan_model: A GANModel tuple.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    with ops.name_scope('cyclegan_x2y_summaries'):
      add_gan_model_summaries(gan_model.model_x2y)
    with ops.name_scope('cyclegan_y2x_summaries'):
      add_gan_model_summaries(gan_model.model_y2x)
    return

  with ops.name_scope('generator_variables'):
    for var in gan_model.generator_variables:
      summary.histogram(var.name, var)
  with ops.name_scope('discriminator_variables'):
    for var in gan_model.discriminator_variables:
      summary.histogram(var.name, var)


def add_regularization_loss_summaries(gan_model):
  """Adds summaries for a regularization losses..

  Args:
    gan_model: A GANModel tuple.
  """
  if isinstance(gan_model, namedtuples.CycleGANModel):
    with ops.name_scope('cyclegan_x2y_regularization_loss_summaries'):
      add_regularization_loss_summaries(gan_model.model_x2y)
    with ops.name_scope('cyclegan_y2x_regularization_loss_summaries'):
      add_regularization_loss_summaries(gan_model.model_y2x)
    return

  if gan_model.generator_scope:
    summary.scalar(
        'generator_regularization_loss',
        loss_util.get_regularization_loss(gan_model.generator_scope.name))
  if gan_model.discriminator_scope:
    summary.scalar(
        'discriminator_regularization_loss',
        loss_util.get_regularization_loss(gan_model.discriminator_scope.name))
