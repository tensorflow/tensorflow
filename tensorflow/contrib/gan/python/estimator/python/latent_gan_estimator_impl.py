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
"""Implements an estimator wrapper that allows training the input latent space.

This file implements a latent gan estimator that wraps around a previously
trained GAN. The latent gan estimator trains a single variable z, representing
the hidden latent distribution that is the 'noise' input to the GAN. By training
z, the inpainting estimator can move around the latent z space towards
minimizing a specific loss function.

The latent gan estimator has a few key differences from a normal estimator.

First: the variables in the estimator should not be saved, as we are not
updating the original GAN and are only adding a new z variable that is meant
to be different for each run. In order to do distributed training using
train_and_evaluate, the Tensorflow RunConfig is expected to save checkpoints
by having either save_checkpoints_steps or save_checkpoints_secs saved.
To avoid this conflict, we purposely set the save_checkpoints_steps value in
the RunConfig to be one step more than the total number of steps that the
inpainter estimator will run.

Second: we need to specify warm start settings, as we are reloading the
GAN model into a different graph (specifically, one with a new z variable).
The warm start settings defined below reload all GAN variables and ignore the
new z variable (and the optimizer).

Usage:

  def _generator(net, mode):
    ...

  def _discriminator(net, condition, mode):
    ...

  def _loss(gan_model, features, labels, add_summaries):
    ...

  def optimizer():
    ...

  params = {<required params>}
  config = tf.estimator.RunConfig()
  tmp_dir = path/to/output/storage

  estimator = latent_gan_estimator.get_latent_gan_estimator(
      _generator, _discriminator, _loss, optimizer, params, config, tmp_dir)

  def input_fn():
    ...

  estimator.train(input_fn=input_fn)

See latent_gan_estimator_test.py or tensorflow_models/gan/face_inpainting for
further examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from tensorflow.contrib.gan.python import train as tfgan_train
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util


INPUT_NAME = 'new_var_z_input'  # The name for the new z space input variable.
OPTIMIZER_NAME = 'latent_gan_optimizer'  # The name for the new optimizer vars.

__all__ = [
    'get_latent_gan_estimator',
]


def _get_latent_gan_model_fn(generator_fn, discriminator_fn, loss_fn,
                             optimizer):
  """Sets up a model function that wraps around a given GAN."""
  def model_fn(features, labels, mode, params):
    """Model function defining an inpainting estimator."""
    batch_size = params['batch_size']
    z_shape = [batch_size] + params['z_shape']
    add_summaries = params['add_summaries']
    input_clip = params['input_clip']

    z = variable_scope.get_variable(
        name=INPUT_NAME, initializer=random_ops.truncated_normal(z_shape),
        constraint=lambda x: clip_ops.clip_by_value(x, -input_clip, input_clip))

    generator = functools.partial(generator_fn, mode=mode)
    discriminator = functools.partial(discriminator_fn, mode=mode)
    gan_model = tfgan_train.gan_model(generator_fn=generator,
                                      discriminator_fn=discriminator,
                                      real_data=labels,
                                      generator_inputs=z,
                                      check_shapes=False)

    loss = loss_fn(gan_model, features, labels, add_summaries)

    # Use a variable scope to make sure that estimator variables dont cause
    # save/load problems when restoring from ckpts.
    with variable_scope.variable_scope(OPTIMIZER_NAME):
      opt = optimizer(learning_rate=params['learning_rate'],
                      **params['opt_kwargs'])
      train_op = opt.minimize(
          loss=loss, global_step=training_util.get_or_create_global_step(),
          var_list=[z])

    if add_summaries:
      z_grads = gradients_impl.gradients(loss, z)
      summary.scalar('z_loss/z_grads', clip_ops.global_norm(z_grads))
      summary.scalar('z_loss/loss', loss)

    return model_fn_lib.EstimatorSpec(mode=mode,
                                      predictions=gan_model.generated_data,
                                      loss=loss,
                                      train_op=train_op)
  return model_fn


def get_latent_gan_estimator(generator_fn, discriminator_fn, loss_fn,
                             optimizer, params, config, ckpt_dir,
                             warmstart_options=True):
  """Gets an estimator that passes gradients to the input.

  This function takes in a generator and adds a trainable z variable that is
  used as input to this generator_fn. The generator itself is treated as a black
  box through which gradients can pass through without updating any weights. The
  result is a trainable way to traverse the GAN latent space. The loss_fn is
  used to actually train the z variable. The generator_fn and discriminator_fn
  should be previously trained by the tfgan library (on reload, the variables
  are expected to follow the tfgan format. It may be possible to use the
  latent gan estimator with entirely custom GANs that do not use the tfgan
  library as long as the appropriate variables are wired properly).

  Args:
    generator_fn: a function defining a Tensorflow graph for a GAN generator.
      The weights defined in this graph should already be defined in the given
      checkpoint location. Should have 'mode' as an argument.
    discriminator_fn: a function defining a Tensorflow graph for a GAN
      discriminator. Should have 'mode' as an argument.
    loss_fn: a function defining a Tensorflow graph for a GAN loss. Takes in a
      GANModel tuple, features, labels, and add_summaries as inputs.
    optimizer: a tf.Optimizer or a function that returns a tf.Optimizer with no
      inputs.
   params: An object containing the following parameters:
      - batch_size: an int indicating the size of the training batch.
      - z_shape: the desired shape of the input z values (not counting batch).
      - learning_rate: a scalar or function defining a learning rate applied to
        optimizer.
      - input_clip: the amount to clip the x training variable by.
      - add_summaries: whether or not to add summaries.
      - opt_kwargs: optimizer kwargs.
    config: tf.RunConfig. Should point model to output dir and should indicate
     whether to save checkpoints (to avoid saving checkpoints, set
     save_checkpoints_steps to a number larger than the number of train steps).
     The model_dir field in the RunConfig should point to a directory WITHOUT
     any saved checkpoints.
    ckpt_dir: the directory where the model checkpoints live. The checkpoint is
     used to warm start the underlying GAN. This should NOT be the same as
     config.model_dir.
    warmstart_options: boolean, None, or a WarmStartSettings object. If set to
      True, uses a default WarmStartSettings object. If set to False or None,
      does not use warm start. If using a custom WarmStartSettings object, make
      sure that new variables are properly accounted for when reloading the
      underlying GAN. Defaults to True.
  Returns:
    An estimator spec defining a GAN input training estimator.
  """
  model_fn = _get_latent_gan_model_fn(generator_fn, discriminator_fn,
                                      loss_fn, optimizer)

  if isinstance(warmstart_options, estimator.WarmStartSettings):
    ws = warmstart_options
  elif warmstart_options:
    # Default WarmStart loads all variable names except INPUT_NAME and
    # OPTIMIZER_NAME.
    var_regex = '^(?!.*(%s|%s).*)' % (INPUT_NAME, OPTIMIZER_NAME)
    ws = estimator.WarmStartSettings(ckpt_to_initialize_from=ckpt_dir,
                                     vars_to_warm_start=var_regex)
  else:
    ws = None

  if 'opt_kwargs' not in params:
    params['opt_kwargs'] = {}

  return estimator.Estimator(model_fn=model_fn, config=config, params=params,
                             warm_start_from=ws)
