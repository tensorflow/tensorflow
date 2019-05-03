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
"""Tests for latent_gan_estimator.

See g3.tp.tensorflow.contrib.gan.python.estimator.python.latent_gan_estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import numpy as np
from tensorflow.contrib.gan.python.estimator.python import latent_gan_estimator
from tensorflow.python.estimator import run_config as run_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import training


class TrainInputEstimatorTest(test.TestCase):

  def test_get_input_training_estimator(self):
    """Integration test to make sure the input_training_estimator works."""

    # Create dummy test input tensors.
    true_features = np.reshape(np.random.uniform(size=100), (10, 10))
    true_labels = np.reshape(np.random.uniform(size=100), (5, 20))
    expected_z_output = [[1, -1], [-1, 1]]

    # Fill out required parameters randomly, includes optimizer kwargs.
    params = {
        'batch_size': 2,
        'z_shape': [2],
        'learning_rate': 1.0,
        'input_clip': 1.0,
        'add_summaries': False,
        'opt_kwargs': {
            'beta1': 0.1
        }
    }

    input_z_shape = [params['batch_size']] + params['z_shape']

    # Create dummy model functions that represent an underlying GANEstimator and
    # the input training wrapper. Make sure that everything is wired up
    # correctly in the internals of each dummy function.
    def _generator(net, mode):
      """The generator function will get the newly created z variable."""
      del mode
      self.assertSequenceEqual(net.shape, input_z_shape)
      gen_dummy_var = variable_scope.get_variable(
          name='generator_dummy_variable',
          initializer=array_ops.ones(input_z_shape))
      return net * gen_dummy_var

    def _discriminator(net, condition, mode):
      """The discriminator function will get either the z variable or labels."""
      del condition, mode
      try:
        self.assertSequenceEqual(net.shape, true_labels.shape)
      except AssertionError:
        self.assertSequenceEqual(net.shape, input_z_shape)
      return net

    def _loss(gan_model, features, labels, _):
      """Make sure that features and labels are passed in from input."""
      self.assertTrue(np.array_equal(features, true_features))
      self.assertTrue(np.array_equal(labels, true_labels))
      return losses.absolute_difference(expected_z_output,
                                        gan_model.generated_data)

    optimizer = training.AdamOptimizer

    # We are not loading checkpoints, so set the corresponding directory to a
    # dummy directories.
    tmp_dir = tempfile.mkdtemp()
    config = run_config.RunConfig(model_dir=tmp_dir,
                                  save_summary_steps=None,
                                  save_checkpoints_steps=1,
                                  save_checkpoints_secs=None)

    # Get the estimator. Disable warm start so that there is no attempted
    # checkpoint reloading.
    estimator = latent_gan_estimator.get_latent_gan_estimator(
        _generator, _discriminator, _loss, optimizer, params, config, tmp_dir,
        warmstart_options=None)

    # Train for a few steps.
    def dummy_input():
      return true_features, true_labels
    estimator.train(input_fn=dummy_input, steps=10)

    # Make sure the generator variables did not change, but the z variables did
    # change.
    self.assertTrue(np.array_equal(
        estimator.get_variable_value('Generator/generator_dummy_variable'),
        np.ones(input_z_shape)))
    self.assertTrue(np.array_equal(
        estimator.get_variable_value('new_var_z_input'),
        expected_z_output))


if __name__ == '__main__':
  test.main()
