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
"""Tests for TF-GAN's TPU Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python.estimator.python import tpu_gan_estimator_impl as estimator
from tensorflow.contrib.gan.python.losses.python import tuple_losses as losses
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.estimator import WarmStartSettings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training
from tensorflow.python.training import training_util

FLAGS = flags.FLAGS

flags.DEFINE_bool('use_tpu', False, 'Whether to run test on TPU or not.')


def generator_fn(noise, mode):
  del mode
  return layers.fully_connected(noise, tensor_shape.dimension_value(
      noise.shape[1]))


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return layers.fully_connected(data, 1)


def get_dummy_gan_model():
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
      real_data=array_ops.zeros([3, 4]),
      discriminator_real_outputs=array_ops.ones([1, 2, 3]) * dis_var,
      discriminator_gen_outputs=array_ops.ones([1, 2, 3]) * gen_var * dis_var,
      discriminator_variables=[dis_var],
      discriminator_scope=dis_scope,
      discriminator_fn=None)


def get_metrics(generator_inputs, generated_data, real_data,
                discriminator_real_outputs, discriminator_gen_outputs):
  del generator_inputs, discriminator_real_outputs, discriminator_gen_outputs
  return {
      'mse_custom_metric': metrics_lib.mean_squared_error(
          real_data, generated_data)
  }


class GetTPUEstimatorSpecTest(test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    super(GetTPUEstimatorSpecTest, cls).setUpClass()
    cls._generator_optimizer = tpu_optimizer.CrossShardOptimizer(
        training.GradientDescentOptimizer(1.0))
    cls._discriminator_optimizer = tpu_optimizer.CrossShardOptimizer(
        training.GradientDescentOptimizer(1.0))

  @parameterized.named_parameters(
      ('joint_train', model_fn_lib.ModeKeys.TRAIN, True),
      ('train_sequential', model_fn_lib.ModeKeys.TRAIN, False),
      ('eval', model_fn_lib.ModeKeys.EVAL, None),
      ('predict', model_fn_lib.ModeKeys.PREDICT, None))
  def test_get_estimator_spec(self, mode, joint_train):
    with ops.Graph().as_default():
      self._gan_model = get_dummy_gan_model()
      spec = estimator._get_estimator_spec(
          mode,
          self._gan_model,
          generator_loss_fn=losses.wasserstein_generator_loss,
          discriminator_loss_fn=losses.wasserstein_discriminator_loss,
          get_eval_metric_ops_fn=get_metrics,
          generator_optimizer=self._generator_optimizer,
          discriminator_optimizer=self._discriminator_optimizer,
          joint_train=joint_train,
          is_on_tpu=FLAGS.use_tpu,
          gan_train_steps=tfgan_tuples.GANTrainSteps(1, 1))

    self.assertIsInstance(spec, tpu_estimator.TPUEstimatorSpec)
    self.assertEqual(mode, spec.mode)
    if mode == model_fn_lib.ModeKeys.PREDICT:
      self.assertEqual({'generated_data': self._gan_model.generated_data},
                       spec.predictions)
    elif mode == model_fn_lib.ModeKeys.TRAIN:
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.train_op)
      self.assertIsNotNone(spec.training_hooks)
    elif mode == model_fn_lib.ModeKeys.EVAL:
      self.assertEqual(self._gan_model.generated_data, spec.predictions)
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.eval_metrics)


class TPUGANEstimatorIntegrationTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TPUGANEstimatorIntegrationTest, self).setUp()
    self._model_dir = tempfile.mkdtemp()
    self._config = tpu_config.RunConfig(model_dir=self._model_dir)

  def tearDown(self):
    super(TPUGANEstimatorIntegrationTest, self).tearDown()
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, prediction_size,
      lr_decay=False, joint_train=True):
    def make_opt():
      gstep = training_util.get_or_create_global_step()
      lr = learning_rate_decay.exponential_decay(1.0, gstep, 10, 0.9)
      return training.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    dopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    est = estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=losses.wasserstein_generator_loss,
        discriminator_loss_fn=losses.wasserstein_discriminator_loss,
        generator_optimizer=gopt,
        discriminator_optimizer=dopt,
        joint_train=joint_train,
        get_eval_metric_ops_fn=get_metrics,
        train_batch_size=4,
        eval_batch_size=10,
        predict_batch_size=8,
        use_tpu=FLAGS.use_tpu,
        config=self._config)

    # Train.
    num_steps_train = 10
    est.train(train_input_fn, steps=num_steps_train)

    # Evaluate.
    num_steps_eval = 2
    scores = est.evaluate(eval_input_fn, steps=num_steps_eval)
    self.assertIn(ops.GraphKeys.GLOBAL_STEP, scores)
    self.assertIn('loss', scores)
    self.assertEqual(scores['discriminator_loss'] + scores['generator_loss'],
                     scores['loss'])
    self.assertIn('mse_custom_metric', scores)

    # Predict.
    predictions = np.array([x['generated_data'] for x in
                            est.predict(predict_input_fn)])
    self.assertAllEqual(prediction_size, predictions.shape)

  @parameterized.named_parameters(
      ('joint_train', True, False, False),
      ('train_sequential', False, False, False),
      ('lr_decay', False, True, False),
      ('train_sequential_ds', False, False, True))
  def test_numpy_input_fn(self, joint_train, lr_decay, return_ds):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4
    def train_input_fn(params):
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (dataset_ops.Dataset
            .from_tensors((data, data))
            .repeat()
            .batch(params['batch_size'], drop_remainder=True))
      if return_ds:
        return ds
      else:
        x, y = ds.make_one_shot_iterator().get_next()
        return x, y
    def eval_input_fn(params):
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (dataset_ops.Dataset
            .from_tensors((data, data))
            .repeat()
            .batch(params['batch_size'], drop_remainder=True))
      if return_ds:
        return ds
      else:
        x, y = ds.make_one_shot_iterator().get_next()
        return x, y
    predict_size = 10
    def predict_input_fn(params):
      del params  # unused
      data = np.zeros([input_dim], dtype=np.float32)
      ds = (dataset_ops.Dataset
            .from_tensors(data)
            .repeat(predict_size)
            .batch(1, drop_remainder=True))
      return ds

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[predict_size, input_dim],
        lr_decay=lr_decay,
        joint_train=joint_train)


class TPUGANEstimatorWarmStartTest(test.TestCase):

  def setUp(self):
    self._model_dir = self.get_temp_dir()
    self._config = tpu_config.RunConfig(model_dir=self._model_dir)
    self.new_variable_name = 'new_var'
    self.new_variable_value = [1.0, 2.0, 3.0]

  def tearDown(self):
    writer_cache.FileWriterCache.clear()

  def _test_warm_start(self, warm_start_from=None):
    """Tests whether WarmStartSettings work as intended."""
    def generator_with_new_variable(noise_dict, mode):
      variable_scope.get_variable(name=self.new_variable_name,
                                  initializer=self.new_variable_value,
                                  trainable=True)
      return generator_fn(noise_dict, mode)

    est = estimator.TPUGANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=losses.wasserstein_generator_loss,
        discriminator_loss_fn=losses.wasserstein_discriminator_loss,
        generator_optimizer=training.GradientDescentOptimizer(1.0),
        discriminator_optimizer=training.GradientDescentOptimizer(1.0),
        train_batch_size=4,
        use_tpu=FLAGS.use_tpu,
        config=self._config)

    def train_input_fn(params):
      data = np.zeros([params['batch_size'], 4], dtype=np.float32)
      return data, data

    est.train(train_input_fn, steps=1)

    est_warm = estimator.TPUGANEstimator(
        generator_fn=generator_with_new_variable,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=losses.wasserstein_generator_loss,
        discriminator_loss_fn=losses.wasserstein_discriminator_loss,
        generator_optimizer=training.GradientDescentOptimizer(1.0),
        discriminator_optimizer=training.GradientDescentOptimizer(1.0),
        config=tpu_config.RunConfig(
            model_dir=None if warm_start_from else self._model_dir),
        train_batch_size=4,
        use_tpu=FLAGS.use_tpu,
        warm_start_from=warm_start_from)

    est_warm.train(train_input_fn, steps=1)

    return est_warm

  def test_warm_start_error(self):
    """Test if exception when reloading different estimators."""
    with self.assertRaises(NotFoundError):
      self._test_warm_start()

  def test_warm_start_success(self):
    """Test if GANEstimator allows explicit warm start variable assignment."""
    # Regex matches all variable names in ckpt except for new_var.
    var_regex = '^(?!.*%s.*)' % self.new_variable_name
    warmstart = WarmStartSettings(ckpt_to_initialize_from=self._model_dir,
                                  vars_to_warm_start=var_regex)
    est_warm = self._test_warm_start(warm_start_from=warmstart)
    full_variable_name = 'Generator/%s' % self.new_variable_name
    self.assertIn(full_variable_name, est_warm.get_variable_names())
    equal_vals = np.array_equal(est_warm.get_variable_value(full_variable_name),
                                self.new_variable_value)
    self.assertTrue(equal_vals)

if __name__ == '__main__':
  test.main()
