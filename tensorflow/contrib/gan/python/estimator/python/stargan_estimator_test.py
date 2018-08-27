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
"""Tests for TFGAN's stargan_estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.contrib import layers
from tensorflow.contrib.gan.python import namedtuples as tfgan_tuples
from tensorflow.contrib.gan.python.estimator.python import stargan_estimator_impl as estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training
from tensorflow.python.training import training_util


def dummy_generator_fn(input_data, input_data_domain_label, mode):
  del input_data_domain_label, mode

  return variable_scope.get_variable('dummy_g', initializer=0.5) * input_data


def dummy_discriminator_fn(input_data, num_domains, mode):
  del mode

  hidden = layers.flatten(input_data)
  output_src = math_ops.reduce_mean(hidden, axis=1)
  output_cls = layers.fully_connected(
      inputs=hidden, num_outputs=num_domains, scope='debug')

  return output_src, output_cls


class StarGetGANModelTest(test.TestCase, parameterized.TestCase):
  """Tests that `StarGetGANModel` produces the correct model."""

  @parameterized.named_parameters(('train', model_fn_lib.ModeKeys.TRAIN),
                                  ('eval', model_fn_lib.ModeKeys.EVAL),
                                  ('predict', model_fn_lib.ModeKeys.PREDICT))
  def test_get_gan_model(self, mode):
    with ops.Graph().as_default():
      input_data = array_ops.ones([6, 4, 4, 3])
      input_data_domain_label = array_ops.one_hot([0] * 6, 5)
      gan_model = estimator._get_gan_model(
          mode,
          dummy_generator_fn,
          dummy_discriminator_fn,
          input_data,
          input_data_domain_label,
          add_summaries=False)

    self.assertEqual(input_data, gan_model.input_data)
    self.assertIsNotNone(gan_model.generated_data)
    self.assertIsNotNone(gan_model.generated_data_domain_target)
    self.assertEqual(1, len(gan_model.generator_variables))
    self.assertIsNotNone(gan_model.generator_scope)
    self.assertIsNotNone(gan_model.generator_fn)
    if mode == model_fn_lib.ModeKeys.PREDICT:
      self.assertIsNone(gan_model.input_data_domain_label)
      self.assertEqual(input_data_domain_label,
                       gan_model.generated_data_domain_target)
      self.assertIsNone(gan_model.reconstructed_data)
      self.assertIsNone(gan_model.discriminator_input_data_source_predication)
      self.assertIsNone(
          gan_model.discriminator_generated_data_source_predication)
      self.assertIsNone(gan_model.discriminator_input_data_domain_predication)
      self.assertIsNone(
          gan_model.discriminator_generated_data_domain_predication)
      self.assertIsNone(gan_model.discriminator_variables)
      self.assertIsNone(gan_model.discriminator_scope)
      self.assertIsNone(gan_model.discriminator_fn)
    else:
      self.assertEqual(input_data_domain_label,
                       gan_model.input_data_domain_label)
      self.assertIsNotNone(gan_model.reconstructed_data.shape)
      self.assertIsNotNone(
          gan_model.discriminator_input_data_source_predication)
      self.assertIsNotNone(
          gan_model.discriminator_generated_data_source_predication)
      self.assertIsNotNone(
          gan_model.discriminator_input_data_domain_predication)
      self.assertIsNotNone(
          gan_model.discriminator_generated_data_domain_predication)
      self.assertEqual(2, len(gan_model.discriminator_variables))  # 1 FC layer
      self.assertIsNotNone(gan_model.discriminator_scope)
      self.assertIsNotNone(gan_model.discriminator_fn)


def get_dummy_gan_model():
  """Similar to get_gan_model()."""
  # TODO(joelshor): Find a better way of creating a variable scope.
  with variable_scope.variable_scope('generator') as gen_scope:
    gen_var = variable_scope.get_variable('dummy_var', initializer=0.0)
  with variable_scope.variable_scope('discriminator') as dis_scope:
    dis_var = variable_scope.get_variable('dummy_var', initializer=0.0)
  return tfgan_tuples.StarGANModel(
      input_data=array_ops.ones([1, 2, 2, 3]),
      input_data_domain_label=array_ops.ones([1, 2]),
      generated_data=array_ops.ones([1, 2, 2, 3]),
      generated_data_domain_target=array_ops.ones([1, 2]),
      reconstructed_data=array_ops.ones([1, 2, 2, 3]),
      discriminator_input_data_source_predication=array_ops.ones([1]) * dis_var,
      discriminator_generated_data_source_predication=array_ops.ones(
          [1]) * gen_var * dis_var,
      discriminator_input_data_domain_predication=array_ops.ones([1, 2
                                                                 ]) * dis_var,
      discriminator_generated_data_domain_predication=array_ops.ones([1, 2]) *
      gen_var * dis_var,
      generator_variables=[gen_var],
      generator_scope=gen_scope,
      generator_fn=None,
      discriminator_variables=[dis_var],
      discriminator_scope=dis_scope,
      discriminator_fn=None)


def dummy_loss_fn(gan_model):
  loss = math_ops.reduce_sum(
      gan_model.discriminator_input_data_domain_predication -
      gan_model.discriminator_generated_data_domain_predication)
  loss += math_ops.reduce_sum(gan_model.input_data - gan_model.generated_data)
  return tfgan_tuples.GANLoss(loss, loss)


def get_metrics(gan_model):
  return {
      'mse_custom_metric':
          metrics_lib.mean_squared_error(gan_model.input_data,
                                         gan_model.generated_data)
  }


class GetEstimatorSpecTest(test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    cls._generator_optimizer = training.GradientDescentOptimizer(1.0)
    cls._discriminator_optimizer = training.GradientDescentOptimizer(1.0)

  @parameterized.named_parameters(('train', model_fn_lib.ModeKeys.TRAIN),
                                  ('eval', model_fn_lib.ModeKeys.EVAL),
                                  ('predict', model_fn_lib.ModeKeys.PREDICT))
  def test_get_estimator_spec(self, mode):
    with ops.Graph().as_default():
      self._gan_model = get_dummy_gan_model()
      spec = estimator._get_estimator_spec(
          mode,
          self._gan_model,
          loss_fn=dummy_loss_fn,
          get_eval_metric_ops_fn=get_metrics,
          generator_optimizer=self._generator_optimizer,
          discriminator_optimizer=self._discriminator_optimizer)

    self.assertEqual(mode, spec.mode)
    if mode == model_fn_lib.ModeKeys.PREDICT:
      self.assertEqual(self._gan_model.generated_data, spec.predictions)
    elif mode == model_fn_lib.ModeKeys.TRAIN:
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.train_op)
      self.assertIsNotNone(spec.training_hooks)
    elif mode == model_fn_lib.ModeKeys.EVAL:
      self.assertEqual(self._gan_model.generated_data, spec.predictions)
      self.assertShapeEqual(np.array(0), spec.loss)  # must be a scalar
      self.assertIsNotNone(spec.eval_metric_ops)


# TODO(joelshor): Add pandas test.
class StarGANEstimatorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(self,
                          train_input_fn,
                          eval_input_fn,
                          predict_input_fn,
                          prediction_size,
                          lr_decay=False):

    def make_opt():
      gstep = training_util.get_or_create_global_step()
      lr = learning_rate_decay.exponential_decay(1.0, gstep, 10, 0.9)
      return training.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    dopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    est = estimator.StarGANEstimator(
        generator_fn=dummy_generator_fn,
        discriminator_fn=dummy_discriminator_fn,
        loss_fn=dummy_loss_fn,
        generator_optimizer=gopt,
        discriminator_optimizer=dopt,
        get_eval_metric_ops_fn=get_metrics,
        model_dir=self._model_dir)

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))
    self.assertEqual(scores['discriminator_loss'] + scores['generator_loss'],
                     scores['loss'])
    self.assertIn('mse_custom_metric', six.iterkeys(scores))

    # PREDICT
    predictions = np.array([x for x in est.predict(predict_input_fn)])

    self.assertAllEqual(prediction_size, predictions.shape)

  @staticmethod
  def _numpy_input_fn_wrapper(numpy_input_fn, batch_size, label_size):
    """Wrapper to remove the dictionary in numpy_input_fn.

    NOTE:
      We create the domain_label here because the model expect a fully define
      batch_size from the input.

    Args:
      numpy_input_fn: input_fn created from numpy_io
      batch_size: (int) number of items for each batch
      label_size: (int) number of domains

    Returns:
      a new input_fn
    """

    def new_input_fn():
      features = numpy_input_fn()
      return features['x'], array_ops.one_hot([0] * batch_size, label_size)

    return new_input_fn

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    batch_size = 5
    img_size = 8
    channel_size = 3
    label_size = 3
    image_data = np.zeros(
        [batch_size, img_size, img_size, channel_size], dtype=np.float32)
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': image_data},
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': image_data}, batch_size=batch_size, shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': image_data}, shuffle=False)

    train_input_fn = self._numpy_input_fn_wrapper(train_input_fn, batch_size,
                                                  label_size)
    eval_input_fn = self._numpy_input_fn_wrapper(eval_input_fn, batch_size,
                                                 label_size)
    predict_input_fn = self._numpy_input_fn_wrapper(predict_input_fn,
                                                    batch_size, label_size)

    predict_input_fn = estimator.stargan_prediction_input_fn_wrapper(
        predict_input_fn)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[batch_size, img_size, img_size, channel_size])


if __name__ == '__main__':
  test.main()
