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
"""Tests for TFGAN's estimator.py."""

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
from tensorflow.contrib.gan.python.estimator.python import gan_estimator_impl as estimator
from tensorflow.contrib.gan.python.losses.python import tuple_losses as losses
from tensorflow.contrib.learn.python.learn.learn_io import graph_io
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import training
from tensorflow.python.training import training_util


def generator_fn(noise_dict, mode):
  del mode
  noise = noise_dict['x']
  return layers.fully_connected(noise, noise.shape[1].value)


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return layers.fully_connected(data, 1)


class GetGANModelTest(test.TestCase, parameterized.TestCase):
  """Tests that `GetGANModel` produces the correct model."""

  @parameterized.named_parameters(
      ('train', model_fn_lib.ModeKeys.TRAIN),
      ('eval', model_fn_lib.ModeKeys.EVAL),
      ('predict', model_fn_lib.ModeKeys.PREDICT))
  def test_get_gan_model(self, mode):
    with ops.Graph().as_default():
      generator_inputs = {'x': array_ops.ones([3, 4])}
      real_data = (array_ops.zeros([3, 4]) if
                   mode != model_fn_lib.ModeKeys.PREDICT else None)
      gan_model = estimator._get_gan_model(
          mode, generator_fn, discriminator_fn, real_data, generator_inputs,
          add_summaries=False)

    self.assertEqual(generator_inputs, gan_model.generator_inputs)
    self.assertIsNotNone(gan_model.generated_data)
    self.assertEqual(2, len(gan_model.generator_variables))  # 1 FC layer
    self.assertIsNotNone(gan_model.generator_fn)
    if mode == model_fn_lib.ModeKeys.PREDICT:
      self.assertIsNone(gan_model.real_data)
      self.assertIsNone(gan_model.discriminator_real_outputs)
      self.assertIsNone(gan_model.discriminator_gen_outputs)
      self.assertIsNone(gan_model.discriminator_variables)
      self.assertIsNone(gan_model.discriminator_scope)
      self.assertIsNone(gan_model.discriminator_fn)
    else:
      self.assertIsNotNone(gan_model.real_data)
      self.assertIsNotNone(gan_model.discriminator_real_outputs)
      self.assertIsNotNone(gan_model.discriminator_gen_outputs)
      self.assertEqual(2, len(gan_model.discriminator_variables))  # 1 FC layer
      self.assertIsNotNone(gan_model.discriminator_scope)
      self.assertIsNotNone(gan_model.discriminator_fn)


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


def dummy_loss_fn(gan_model):
  return math_ops.reduce_sum(gan_model.discriminator_real_outputs -
                             gan_model.discriminator_gen_outputs)


def get_metrics(gan_model):
  return {
      'mse_custom_metric': metrics_lib.mean_squared_error(
          gan_model.real_data, gan_model.generated_data)
  }


class GetEstimatorSpecTest(test.TestCase, parameterized.TestCase):
  """Tests that the EstimatorSpec is constructed appropriately."""

  @classmethod
  def setUpClass(cls):
    cls._generator_optimizer = training.GradientDescentOptimizer(1.0)
    cls._discriminator_optimizer = training.GradientDescentOptimizer(1.0)

  @parameterized.named_parameters(
      ('train', model_fn_lib.ModeKeys.TRAIN),
      ('eval', model_fn_lib.ModeKeys.EVAL),
      ('predict', model_fn_lib.ModeKeys.PREDICT))
  def test_get_estimator_spec(self, mode):
    with ops.Graph().as_default():
      self._gan_model = get_dummy_gan_model()
      spec = estimator._get_estimator_spec(
          mode,
          self._gan_model,
          generator_loss_fn=dummy_loss_fn,
          discriminator_loss_fn=dummy_loss_fn,
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
class GANEstimatorIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_complete_flow(
      self, train_input_fn, eval_input_fn, predict_input_fn, prediction_size,
      lr_decay=False):
    def make_opt():
      gstep = training_util.get_or_create_global_step()
      lr = learning_rate_decay.exponential_decay(1.0, gstep, 10, 0.9)
      return training.GradientDescentOptimizer(lr)

    gopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    dopt = make_opt if lr_decay else training.GradientDescentOptimizer(1.0)
    est = estimator.GANEstimator(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=losses.wasserstein_generator_loss,
        discriminator_loss_fn=losses.wasserstein_discriminator_loss,
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

  def test_numpy_input_fn(self):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4
    batch_size = 5
    data = np.zeros([batch_size, input_dim])
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[batch_size, input_dim])

  def test_numpy_input_fn_lrdecay(self):
    """Tests complete flow with numpy_input_fn."""
    input_dim = 4
    batch_size = 5
    data = np.zeros([batch_size, input_dim])
    train_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        y=data,
        batch_size=batch_size,
        shuffle=False)
    predict_input_fn = numpy_io.numpy_input_fn(
        x={'x': data},
        batch_size=batch_size,
        shuffle=False)

    self._test_complete_flow(
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        prediction_size=[batch_size, input_dim],
        lr_decay=True)

  def test_input_fn_from_parse_example(self):
    """Tests complete flow with input_fn constructed from parse_example."""
    input_dim = 4
    batch_size = 6
    data = np.zeros([batch_size, input_dim])

    serialized_examples = []
    for datum in data:
      example = example_pb2.Example(features=feature_pb2.Features(
          feature={
              'x': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
              'y': feature_pb2.Feature(
                  float_list=feature_pb2.FloatList(value=datum)),
          }))
      serialized_examples.append(example.SerializeToString())

    feature_spec = {
        'x': parsing_ops.FixedLenFeature([input_dim], dtypes.float32),
        'y': parsing_ops.FixedLenFeature([input_dim], dtypes.float32),
    }
    def _train_input_fn():
      feature_map = parsing_ops.parse_example(
          serialized_examples, feature_spec)
      _, features = graph_io.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _eval_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      _, features = graph_io.queue_parsed_features(feature_map)
      labels = features.pop('y')
      return features, labels
    def _predict_input_fn():
      feature_map = parsing_ops.parse_example(
          input_lib.limit_epochs(serialized_examples, num_epochs=1),
          feature_spec)
      _, features = graph_io.queue_parsed_features(feature_map)
      features.pop('y')
      return features, None

    self._test_complete_flow(
        train_input_fn=_train_input_fn,
        eval_input_fn=_eval_input_fn,
        predict_input_fn=_predict_input_fn,
        prediction_size=[batch_size, input_dim])


if __name__ == '__main__':
  test.main()
