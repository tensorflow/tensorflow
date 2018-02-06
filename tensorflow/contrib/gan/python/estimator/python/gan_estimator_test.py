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

import numpy as np
import six

from tensorflow.contrib import layers
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.gan.python.estimator.python import gan_estimator_impl as estimator
from tensorflow.contrib.gan.python.losses.python import tuple_losses as losses
from tensorflow.contrib.learn.python.learn.learn_io import graph_io
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training
from tensorflow.python.training import training_util


def generator_fn(noise_dict, mode):
  del mode
  noise = noise_dict['x']
  return layers.fully_connected(noise, noise.shape[1].value)


def discriminator_fn(data, unused_conditioning, mode):
  del unused_conditioning, mode
  return layers.fully_connected(data, 1)


def mock_head(testcase, expected_generator_inputs, expected_real_data,
              generator_scope_name):
  """Returns a mock head that validates logits values and variable names."""
  discriminator_scope_name = 'Discriminator'  # comes from TFGAN defaults
  generator_var_names = set([
      '%s/fully_connected/weights:0' % generator_scope_name,
      '%s/fully_connected/biases:0' % generator_scope_name])
  discriminator_var_names = set([
      '%s/fully_connected/weights:0' % discriminator_scope_name,
      '%s/fully_connected/biases:0' % discriminator_scope_name])

  def _create_estimator_spec(features, mode, logits, labels):
    gan_model = logits  # renaming for clarity
    is_predict = mode == model_fn_lib.ModeKeys.PREDICT
    testcase.assertIsNone(features)
    testcase.assertIsNone(labels)
    testcase.assertIsInstance(gan_model, namedtuples.GANModel)

    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    expected_var_names = (generator_var_names if is_predict else
                          generator_var_names | discriminator_var_names)
    testcase.assertItemsEqual(expected_var_names,
                              [var.name for var in trainable_vars])

    assertions = []
    def _or_none(x):
      return None if is_predict else x
    testcase.assertEqual(expected_generator_inputs, gan_model.generator_inputs)
    # TODO(joelshor): Add check on `generated_data`.
    testcase.assertItemsEqual(
        generator_var_names,
        set([x.name for x in gan_model.generator_variables]))
    testcase.assertEqual(generator_scope_name, gan_model.generator_scope.name)
    testcase.assertEqual(_or_none(expected_real_data), gan_model.real_data)
    # TODO(joelshor): Add check on `discriminator_real_outputs`.
    # TODO(joelshor): Add check on `discriminator_gen_outputs`.
    if is_predict:
      testcase.assertIsNone(gan_model.discriminator_scope)
    else:
      testcase.assertEqual(discriminator_scope_name,
                           gan_model.discriminator_scope.name)

    with ops.control_dependencies(assertions):
      if mode == model_fn_lib.ModeKeys.TRAIN:
        return model_fn_lib.EstimatorSpec(
            mode=mode, loss=array_ops.zeros([]),
            train_op=control_flow_ops.no_op(), training_hooks=[])
      elif mode == model_fn_lib.ModeKeys.EVAL:
        return model_fn_lib.EstimatorSpec(
            mode=mode, predictions=gan_model.generated_data,
            loss=array_ops.zeros([]))
      elif mode == model_fn_lib.ModeKeys.PREDICT:
        return model_fn_lib.EstimatorSpec(
            mode=mode, predictions=gan_model.generated_data)
      else:
        testcase.fail('Invalid mode: {}'.format(mode))

  head = test.mock.NonCallableMagicMock(spec=head_lib._Head)
  head.create_estimator_spec = test.mock.MagicMock(
      wraps=_create_estimator_spec)

  return head


class GANModelFnTest(test.TestCase):
  """Tests that _gan_model_fn passes expected logits to mock head."""

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      writer_cache.FileWriterCache.clear()
      shutil.rmtree(self._model_dir)

  def _test_logits_helper(self, mode):
    """Tests that the expected logits are passed to mock head."""
    with ops.Graph().as_default():
      training_util.get_or_create_global_step()
      generator_inputs = {'x': array_ops.zeros([5, 4])}
      real_data = (None if mode == model_fn_lib.ModeKeys.PREDICT else
                   array_ops.zeros([5, 4]))
      generator_scope_name = 'generator'
      head = mock_head(self,
                       expected_generator_inputs=generator_inputs,
                       expected_real_data=real_data,
                       generator_scope_name=generator_scope_name)
      estimator_spec = estimator._gan_model_fn(
          features=generator_inputs,
          labels=real_data,
          mode=mode,
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          generator_scope_name=generator_scope_name,
          head=head)
      with monitored_session.MonitoredTrainingSession(
          checkpoint_dir=self._model_dir) as sess:
        if mode == model_fn_lib.ModeKeys.TRAIN:
          sess.run(estimator_spec.train_op)
        elif mode == model_fn_lib.ModeKeys.EVAL:
          sess.run(estimator_spec.loss)
        elif mode == model_fn_lib.ModeKeys.PREDICT:
          sess.run(estimator_spec.predictions)
        else:
          self.fail('Invalid mode: {}'.format(mode))

  def test_logits_predict(self):
    self._test_logits_helper(model_fn_lib.ModeKeys.PREDICT)

  def test_logits_eval(self):
    self._test_logits_helper(model_fn_lib.ModeKeys.EVAL)

  def test_logits_train(self):
    self._test_logits_helper(model_fn_lib.ModeKeys.TRAIN)


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
        model_dir=self._model_dir)

    # TRAIN
    num_steps = 10
    est.train(train_input_fn, steps=num_steps)

    # EVALUTE
    scores = est.evaluate(eval_input_fn)
    self.assertEqual(num_steps, scores[ops.GraphKeys.GLOBAL_STEP])
    self.assertIn('loss', six.iterkeys(scores))

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
