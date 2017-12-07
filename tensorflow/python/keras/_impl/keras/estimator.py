# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
"""Home of estimator related functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.client import session
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import export as export_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import models
from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import metrics as metrics_module
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def _create_ordered_io(keras_model, estimator_io_dict, is_input=True):
  """Create a list of tensors from IO dictionary based on Keras IO order.

  Args:
    keras_model: an instance of compiled keras model.
    estimator_io_dict: features or labels dictionary from model_fn.
    is_input: True if dictionary is for inputs.

  Returns:
    a list of tensors based on Keras IO order.

  Raises:
    ValueError: if dictionary keys cannot be found in Keras model input_names
      or output_names.
  """
  if is_input:
    keras_io_names = keras_model.input_names
  else:
    keras_io_names = keras_model.output_names

  for key in estimator_io_dict:
    if key not in keras_io_names:
      raise ValueError(
          'Cannot find %s with name "%s" in Keras Model. It needs to match '
          'one of the following: %s' % ('input' if is_input else 'output', key,
                                        ', '.join(keras_io_names)))
  tensors = []
  for io_name in keras_io_names:
    tensors.append(estimator_io_dict[io_name])
  return tensors


def _clone_and_build_model(mode,
                           keras_model,
                           custom_objects,
                           features=None,
                           labels=None):
  """Clone and build the given keras_model.

  Args:
    mode: training mode.
    keras_model: an instance of compiled keras model.
    custom_objects: Dictionary for custom objects.
    features:
    labels:

  Returns:
    The newly built model.
  """
  # Set to True during training, False for inference.
  K.set_learning_phase(mode == model_fn_lib.ModeKeys.TRAIN)

  # Clone keras model.
  input_tensors = None if features is None else _create_ordered_io(
      keras_model, features)
  if custom_objects:
    with CustomObjectScope(custom_objects):
      model = models.clone_model(keras_model, input_tensors=input_tensors)
  else:
    model = models.clone_model(keras_model, input_tensors=input_tensors)

  # Compile/Build model
  if mode is model_fn_lib.ModeKeys.PREDICT and not model.built:
    model.build()
  else:
    optimizer_config = keras_model.optimizer.get_config()
    optimizer = keras_model.optimizer.__class__.from_config(optimizer_config)
    optimizer.iterations = training_util.get_or_create_global_step()

    # Get list of outputs.
    if labels is None:
      target_tensors = None
    elif isinstance(labels, dict):
      target_tensors = _create_ordered_io(keras_model, labels, is_input=False)
    else:
      target_tensors = [
          sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(labels)
      ]

    model.compile(
        optimizer,
        keras_model.loss,
        metrics=keras_model.metrics,
        loss_weights=keras_model.loss_weights,
        sample_weight_mode=keras_model.sample_weight_mode,
        weighted_metrics=keras_model.weighted_metrics,
        target_tensors=target_tensors)

  if isinstance(model, models.Sequential):
    model = model.model
  return model


def _create_keras_model_fn(keras_model, custom_objects=None):
  """Creates model_fn for keras Estimator.

  Args:
    keras_model: an instance of compiled keras model.
    custom_objects: Dictionary for custom objects.

  Returns:
    The model_fn for a keras Estimator.
  """

  def model_fn(features, labels, mode):
    """model_fn for keras Estimator."""
    model = _clone_and_build_model(mode, keras_model, custom_objects, features,
                                   labels)
    # Get inputs to EstimatorSpec
    predictions = dict(zip(model.output_names, model.outputs))

    loss = None
    train_op = None
    eval_metric_ops = None

    # Set loss and metric only during train and evaluate.
    if mode is not model_fn_lib.ModeKeys.PREDICT:
      model._make_train_function()  # pylint: disable=protected-access
      loss = model.total_loss

      if model.metrics:
        eval_metric_ops = {}
        # When each metric maps to an output
        if isinstance(model.metrics, dict):
          for i, output_name in enumerate(model.metrics.keys()):
            metric_name = model.metrics[output_name]
            if callable(metric_name):
              metric_name = metric_name.__name__
            # When some outputs use the same metric
            if list(model.metrics.values()).count(metric_name) > 1:
              metric_name += '_' + output_name
            eval_metric_ops[metric_name] = metrics_module.mean(
                model.metrics_tensors[i - len(model.metrics)])
        else:
          for i, metric_name in enumerate(model.metrics):
            if callable(metric_name):
              metric_name = metric_name.__name__
            eval_metric_ops[metric_name] = metrics_module.mean(
                model.metrics_tensors[i])

    # Set train_op only during train.
    if mode is model_fn_lib.ModeKeys.TRAIN:
      train_op = model.train_function.updates_op

    return model_fn_lib.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={
            _DEFAULT_SERVING_KEY:
            export_lib.export_output.PredictOutput(predictions)
        })

  return model_fn


def _save_first_checkpoint(keras_model, estimator, custom_objects,
                           keras_weights):
  """Save first checkpoint for the keras Estimator.

  Args:
    keras_model: an instance of compiled keras model.
    estimator: keras estimator.
    custom_objects: Dictionary for custom objects.
    keras_weights: A flat list of Numpy arrays for weights of given keras_model.

  Returns:
    The model_fn for a keras Estimator.
  """
  with ops.Graph().as_default() as g, g.device(estimator._device_fn):
    random_seed.set_random_seed(estimator.config.tf_random_seed)
    training_util.create_global_step()
    model = _clone_and_build_model(model_fn_lib.ModeKeys.TRAIN, keras_model,
                                   custom_objects)

    if isinstance(model, models.Sequential):
      model = model.model
    # Load weights and save to checkpoint if there is no checkpoint
    latest_path = saver_lib.latest_checkpoint(estimator.model_dir)
    if not latest_path:
      with session.Session() as sess:
        model.set_weights(keras_weights)
        # Make update ops and initialize all variables.
        if not model.train_function:
          # pylint: disable=protected-access
          model._make_train_function()
          K._initialize_variables(sess)
          # pylint: enable=protected-access
        saver = saver_lib.Saver()
        saver.save(sess, os.path.join(estimator.model_dir, 'keras_model.ckpt'))


def model_to_estimator(keras_model=None,
                       keras_model_path=None,
                       custom_objects=None,
                       model_dir=None,
                       config=None):
  """Constructs an `Estimator` instance from given keras model.

  For usage example, please see
  @{$programmers_guide/estimators$creating_estimators_from_keras_models}.

  Args:
    keras_model: Keras model in memory.
    keras_model_path: Directory to a keras model on disk.
    custom_objects: Dictionary for custom objects.
    model_dir: Directory to save Estimator model parameters, graph and etc.
    config: Configuration object.

  Returns:
    An Estimator from given keras model.

  Raises:
    ValueError: if neither keras_model nor keras_model_path was given.
    ValueError: if both keras_model and keras_model_path was given.
    ValueError: if the keras_model_path is a GCS URI.
    ValueError: if keras_model has not been compiled.
  """
  if (not keras_model) and (not keras_model_path):
    raise ValueError(
        'Either keras_model or keras_model_path needs to be provided.')
  if keras_model and keras_model_path:
    raise ValueError(
        'Please specity either keras_model or keras_model_path but not both.')

  if not keras_model:
    if keras_model_path.startswith(
        'gs://') or 'storage.googleapis.com' in keras_model_path:
      raise ValueError(
          '%s is not a local path. Please copy the model locally first.' %
          keras_model_path)
    logging.info('Loading models from %s', keras_model_path)
    keras_model = models.load_model(keras_model_path)
  else:
    logging.info('Using the Keras model from memory.')
    keras_model = keras_model

  if not hasattr(keras_model, 'optimizer'):
    raise ValueError(
        'Given keras model has not been compiled yet. Please compile first '
        'before creating the estimator.')

  keras_weights = keras_model.get_weights()
  keras_model_fn = _create_keras_model_fn(keras_model, custom_objects)
  est = estimator_lib.Estimator(
      keras_model_fn, model_dir=model_dir, config=config)
  # TODO(yifeif): move checkpoint initialization to scaffold.init_fn
  _save_first_checkpoint(keras_model, est, custom_objects, keras_weights)
  return est
