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
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import models
from tensorflow.python.keras._impl.keras import optimizers
from tensorflow.python.keras._impl.keras.engine.base_layer import Layer
from tensorflow.python.keras._impl.keras.engine.network import Network
from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_module
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def _cast_tensor_to_floatx(x):
  """Cast tensor to keras's floatx dtype if it is not already the same dtype."""
  if x.dtype == K.floatx():
    return x
  else:
    return math_ops.cast(x, K.floatx())


def _create_ordered_io(keras_model, estimator_io, is_input=True):
  """Create a list of tensors from IO dictionary based on Keras IO order.

  Args:
    keras_model: An instance of compiled keras model.
    estimator_io: The features or labels (dict or plain array) from model_fn.
    is_input: True if dictionary is for inputs.

  Returns:
    A list of tensors based on Keras IO order.

  Raises:
    ValueError: if dictionary keys cannot be found in Keras model input_names
      or output_names.
  """
  if isinstance(estimator_io, (list, tuple)):
    # Case currently not supported by most built-in input_fn,
    # but it's good to have for sanity
    return [_cast_tensor_to_floatx(x) for x in estimator_io]
  elif isinstance(estimator_io, dict):
    if is_input:
      if keras_model._is_graph_network:
        keras_io_names = keras_model.input_names
      else:
        keras_io_names = [
            'input_%d' % i for i in range(1, len(estimator_io) + 1)]
    else:
      if keras_model._is_graph_network:
        keras_io_names = keras_model.output_names
      else:
        keras_io_names = [
            'output_%d' % i for i in range(1, len(estimator_io) + 1)]

    for key in estimator_io:
      if key not in keras_io_names:
        raise ValueError(
            'Cannot find %s with name "%s" in Keras Model. '
            'It needs to match one '
            'of the following: %s' % ('input' if is_input else 'output', key,
                                      ', '.join(keras_io_names)))
      tensors = [_cast_tensor_to_floatx(estimator_io[io_name])
                 for io_name in keras_io_names]
    return tensors
  else:
    # Plain array.
    return _cast_tensor_to_floatx(estimator_io)


def _in_place_subclassed_model_reset(model):
  """Substitute for model cloning that works for subclassed models.

  Subclassed models cannot be cloned because their topology is not serializable.
  To "instantiate" an identical model in a new TF graph, we reuse the original
  model object, but we clear its state.

  After calling this function on a model intance, you can use the model instance
  as if it were a model clone (in particular you can use it in a new graph).

  This method clears the state of the input model. It is thus destructive.
  However the original state can be restored fully by calling
  `_in_place_subclassed_model_state_restoration`.

  Args:
    model: Instance of a Keras model created via subclassing.

  Raises:
    ValueError: In case the model uses a subclassed model as inner layer.
  """
  assert not model._is_graph_network  # Only makes sense for subclassed networks
  # Retrieve all layers tracked by the model as well as their attribute names
  attributes_cache = {}
  for name in dir(model):
    try:
      value = getattr(model, name)
    except (AttributeError, ValueError, TypeError):
      continue
    if isinstance(value, Layer):
      attributes_cache[name] = value
      assert value in model._layers
    elif isinstance(value, (list, tuple)) and name not in ('layers', '_layers'):
      # Handle case: list/tuple of layers (also tracked by the Network API).
      if value and all(isinstance(val, Layer) for val in value):
        raise ValueError('We do not support the use of list-of-layers '
                         'attributes in subclassed models used with '
                         '`model_to_estimator` at this time. Found list '
                         'model: %s' % name)

  # Replace layers on the model with fresh layers
  layers_to_names = {value: key for key, value in attributes_cache.items()}
  original_layers = model._layers[:]
  model._layers = []
  for layer in original_layers:  # We preserve layer order.
    config = layer.get_config()
    # This will not work for nested subclassed models used as layers.
    # This would be theoretically possible to support, but would add complexity.
    # Only do it if users complain.
    if isinstance(layer, Network) and not layer._is_graph_network:
      raise ValueError('We do not support the use of nested subclassed models '
                       'in `model_to_estimator` at this time. Found nested '
                       'model: %s' % layer)
    fresh_layer = layer.__class__.from_config(config)
    name = layers_to_names[layer]
    setattr(model, name, fresh_layer)

  # Cache original model build attributes (in addition to layers)
  if (not hasattr(model, '_original_attributes_cache') or
      model._original_attributes_cache is None):
    if model.built:
      attributes_to_cache = [
          'inputs',
          'outputs',
          '_feed_outputs',
          '_feed_output_names',
          '_feed_output_shapes',
          '_feed_loss_fns',
          'loss_weights_list',
          'targets',
          '_feed_targets',
          'sample_weight_modes',
          'weighted_metrics',
          'metrics_names',
          'metrics_tensors',
          'metrics_updates',
          'stateful_metric_names',
          'total_loss',
          'sample_weights',
          '_feed_sample_weights',
          'train_function',
          'test_function',
          'predict_function',
          '_collected_trainable_weights',
          '_feed_inputs',
          '_feed_input_names',
          '_feed_input_shapes',
          'optimizer',
      ]
      for name in attributes_to_cache:
        attributes_cache[name] = getattr(model, name)
  model._original_attributes_cache = attributes_cache

  # Reset built state
  model.built = False
  model.inputs = None
  model.outputs = None


def _in_place_subclassed_model_state_restoration(model):
  """Restores the original state of a model after it was "reset".

  This undoes this action of `_in_place_subclassed_model_reset`.

  Args:
    model: Instance of a Keras model created via subclassing, on which
      `_in_place_subclassed_model_reset` was previously called.
  """
  assert not model._is_graph_network
  # Restore layers and build attributes
  if (hasattr(model, '_original_attributes_cache') and
      model._original_attributes_cache is not None):
    model._layers = []
    for name, value in model._original_attributes_cache.items():
      setattr(model, name, value)
    model._original_attributes_cache = None
  else:
    # Restore to the state of a never-called model.
    model.built = False
    model.inputs = None
    model.outputs = None


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
    features: Dict of tensors.
    labels: Dict of tensors, or single tensor instance.

  Returns:
    The newly built model.
  """
  # Set to True during training, False for inference.
  K.set_learning_phase(mode == model_fn_lib.ModeKeys.TRAIN)

  # Get list of inputs.
  if features is None:
    input_tensors = None
  else:
    input_tensors = _create_ordered_io(keras_model,
                                       estimator_io=features,
                                       is_input=True)
  # Get list of outputs.
  if labels is None:
    target_tensors = None
  elif isinstance(labels, dict):
    target_tensors = _create_ordered_io(keras_model,
                                        estimator_io=labels,
                                        is_input=False)
  else:
    target_tensors = [
        _cast_tensor_to_floatx(
            sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(labels))
    ]

  if keras_model._is_graph_network:
    if custom_objects:
      with CustomObjectScope(custom_objects):
        model = models.clone_model(keras_model, input_tensors=input_tensors)
    else:
      model = models.clone_model(keras_model, input_tensors=input_tensors)
  else:
    model = keras_model
    _in_place_subclassed_model_reset(model)
    if input_tensors is not None:
      model._set_inputs(input_tensors)

  # Compile/Build model
  if mode is model_fn_lib.ModeKeys.PREDICT:
    if isinstance(model, models.Sequential):
      model.build()
  else:
    if isinstance(keras_model.optimizer, optimizers.TFOptimizer):
      optimizer = keras_model.optimizer
    else:
      optimizer_config = keras_model.optimizer.get_config()
      optimizer = keras_model.optimizer.__class__.from_config(optimizer_config)
    optimizer.iterations = training_util.get_or_create_global_step()

    model.compile(
        optimizer,
        keras_model.loss,
        metrics=keras_model.metrics,
        loss_weights=keras_model.loss_weights,
        sample_weight_mode=keras_model.sample_weight_mode,
        weighted_metrics=keras_model.weighted_metrics,
        target_tensors=target_tensors)
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
      if mode is model_fn_lib.ModeKeys.TRAIN:
        model._make_train_function()  # pylint: disable=protected-access
      else:
        model._make_test_function()  # pylint: disable=protected-access
      loss = model.total_loss

      if model.metrics:
        # TODO(fchollet): support stateful metrics
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

    if not model._is_graph_network:
      # Reset model state to original state,
      # to avoid `model_fn` being destructive for the initial model argument.
      _in_place_subclassed_model_state_restoration(keras_model)
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
  # Load weights and save to checkpoint if there is no checkpoint
  latest_path = saver_lib.latest_checkpoint(estimator.model_dir)
  if not latest_path:
    with ops.Graph().as_default():
      random_seed.set_random_seed(estimator.config.tf_random_seed)
      training_util.create_global_step()
      model = _clone_and_build_model(model_fn_lib.ModeKeys.TRAIN, keras_model,
                                     custom_objects)
      # save to checkpoint
      with session.Session(config=estimator._session_config) as sess:
        model.set_weights(keras_weights)
        # Make update ops and initialize all variables.
        if not model.train_function:
          # pylint: disable=protected-access
          model._make_train_function()
          K._initialize_variables(sess)
          # pylint: enable=protected-access
        saver = saver_lib.Saver()
        saver.save(sess, os.path.join(estimator.model_dir, 'keras_model.ckpt'))


@tf_export('keras.estimator.model_to_estimator')
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
        'Either `keras_model` or `keras_model_path` needs to be provided.')
  if keras_model and keras_model_path:
    raise ValueError(
        'Please specity either `keras_model` or `keras_model_path`, '
        'but not both.')

  if not keras_model:
    if keras_model_path.startswith(
        'gs://') or 'storage.googleapis.com' in keras_model_path:
      raise ValueError(
          '%s is not a local path. Please copy the model locally first.' %
          keras_model_path)
    logging.info('Loading models from %s', keras_model_path)
    keras_model = models.load_model(keras_model_path)
  else:
    logging.info('Using the Keras model provided.')
    keras_model = keras_model

  if not hasattr(keras_model, 'optimizer') or not keras_model.optimizer:
    raise ValueError(
        'The given keras model has not been compiled yet. Please compile first '
        'before calling `model_to_estimator`.')

  if isinstance(config, dict):
    config = run_config_lib.RunConfig(**config)

  keras_model_fn = _create_keras_model_fn(keras_model, custom_objects)
  estimator = estimator_lib.Estimator(
      keras_model_fn, model_dir=model_dir, config=config)

  # Pass the config into keras backend's default session.
  sess = session.Session(config=estimator._session_config)
  K.set_session(sess)

  keras_weights = keras_model.get_weights()
  if keras_model._is_graph_network:
    # TODO(yifeif): move checkpoint initialization to scaffold.init_fn
    _save_first_checkpoint(keras_model,
                           estimator,
                           custom_objects,
                           keras_weights)
  elif keras_model.built:
    logging.warning('You are creating an Estimator from a Keras model '
                    'manually subclassed from `Model`, that was '
                    'already called on some inputs (and thus already had '
                    'weights). We are currently unable to preserve '
                    'the model\'s state (its weights) '
                    'as part of the estimator '
                    'in this case. Be warned that the estimator '
                    'has been created using '
                    'a freshly initialized version of your model.\n'
                    'Note that this doesn\'t affect the state of the '
                    'model instance you passed as `keras_model` argument.')
  return estimator
