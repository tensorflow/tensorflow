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
"""Utils related to keras model saving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import six

from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def extract_model_metrics(model):
  """Convert metrics from a Keras model `compile` API to dictionary.

  This is used for converting Keras models to Estimators and SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to metric instances. May return `None` if
    the model does not contain any metrics.
  """
  if getattr(model, '_compile_metrics', None):
    # TODO(psv/kathywu): use this implementation in model to estimator flow.
    # We are not using model.metrics here because we want to exclude the metrics
    # added using `add_metric` API.
    return {m.name: m for m in model._compile_metric_functions}  # pylint: disable=protected-access
  return None


def model_input_signature(model, keep_original_batch_size=False):
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  input_specs = model._get_save_spec(dynamic_batch=not keep_original_batch_size)  # pylint: disable=protected-access
  if input_specs is None:
    return None
  input_specs = _enforce_names_consistency(input_specs)
  # Return a list with a single element as the model's input signature.
  if isinstance(input_specs, collections.Sequence) and len(input_specs) == 1:
    # Note that the isinstance check filters out single-element dictionaries,
    # which should also be wrapped as a single-element list.
    return input_specs
  else:
    return [input_specs]


def raise_model_input_error(model):
  raise ValueError(
      'Model {} cannot be saved because the input shapes have not been '
      'set. Usually, input shapes are automatically determined from calling'
      ' .fit() or .predict(). To manually set the shapes, call '
      'model._set_inputs(inputs).'.format(model))


def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature

  if input_signature is None:
    input_signature = model_input_signature(model)

  if input_signature is None:
    raise_model_input_error(model)

  # TODO(mdan): Should the model's call be autographed by default?
  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)

    with base_layer_utils.call_context().enter(
        model, inputs=inputs, build_graph=False, training=False, saving=True):
      outputs = model(inputs, training=False)

    # Outputs always has to be a flat dict.
    output_names = model.output_names  # Functional Model.
    if output_names is None:  # Subclassed Model.
      from tensorflow.python.keras.engine import compile_utils  # pylint: disable=g-import-not-at-top
      output_names = compile_utils.create_pseudo_output_names(outputs)
    outputs = nest.flatten(outputs)
    return {name: output for name, output in zip(output_names, outputs)}

  return _wrapped_model


def model_metadata(model, include_optimizer=True, require_config=True):
  """Returns a dictionary containing the model metadata."""
  from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.optimizer_v2 import optimizer_v2  # pylint: disable=g-import-not-at-top

  model_config = {'class_name': model.__class__.__name__}
  try:
    model_config['config'] = model.get_config()
  except NotImplementedError as e:
    if require_config:
      raise e

  metadata = dict(
      keras_version=str(keras_version),
      backend=K.backend(),
      model_config=model_config)
  if model.optimizer and include_optimizer:
    if isinstance(model.optimizer, optimizers.TFOptimizer):
      logging.warning(
          'TensorFlow optimizers do not '
          'make it possible to access '
          'optimizer attributes or optimizer state '
          'after instantiation. '
          'As a result, we cannot save the optimizer '
          'as part of the model save file. '
          'You will have to compile your model again after loading it. '
          'Prefer using a Keras optimizer instead '
          '(see keras.io/optimizers).')
    elif model._compile_was_called:  # pylint: disable=protected-access
      training_config = model._get_compile_args()  # pylint: disable=protected-access
      training_config.pop('optimizer', None)  # Handled separately.
      metadata['training_config'] = _serialize_nested_config(training_config)
      if isinstance(model.optimizer, optimizer_v2.RestoredOptimizer):
        raise NotImplementedError(
            'As of now, Optimizers loaded from SavedModel cannot be saved. '
            'If you\'re calling `model.save` or `tf.keras.models.save_model`,'
            ' please set the `include_optimizer` option to `False`. For '
            '`tf.saved_model.save`, delete the optimizer from the model.')
      else:
        optimizer_config = {
            'class_name':
                generic_utils.get_registered_name(model.optimizer.__class__),
            'config':
                model.optimizer.get_config()
        }
      metadata['training_config']['optimizer_config'] = optimizer_config
  return metadata


def should_overwrite(filepath, overwrite):
  """Returns whether the filepath should be overwritten."""
  # If file exists and should not be overwritten.
  if not overwrite and os.path.isfile(filepath):
    return ask_to_proceed_with_overwrite(filepath)
  return True


def compile_args_from_training_config(training_config, custom_objects=None):
  """Return model.compile arguments from training config."""
  if custom_objects is None:
    custom_objects = {}

  with generic_utils.CustomObjectScope(custom_objects):
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config)

    # Recover losses.
    loss = None
    loss_config = training_config.get('loss', None)
    if loss_config is not None:
      loss = _deserialize_nested_config(losses.deserialize, loss_config)

    # Recover metrics.
    metrics = None
    metrics_config = training_config.get('metrics', None)
    if metrics_config is not None:
      metrics = _deserialize_nested_config(_deserialize_metric, metrics_config)

    # Recover weighted metrics.
    weighted_metrics = None
    weighted_metrics_config = training_config.get('weighted_metrics', None)
    if weighted_metrics_config is not None:
      weighted_metrics = _deserialize_nested_config(_deserialize_metric,
                                                    weighted_metrics_config)

    sample_weight_mode = training_config['sample_weight_mode']
    loss_weights = training_config['loss_weights']

  return dict(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)


def _deserialize_nested_config(deserialize_fn, config):
  """Deserializes arbitrary Keras `config` using `deserialize_fn`."""

  def _is_single_object(obj):
    if isinstance(obj, dict) and 'class_name' in obj:
      return True  # Serialized Keras object.
    if isinstance(obj, six.string_types):
      return True  # Serialized function or string.
    return False

  if config is None:
    return None
  if _is_single_object(config):
    return deserialize_fn(config)
  elif isinstance(config, dict):
    return {
        k: _deserialize_nested_config(deserialize_fn, v)
        for k, v in config.items()
    }
  elif isinstance(config, (tuple, list)):
    return [_deserialize_nested_config(deserialize_fn, obj) for obj in config]

  raise ValueError('Saved configuration not understood.')


def _serialize_nested_config(config):
  """Serialized a nested structure of Keras objects."""

  def _serialize_fn(obj):
    if callable(obj):
      return generic_utils.serialize_keras_object(obj)
    return obj

  return nest.map_structure(_serialize_fn, config)


def _deserialize_metric(metric_config):
  """Deserialize metrics, leaving special strings untouched."""
  from tensorflow.python.keras import metrics as metrics_module  # pylint:disable=g-import-not-at-top
  if metric_config in ['accuracy', 'acc', 'crossentropy', 'ce']:
    # Do not deserialize accuracy and cross-entropy strings as we have special
    # case handling for these in compile, based on model output shape.
    return metric_config
  return metrics_module.deserialize(metric_config)


def _enforce_names_consistency(specs):
  """Enforces that either all specs have names or none do."""

  def _has_name(spec):
    return hasattr(spec, 'name') and spec.name is not None

  def _clear_name(spec):
    spec = copy.deepcopy(spec)
    if hasattr(spec, 'name'):
      spec._name = None  # pylint:disable=protected-access
    return spec

  flat_specs = nest.flatten(specs)
  name_inconsistency = (
      any(_has_name(s) for s in flat_specs) and
      not all(_has_name(s) for s in flat_specs))

  if name_inconsistency:
    specs = nest.map_structure(_clear_name, specs)
  return specs
