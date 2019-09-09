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
import os

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
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
  if not getattr(model, '_compile_metrics', None):
    return None

  # TODO(psv/kathywu): use this implementation in model to estimator flow.
  # We are not using model.metrics here because we want to exclude the metrics
  # added using `add_metric` API.
  return {m.name: m for m in model._compile_metric_functions}  # pylint: disable=protected-access


def model_input_signature(model):
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  try:
    inputs = model.inputs
    input_names = model.input_names
  except AttributeError:
    return None
  flat_inputs = nest.flatten(inputs)
  flat_input_names = nest.flatten(input_names)
  flat_input_specs = []
  for input_tensor, input_name in zip(flat_inputs, flat_input_names):
    # If the user has not explicitly provided the input_signature, we
    # create it from the inputs. We make sure to set the first dimension
    # (batch) to None here, as in serving or retraining, batch should not
    # be fixed. See b/132783590 for context.
    input_shape = [None] + input_tensor.shape[1:].as_list()
    flat_input_specs.append(tensor_spec.TensorSpec(
        shape=input_shape, dtype=input_tensor.dtype,
        name=input_name))
  input_specs = nest.pack_sequence_as(structure=inputs,
                                      flat_sequence=flat_input_specs)

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
      outputs_list = nest.flatten(model(inputs=inputs, training=False))

    try:
      output_names = model.output_names
    except AttributeError:
      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
      output_names = training_utils.generic_output_names(outputs_list)
    return {name: output for name, output in zip(output_names, outputs_list)}

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
    else:
      metadata['training_config'] = {
          'loss': model.loss,
          # pylint: disable=protected-access
          'metrics': model._compile_metrics,
          'weighted_metrics': model._compile_weighted_metrics,
          # pylint: enable=protected-access
          'sample_weight_mode': model.sample_weight_mode,
          'loss_weights': model.loss_weights,
      }
      if isinstance(model.optimizer, optimizer_v2.RestoredOptimizer):
        raise NotImplementedError(
            'As of now, Optimizers loaded from SavedModel cannot be saved. '
            'If you\'re calling `model.save` or `tf.keras.models.save_model`, '
            'please set the `include_optimizer` option to `False`. For '
            '`tf.saved_model.save`, delete the optimizer from the model.')
      else:
        optimizer_config = {
            'class_name': model.optimizer.__class__.__name__,
            'config': model.optimizer.get_config()}
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

  optimizer_config = training_config['optimizer_config']
  optimizer = optimizers.deserialize(
      optimizer_config, custom_objects=custom_objects)

  # Recover loss functions and metrics.
  loss_config = training_config['loss']  # Deserialize loss class.
  if isinstance(loss_config, dict) and 'class_name' in loss_config:
    loss_config = losses.get(loss_config)
  loss = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), loss_config)
  metrics = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), training_config['metrics'])
  weighted_metrics = nest.map_structure(
      lambda obj: custom_objects.get(obj, obj),
      training_config.get('weighted_metrics', None))
  sample_weight_mode = training_config['sample_weight_mode']
  loss_weights = training_config['loss_weights']

  return dict(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)
