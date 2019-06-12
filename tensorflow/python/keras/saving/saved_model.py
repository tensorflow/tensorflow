# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions to save/load keras Model to/from SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import six

from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_from_json
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import model_utils
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.training.tracking.tracking import delete_tracking
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import keras_export

# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
metrics_lib = LazyLoader("metrics_lib", globals(),
                         "tensorflow.python.keras.metrics")
models_lib = LazyLoader("models_lib", globals(),
                        "tensorflow.python.keras.models")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
network_lib = LazyLoader(
    "network_lib", globals(),
    "tensorflow.python.keras.engine.network")
sequential = LazyLoader(
    "sequential", globals(),
    "tensorflow.python.keras.engine.sequential")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")

# pylint:enable=g-inconsistent-quotes


@keras_export('keras.experimental.export_saved_model')
def export_saved_model(model,
                       saved_model_path,
                       custom_objects=None,
                       as_text=False,
                       input_signature=None,
                       serving_only=False):
  """Exports a `tf.keras.Model` as a Tensorflow SavedModel.

  Note that at this time, subclassed models can only be saved using
  `serving_only=True`.

  The exported `SavedModel` is a standalone serialization of Tensorflow objects,
  and is supported by TF language APIs and the Tensorflow Serving system.
  To load the model, use the function
  `tf.keras.experimental.load_from_saved_model`.

  The `SavedModel` contains:

  1. a checkpoint containing the model weights.
  2. a `SavedModel` proto containing the Tensorflow backend graph. Separate
     graphs are saved for prediction (serving), train, and evaluation. If
     the model has not been compiled, then only the graph computing predictions
     will be exported.
  3. the model's json config. If the model is subclassed, this will only be
     included if the model's `get_config()` method is overwritten.

  Example:

  ```python
  import tensorflow as tf

  # Create a tf.keras model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=[10]))
  model.summary()

  # Save the tf.keras model in the SavedModel format.
  path = '/tmp/simple_keras_model'
  tf.keras.experimental.export_saved_model(model, path)

  # Load the saved keras model back.
  new_model = tf.keras.experimental.load_from_saved_model(path)
  new_model.summary()
  ```

  Args:
    model: A `tf.keras.Model` to be saved. If the model is subclassed, the flag
      `serving_only` must be set to True.
    saved_model_path: a string specifying the path to the SavedModel directory.
    custom_objects: Optional dictionary mapping string names to custom classes
      or functions (e.g. custom loss functions).
    as_text: bool, `False` by default. Whether to write the `SavedModel` proto
      in text format. Currently unavailable in serving-only mode.
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects, used
      to specify the expected model inputs. See `tf.function` for more details.
    serving_only: bool, `False` by default. When this is true, only the
      prediction graph is saved.

  Raises:
    NotImplementedError: If the model is a subclassed model, and serving_only is
      False.
    ValueError: If the input signature cannot be inferred from the model.
    AssertionError: If the SavedModel directory already exists and isn't empty.
  """
  if serving_only:
    save_lib.save(
        model,
        saved_model_path,
        signatures=saving_utils.trace_model_call(model, input_signature))
  else:
    _save_v1_format(model, saved_model_path, custom_objects, as_text,
                    input_signature)

  try:
    _export_model_json(model, saved_model_path)
  except NotImplementedError:
    logging.warning('Skipped saving model JSON, subclassed model does not have '
                    'get_config() defined.')


def _export_model_json(model, saved_model_path):
  """Saves model configuration as a json string under assets folder."""
  model_json = model.to_json()
  model_json_filepath = os.path.join(
      saved_model_utils.get_or_create_assets_dir(saved_model_path),
      compat.as_text(constants.SAVED_MODEL_FILENAME_JSON))
  file_io.write_string_to_file(model_json_filepath, model_json)


def _export_model_variables(model, saved_model_path):
  """Saves model weights in checkpoint format under variables folder."""
  saved_model_utils.get_or_create_variables_dir(saved_model_path)
  checkpoint_prefix = saved_model_utils.get_variables_path(saved_model_path)
  model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True)
  return checkpoint_prefix


def _save_v1_format(model, path, custom_objects, as_text, input_signature):
  """Exports model to v1 SavedModel format."""
  if not model._is_graph_network:  # pylint: disable=protected-access
    if isinstance(model, sequential.Sequential):
      # If input shape is not directly set in the model, the exported model
      # will infer the expected shapes of the input from the model.
      if not model.built:
        raise ValueError('Weights for sequential model have not yet been '
                         'created. Weights are created when the Model is first '
                         'called on inputs or `build()` is called with an '
                         '`input_shape`, or the first layer in the model has '
                         '`input_shape` during construction.')
      # TODO(kathywu): Build the model with input_signature to create the
      # weights before _export_model_variables().
    else:
      raise NotImplementedError(
          'Subclassed models can only be exported for serving. Please set '
          'argument serving_only=True.')

  builder = saved_model_builder._SavedModelBuilder(path)  # pylint: disable=protected-access

  # Manually save variables to export them in an object-based checkpoint. This
  # skips the `builder.add_meta_graph_and_variables()` step, which saves a
  # named-based checkpoint.
  # TODO(b/113134168): Add fn to Builder to save with object-based saver.
  # TODO(b/113178242): This should only export the model json structure. Only
  # one save is needed once the weights can be copied from the model to clone.
  checkpoint_path = _export_model_variables(model, path)

  # Export each mode. Use ModeKeys enums defined for `Estimator` to ensure that
  # Keras models and `Estimator`s are exported with the same format.
  # Every time a mode is exported, the code checks to see if new variables have
  # been created (e.g. optimizer slot variables). If that is the case, the
  # checkpoint is re-saved to include the new variables.
  export_args = {'builder': builder,
                 'model': model,
                 'custom_objects': custom_objects,
                 'checkpoint_path': checkpoint_path,
                 'input_signature': input_signature}

  has_saved_vars = False
  if model.optimizer:
    if isinstance(model.optimizer, (optimizers.TFOptimizer,
                                    optimizer_v2.OptimizerV2)):
      _export_mode(mode_keys.ModeKeys.TRAIN, has_saved_vars, **export_args)
      has_saved_vars = True
      _export_mode(mode_keys.ModeKeys.TEST, has_saved_vars, **export_args)
    else:
      logging.warning(
          'Model was compiled with an optimizer, but the optimizer is not from '
          '`tf.train` (e.g. `tf.train.AdagradOptimizer`). Only the serving '
          'graph was exported. The train and evaluate graphs were not added to '
          'the SavedModel.')
  _export_mode(mode_keys.ModeKeys.PREDICT, has_saved_vars, **export_args)

  builder.save(as_text)


def _get_var_list(model):
  """Returns list of all checkpointed saveable objects in the model."""
  var_list, _, _ = graph_view.ObjectGraphView(model).serialize_object_graph()
  return var_list


def create_placeholder(spec):
  return K.placeholder(shape=spec.shape, dtype=spec.dtype, name=spec.name)


def _export_mode(
    mode, has_saved_vars, builder, model, custom_objects, checkpoint_path,
    input_signature):
  """Exports a model, and optionally saves new vars from the clone model.

  Args:
    mode: A `tf.estimator.ModeKeys` string.
    has_saved_vars: A `boolean` indicating whether the SavedModel has already
      exported variables.
    builder: A `SavedModelBuilder` object.
    model: A `tf.keras.Model` object.
    custom_objects: A dictionary mapping string names to custom classes
      or functions.
    checkpoint_path: String path to checkpoint.
    input_signature: Nested TensorSpec containing the expected inputs. Can be
      `None`, in which case the signature will be inferred from the model.

  Raises:
    ValueError: If the train/eval mode is being exported, but the model does
      not have an optimizer.
  """
  compile_clone = (mode != mode_keys.ModeKeys.PREDICT)
  if compile_clone and not model.optimizer:
    raise ValueError(
        'Model does not have an optimizer. Cannot export mode %s' % mode)

  model_graph = ops.get_default_graph()
  with ops.Graph().as_default() as g, K.learning_phase_scope(
      mode == mode_keys.ModeKeys.TRAIN):

    if input_signature is None:
      input_tensors = None
    else:
      input_tensors = nest.map_structure(create_placeholder, input_signature)

    # Clone the model into blank graph. This will create placeholders for inputs
    # and targets.
    clone = models_lib.clone_and_build_model(
        model, input_tensors=input_tensors, custom_objects=custom_objects,
        compile_clone=compile_clone)

    # Make sure that iterations variable is added to the global step collection,
    # to ensure that, when the SavedModel graph is loaded, the iterations
    # variable is returned by `tf.compat.v1.train.get_global_step()`. This is
    # required for compatibility with the SavedModelEstimator.
    if compile_clone:
      g.add_to_collection(ops.GraphKeys.GLOBAL_STEP, clone.optimizer.iterations)

    # Extract update and train ops from train/test/predict functions.
    train_op = None
    if mode == mode_keys.ModeKeys.TRAIN:
      clone._make_train_function()  # pylint: disable=protected-access
      train_op = clone.train_function.updates_op
    elif mode == mode_keys.ModeKeys.TEST:
      clone._make_test_function()  # pylint: disable=protected-access
    else:
      clone._make_predict_function()  # pylint: disable=protected-access
    g.get_collection_ref(ops.GraphKeys.UPDATE_OPS).extend(clone.state_updates)

    with session.Session().as_default():
      clone_var_list = _get_var_list(clone)
      if has_saved_vars:
        # Confirm all variables in the clone have an entry in the checkpoint.
        status = clone.load_weights(checkpoint_path)
        status.assert_existing_objects_matched()
      else:
        # Confirm that variables between the clone and model match up exactly,
        # not counting optimizer objects. Optimizer objects are ignored because
        # if the model has not trained, the slot variables will not have been
        # created yet.
        # TODO(b/113179535): Replace with trackable equivalence.
        _assert_same_non_optimizer_objects(model, model_graph, clone, g)

        # TODO(b/113178242): Use value transfer for trackable objects.
        clone.load_weights(checkpoint_path)

        # Add graph and variables to SavedModel.
        # TODO(b/113134168): Switch to add_meta_graph_and_variables.
        clone.save_weights(checkpoint_path, save_format='tf', overwrite=True)
        builder._has_saved_variables = True  # pylint: disable=protected-access

      # Add graph to the SavedModel builder.
      builder.add_meta_graph(
          model_utils.EXPORT_TAG_MAP[mode],
          signature_def_map=_create_signature_def_map(clone, mode),
          saver=saver_lib.Saver(
              clone_var_list,
              # Allow saving Models with no variables. This is somewhat odd, but
              # it's not necessarily a bug.
              allow_empty=True),
          init_op=variables.local_variables_initializer(),
          train_op=train_op)
    return None


def _create_signature_def_map(model, mode):
  """Creates a SignatureDef map from a Keras model."""
  inputs_dict = {name: x for name, x in zip(model.input_names, model.inputs)}
  if model.optimizer:
    targets_dict = {x.name.split(':')[0]: x
                    for x in model._targets if x is not None}  # pylint: disable=protected-access
    inputs_dict.update(targets_dict)
  outputs_dict = {name: x
                  for name, x in zip(model.output_names, model.outputs)}
  metrics = saving_utils.extract_model_metrics(model)

  # Add metric variables to the `LOCAL_VARIABLES` collection. Metric variables
  # are by default not added to any collections. We are doing this here, so
  # that metric variables get initialized.
  local_vars = set(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
  vars_to_add = set()
  if metrics is not None:
    for key, value in six.iteritems(metrics):
      if isinstance(value, metrics_lib.Metric):
        vars_to_add.update(value.variables)
        # Convert Metric instances to (value_tensor, update_op) tuple.
        metrics[key] = (value.result(), value.updates[0])
  # Remove variables that are in the local variables collection already.
  vars_to_add = vars_to_add.difference(local_vars)
  for v in vars_to_add:
    ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)

  export_outputs = model_utils.export_outputs_for_mode(
      mode,
      predictions=outputs_dict,
      loss=model.total_loss if model.optimizer else None,
      metrics=metrics)
  return model_utils.build_all_signature_defs(
      inputs_dict,
      export_outputs=export_outputs,
      serving_only=(mode == mode_keys.ModeKeys.PREDICT))


def _assert_same_non_optimizer_objects(model, model_graph, clone, clone_graph):  # pylint: disable=unused-argument
  """Asserts model and clone contain the same trackable objects."""

  # TODO(fchollet, kathywu): make sure this works in eager mode.
  return True


@keras_export('keras.experimental.load_from_saved_model')
def load_from_saved_model(saved_model_path, custom_objects=None):
  """Loads a keras Model from a SavedModel created by `export_saved_model()`.

  This function reinstantiates model state by:
  1) loading model topology from json (this will eventually come
     from metagraph).
  2) loading model weights from checkpoint.

  Example:

  ```python
  import tensorflow as tf

  # Create a tf.keras model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=[10]))
  model.summary()

  # Save the tf.keras model in the SavedModel format.
  path = '/tmp/simple_keras_model'
  tf.keras.experimental.export_saved_model(model, path)

  # Load the saved keras model back.
  new_model = tf.keras.experimental.load_from_saved_model(path)
  new_model.summary()
  ```

  Args:
    saved_model_path: a string specifying the path to an existing SavedModel.
    custom_objects: Optional dictionary mapping names
        (strings) to custom classes or functions to be
        considered during deserialization.

  Returns:
    a keras.Model instance.
  """
  # restore model topology from json string
  model_json_filepath = os.path.join(
      compat.as_bytes(saved_model_path),
      compat.as_bytes(constants.ASSETS_DIRECTORY),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_JSON))
  model_json = file_io.read_file_to_string(model_json_filepath)
  model = model_from_json(model_json, custom_objects=custom_objects)

  # restore model weights
  checkpoint_prefix = os.path.join(
      compat.as_text(saved_model_path),
      compat.as_text(constants.VARIABLES_DIRECTORY),
      compat.as_text(constants.VARIABLES_FILENAME))
  model.load_weights(checkpoint_prefix)
  return model

################################################################################
# Functional Style/V2 SavedModel functions                                     #
################################################################################

# All serialized attributes are listed within SerializedAttributes classes. See
# the docstring in SerializedAttributes for more context

# All attributes are saved under the 'keras_api' namespace. Only common
# endpoints are attached directly to the root object.
_KERAS_ATTR = 'keras_api'
# Keys for the serialization cache.
# Maps to the keras serialization dict {Layer --> SerializedAttributes object}
_KERAS_CACHE_KEY = 'keras_serialized_attributes'


class SerializedAttributes(object):
  """Class that tracks and validates all serialization attributes.

  Keras models contain many Python-defined components. For example, the
  trainable_variable property lists the model's trainable variables by
  recursively retrieving the trainable variables from each of the child layers.
  Another example is model.call, a python function that calls child layers and
  adds ops to the backend graph.

  Only Tensorflow checkpointable objects and functions can be serialized to
  SavedModel. Serializing a Keras model as-is results in a checkpointable object
  that does not resemble a Keras model at all. Thus, extra checkpointable
  objects and functions must be created during serialization.

  **Defining new serialized attributes**
  Child classes should be defined using:
    SerializedAttributes.with_attributes(
        'name', checkpointable_objects=[...], functions=[...], copy_from=[...])
  This class is used to cache generated checkpointable objects and functions,
  ensuring that new objects and functions are generated a single time.

  **Usage during serialization**
  Each Layer/Model object should have a corresponding instance of
  SerializedAttributes. Create a new instance by calling
  `SerializedAttributes.new(obj)`. Objects and functions may be saved using
  `.set_and_validate_checkpointable_objects`/`.set_and_and_validate_functions`.
  The properties `.checkpointable_objects` and `.functions` returns the cached
  values.

  **Adding/changing attributes to save to SavedModel**
  1. Change the call to `SerializedAttributes.with_attributes` in the correct
     class:
     - CommonEndpoints: Base attributes to be added during serialization. If
       these attributes are present in a Trackable object, it can be
       deserialized to a Keras Model.
     - LayerAttributes: Attributes to serialize for Layer objects.
     - ModelAttributes: Attributes to serialize for Model objects.
  2. Update class docstring
  3. Update arguments to any calls to `set_and_validate_*`. For example, if
     `call_raw_tensors` is added to the ModelAttributes function list, then
     a `call_raw_tensors` function should be passed to
     `set_and_validate_functions`.

  **Common endpoints vs other attributes**
  Only common endpoints are attached directly to the root object. Keras-specific
  attributes are saved to a separate trackable object with the name "keras_api".
  The number of objects attached to the root is limited because any naming
  conflicts will cause user code to break.

  Another reason is that this will only affect users who call
  `tf.saved_model.load` instead of `tf.keras.models.load_model`. These are
  advanced users who are likely to have defined their own tf.functions and
  trackable objects. The added Keras-specific attributes are kept out of the way
  in the "keras_api" namespace.

  Properties defined in this class may be used to filter out keras-specific
  attributes:
  - `functions_to_serialize`: Returns dict of functions to attach to the root
      object.
  - `checkpointable_objects_to_serialize`: Returns dict of objects to attach to
      the root object (including separate trackable object containing
      keras-specific attributes)

  All changes to the serialized attributes must be backwards-compatible, so
  attributes should not be removed or modified without sufficient justification.
  """

  @staticmethod
  def with_attributes(
      name, checkpointable_objects=None, functions=None, copy_from=None):
    """Creates a subclass with all attributes as specified in the arguments.

    Args:
      name: Name of subclass
      checkpointable_objects: List of checkpointable objects to be serialized
        in the SavedModel.
      functions: List of functions to be serialized in the SavedModel.
      copy_from: List of other SerializedAttributes subclasses. The returend
        class will copy checkpoint objects/functions from each subclass.

    Returns:
      Child class with attributes as defined in the `checkpointable_objects`
      and `functions` lists.
    """
    checkpointable_objects = checkpointable_objects or []
    functions = functions or []

    if copy_from is not None:
      for cls in copy_from:
        checkpointable_objects.extend(cls.all_checkpointable_objects)
        functions.extend(cls.all_functions)

    classdict = {
        'all_checkpointable_objects': set(checkpointable_objects),
        'all_functions': set(functions)}
    return type(name, (SerializedAttributes,), classdict)

  @staticmethod
  def new(obj):
    if isinstance(obj, training_lib.Model):
      return ModelAttributes()
    elif isinstance(obj, base_layer.Layer):
      return LayerAttributes()
    else:
      raise TypeError('Internal error during serialization: Expected Keras '
                      'Layer object, got {} of type {}'.format(obj, type(obj)))

  def __init__(self):
    self._object_dict = {}
    self._function_dict = {}
    self._keras_trackable = AutoTrackable()

  @property
  def functions(self):
    """Returns dictionary of all functions."""
    return {key: value for key, value in self._function_dict.items()
            if value is not None}

  @property
  def checkpointable_objects(self):
    """Returns dictionary of all checkpointable objects."""
    return {key: value for key, value in self._object_dict.items()
            if value is not None}

  @property
  def functions_to_serialize(self):
    """Returns functions to attach to the root object during serialization."""
    return {key: value for key, value in self.functions.items()
            if key in CommonEndpoints.all_functions}

  @property
  def objects_to_serialize(self):
    """Returns objects to attach to the root object during serialization."""
    objects = {key: value for key, value in self.checkpointable_objects.items()
               if key in CommonEndpoints.all_checkpointable_objects}
    objects[_KERAS_ATTR] = self._keras_trackable
    return objects

  def set_and_validate_functions(self, function_dict):
    """Saves function dictionary, and validates dictionary values."""
    for key in self.all_functions:
      if key in function_dict:
        if (function_dict[key] is not None and  # Not all functions are required
            not isinstance(function_dict[key],
                           (defun.Function, def_function.Function))):
          raise ValueError(
              'Function dictionary contained a non-function object: {} (for key'
              ' {})'.format(function_dict[key], key))
        self._function_dict[key] = function_dict[key]
        setattr(self._keras_trackable, key, function_dict[key])
      else:
        raise ValueError('Function {} missing from serialized function dict.'
                         .format(key))
    return self.functions

  def set_and_validate_objects(self, object_dict):
    """Saves objects to a dictionary, and validates the values."""
    for key in self.all_checkpointable_objects:
      if key in object_dict:
        if not isinstance(object_dict[key], trackable.Trackable):
          raise ValueError(
              'Object dictionary contained a non-trackable object: {} (for key'
              ' {})'.format(object_dict[key], key))
        self._object_dict[key] = object_dict[key]
        setattr(self._keras_trackable, key, object_dict[key])
      else:
        raise ValueError('Object {} missing from serialized object dict.')
    return self.checkpointable_objects


class CommonEndpoints(SerializedAttributes.with_attributes(
    'CommonEndpoints',
    checkpointable_objects=['variables', 'trainable_variables',
                            'regularization_losses'],
    functions=['__call__', 'call_and_return_all_conditional_losses',
               '_default_save_signature'])):
  """Common endpoints shared by all models loadable by Keras.

  List of all attributes:
    variables: List of all variables in the model and its sublayers.
    trainable_variables: List of all trainable variables in the model and its
      sublayers.
    regulariation_losses: List of all unconditional losses (losses not dependent
      on the inputs) in the model and its sublayers.
    __call__: Function that takes inputs and returns the outputs of the model
      call function.
    call_and_return_all_conditional_losses: Function that returns a tuple of
      (call function outputs, list of all losses that depend on the inputs).
    _default_save_signature: Traced model call function. This is only included
      if the top level exported object is a Keras model.
  """


class LayerAttributes(SerializedAttributes.with_attributes(
    'LayerAttributes',
    checkpointable_objects=['non_trainable_variables', 'layers', 'metrics'],
    functions=['call_and_return_conditional_losses', 'activity_regularizer_fn'],
    copy_from=[CommonEndpoints]
    )):
  """Layer checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from CommonEndpoints
    non_trainable_variables: List of non-trainable variables in the layer and
      its sublayers.
    layers: List of all sublayers.
    metrics: List of all metrics in the layer and its sublayers.
    call_and_return_conditional_losses: Function that takes inputs and returns a
      tuple of (outputs of the call function, list of input-dependent losses).
      The list of losses excludes the activity regularizer function, which is
      separate to allow the deserialized Layer object to define a different
      activity regularizer.
    activity_regularizer_fn: Callable that returns the activity regularizer loss
  """


class ModelAttributes(SerializedAttributes.with_attributes(
    'ModelAttributes',
    copy_from=[LayerAttributes])):
  """Model checkpointable objects + functions that are saved to the SavedModel.

  List of all attributes:
    All attributes from LayerAttributes (including CommonEndpoints)
  """
  # TODO(kathywu): Add attributes `compile_losses` and `compile_metrics`, which
  #  list all losses and metrics defined by `model.compile`.


def serialize_all_attributes(layer, serialization_cache):
  """Serialize all attributes in the layer."""
  save_model_default_signature = False
  if _KERAS_CACHE_KEY not in serialization_cache:
    keras_cache = serialization_cache[_KERAS_CACHE_KEY] = {}
    if isinstance(layer, training_lib.Model):
      # Only trace default signature if the root object is a Model. Since the
      # keras cache key is only created in this method, we know that the object
      # is root if the key does not yet exist in the cache.
      save_model_default_signature = True
  else:
    keras_cache = serialization_cache[_KERAS_CACHE_KEY]

  if layer in keras_cache:
    return keras_cache[layer]
  serialized_attr = keras_cache[layer] = SerializedAttributes.new(layer)

  if _should_skip_serialization(layer):
    return serialized_attr

  object_dict = _wrap_layer_objects(layer, serialization_cache)
  try:
    function_dict = _wrap_layer_functions(layer, serialization_cache,
                                          save_model_default_signature)
  except (ValueError, TypeError) as e:
    logging.warning('Skipping full serialization of object {}, because an '
                    'error occurred while tracing layer functions. Error '
                    'message: {}'.format(layer, e))
  else:
    # Add checkpointable objects and functions to the SerializedAttribute object
    # only if all functions are successfully traced.
    # The `set_and_validate_*` function ensures that all required attributes are
    # exported with the correct type.
    serialized_attr.set_and_validate_objects(object_dict)
    serialized_attr.set_and_validate_functions(function_dict)
  return serialized_attr


def _should_skip_serialization(layer):
  """Skip serializing extra objects and functions if layer inputs aren't set."""
  if isinstance(layer, training_lib.Model):
    try:
      # pylint:disable=pointless-statement
      layer.inputs
      layer.input_names
      # pylint:enable=pointless-statement
    except AttributeError:
      # If the model does not have inputs set, because it was not called or its
      # input shapes were not recorded, we won't have a signature so can't trace
      # a function. But the user may still save an object with this Model
      # attached; we won't fail the whole tf.saved_model.save.
      logging.warning('Skipping full serialization of Keras model {}, because '
                      'its inputs are not defined.'.format(layer))
      return True
    else:
      return False
  else:
    if not layer.built:
      logging.warning('Skipping full serialization of Keras layer {}, because '
                      'it is not built.'.format(layer))
      return True
    return False


def _wrap_layer_objects(layer, serialization_cache):
  """Returns extra trackable objects to attach to the serialized layer.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all checkpointable objects from a
    SerializedAttributes object. See LayerAttributes and ModelAttributes for
    entire list of objects
  """
  # Wrap all regularization losses as tf.functions.
  # First, generate list of all regularization losses in this layer and
  # sublayers.
  regularization_losses = layer._callable_losses[:]  # pylint: disable=protected-access
  for child_layer in _list_all_layers(layer):
    regularization_losses.extend(child_layer._callable_losses)  # pylint: disable=protected-access
  # Next, wrap all loss functions as tf.functions. Use the serialization cache
  # to store already-wrapped functions.
  keras_loss_cache = serialization_cache.setdefault('keras_losses', {})
  wrapped_loss_functions = []
  for loss_fn in regularization_losses:
    if loss_fn in keras_loss_cache:
      wrapped_loss_functions.append(keras_loss_cache[loss_fn])
    else:
      wrapped_loss = _wrap_unconditional_loss(loss_fn, len(keras_loss_cache))
      keras_loss_cache[wrapped_loss] = wrapped_loss
      wrapped_loss_functions.append(wrapped_loss)
  return dict(
      variables=data_structures.ListWrapper(layer.variables),
      trainable_variables=data_structures.ListWrapper(
          layer.trainable_variables),
      non_trainable_variables=data_structures.ListWrapper(
          layer.non_trainable_variables),
      layers=data_structures.ListWrapper(_list_all_layers(layer)),
      metrics=data_structures.ListWrapper(layer.metrics),
      regularization_losses=data_structures.ListWrapper(
          wrapped_loss_functions))


def _wrap_layer_functions(layer, serialization_cache,
                          save_model_default_signature=False):
  """Returns dict of wrapped layer call function and losses in tf.functions.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.
    save_model_default_signature: Whether to save traced model call function.

  Returns:
    A dictionary containing all keras tf.functions to serialize. See
    LayerAttributes and ModelAttributes for the list of all attributes.
  """
  # Reset the losses of the layer and its children. The call function in each
  # child layer is replaced with tf.functions.
  original_attrs = _replace_child_layer_functions(layer, serialization_cache)
  original_layer_losses = layer._losses[:]  # pylint: disable=protected-access
  with trackable.no_automatic_dependency_tracking_scope(layer):
    layer._losses = []  # pylint: disable=protected-access
    # Note that eager losses do not need to be saved since these functions
    # create symbolic losses.

  # Wrap all the layer call and activity regularizer functions.
  call_fn_with_losses = _wrap_call_and_conditional_losses(layer)
  fns = {'call_and_return_conditional_losses': call_fn_with_losses,
         '__call__': _extract_outputs_from_fn(layer, call_fn_with_losses)}

  if save_model_default_signature:
    fns['_default_save_signature'] = saving_utils.trace_model_call(layer)
  else:
    fns['_default_save_signature'] = None

  if layer.activity_regularizer is not None:
    fns['activity_regularizer_fn'] = _wrap_activity_regularizer(layer)
    fns['call_and_return_all_conditional_losses'] = (
        _append_activity_regularizer_loss(
            layer, call_fn_with_losses, fns['activity_regularizer_fn']))
  else:
    fns['activity_regularizer_fn'] = None
    fns['call_and_return_all_conditional_losses'] = call_fn_with_losses

  # Manually trigger traces before restoring the overwritten functions. The
  # functions are traced within the layer call context to ensure that layer
  # functions (e.g. add_loss) behave as though running in graph mode.
  with base_layer_utils.call_context().enter(layer, None, True, None):
    for fn in fns.values():
      if fn is not None and fn.input_signature is not None:
        fn.get_concrete_function()

  # Restore overwritten functions/losses
  with trackable.no_automatic_dependency_tracking_scope(layer):
    layer._losses = original_layer_losses  # pylint: disable=protected-access
  _restore_child_layer_functions(original_attrs)

  return fns


def _list_all_layers(obj):
  if isinstance(obj, training_lib.Model):
    return obj.layers
  else:
    return trackable_layer_utils.filter_empty_layer_containers(obj._layers)  # pylint: disable=protected-access


def _replace_child_layer_functions(layer, serialization_cache):
  """Replaces functions in the children layers with wrapped tf.functions.

  This step allows functions from parent layers to reference the wrapped
  functions from their children layers instead of retracing the ops.

  This function also resets all losses stored in the layer. These are stored in
  the returned dictionary. Use `_restore_child_layer_functions` to restore
  the original attributes.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    Dictionary mapping layer objects -> original functions and losses:
      { Child layer 1: {
          'losses': Original losses,
          'call': Original call function
          'activity_regularizer': Original activity regularizer},
        Child layer 2: ...
      }
  """
  original_attrs = {}
  for child_layer in _list_all_layers(layer):
    # Save symbolic layer losses, which will be restored to maintain the same
    # state.
    original_attrs[child_layer] = {'losses': child_layer._losses[:]}  # pylint: disable=protected-access
    if child_layer not in serialization_cache[_KERAS_CACHE_KEY]:
      layer_fns = (serialize_all_attributes(child_layer, serialization_cache)
                   .functions)
    else:
      layer_fns = serialization_cache[_KERAS_CACHE_KEY][child_layer].functions
    if not layer_fns:
      # This indicates either:
      #   - circular dependency, which means the current layer's functions
      #     should be wrapped first.
      #   - Child layer's inputs are not defined, so its functions have not been
      #     wrapped. In this case, no replacement is necessary so move on to the
      #     next child.
      continue

    original_attrs[child_layer]['call'] = child_layer.call
    original_attrs[child_layer]['activity_regularizer'] = (
        child_layer.activity_regularizer)
    with trackable.no_automatic_dependency_tracking_scope(child_layer):
      child_layer.activity_regularizer = layer_fns.get(
          'activity_regularizer_fn')
      child_layer.call = _use_wrapped_call(
          child_layer, layer_fns['call_and_return_conditional_losses'])
      child_layer._losses = []  # pylint: disable=protected-access
  return original_attrs


def _restore_child_layer_functions(original_attrs):
  """Restores attributes replaced with `_replace_child_layer_functions`."""
  for child_layer, attrs in original_attrs.items():
    with trackable.no_automatic_dependency_tracking_scope(child_layer):
      child_layer._losses = attrs['losses']  # pylint: disable=protected-access
      if 'call' in attrs:
        child_layer.call = attrs['call']
        child_layer.activity_regularizer = attrs['activity_regularizer']


def _use_wrapped_call(layer, call_fn):
  """Creates fn that adds the losses returned by call_fn & returns the outputs.

  Args:
    layer: A Keras layer object
    call_fn: tf.function returned by _wrap_call_and_conditional_losses.

  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  # TODO(kathywu): Support mask argument and multi-input call functions.
  def wrapped_call(inputs, **kwargs):
    """Returns the outputs from the call_fn, and adds the losses."""
    if layer._expects_training_arg:  # pylint: disable=protected-access
      training = kwargs.pop('training', None)
      if training is None:
        training = K.learning_phase()
      training = math_ops.cast(training, dtypes.bool)
      outputs, losses = call_fn(inputs, training=training)
    else:
      outputs, losses = call_fn(inputs)
    layer.add_loss(losses, inputs)
    return outputs
  return wrapped_call


def _wrap_call_and_conditional_losses(layer):
  """Wraps call function that returns a tuple of (outputs, losses).

  The losses returned are conditional on the inputs passed to the call function.
  Unconditional losses (e.g. weight regularizeration) are wrapped separately.

  Args:
    layer: a Keras layer object

  Returns:
    call function that returns outputs and conditional losses -- excludes
    activity regularizer
  """
  if isinstance(layer, RevivedLayer):
    return layer.keras_api.call_and_return_conditional_losses

  if (isinstance(layer.call, def_function.Function) and
      layer.call.input_signature is not None):
    input_signature = layer.call.input_signature
  else:
    if (isinstance(layer, training_lib.Model) and
        saving_utils.model_input_signature(layer) is not None):
      input_signature = saving_utils.model_input_signature(layer)
    elif layer.input_spec is not None:
      input_signature = [nest.map_structure(
          lambda x: input_spec.to_tensor_spec(x, layer.dtype),
          layer.input_spec)]
      # If input spec is too general, then don't define an input signature.
      for spec in nest.flatten(input_signature):
        if spec.shape == tensor_shape.TensorShape(None):
          input_signature = None
          break
    else:
      input_signature = None

    if input_signature is not None and layer._expects_training_arg:  # pylint: disable=protected-access
      input_signature.append(
          tensor_spec.TensorSpec(shape=[], dtype=dtypes.bool))

  # Create function that generates both outputs and losses
  layer_call = layer.call
  if layer._expects_training_arg:  # pylint: disable=protected-access
    def call_and_return_conditional_losses(inputs, training):
      _set_symbolic_learning_phase(training)
      return layer_call(inputs, training=training), layer.get_losses_for(inputs)
  else:
    def call_and_return_conditional_losses(inputs):
      K.set_learning_phase(0)
      return layer_call(inputs), layer.get_losses_for(inputs)
  return def_function.Function(
      call_and_return_conditional_losses,
      '{}_layer_call_and_return_conditional_losses'.format(layer.name),
      input_signature=input_signature,
      # TODO(kathywu): Investigate autograph error.
      autograph=False)


def _extract_outputs_from_fn(layer, call_and_return_conditional_losses):
  """Returns a function that returns only call function outputs."""
  if isinstance(layer, RevivedLayer):
    return layer.keras_api.__call__  # pylint: disable=protected-access
  if layer._expects_training_arg:  # pylint: disable=protected-access
    def call(inputs, training):
      return call_and_return_conditional_losses(inputs, training)[0]
  else:
    def call(inputs):
      return call_and_return_conditional_losses(inputs)[0]
  return def_function.Function(
      call, '{}_layer_call_fn'.format(layer.name),
      input_signature=call_and_return_conditional_losses.input_signature,
      # TODO(kathywu): Investigate autograph error.
      autograph=False)


def _set_symbolic_learning_phase(value):
  """Set learning phase to a tensor value (for internal use only).

  This is used when wrapping call functions as tf.functions that have training
  as a tensor input. Thus, when `learning_phase()` is called, the training
  tensor is returned. This function is called when saving a model to SavedModel.

  Args:
    value: A Tensor object.

  Raises:
    ValueError: If the input value is not a graph tensor
  """
  graph = K.get_graph()
  if not isinstance(value, ops.Tensor):
    raise ValueError('Symbolic learning phase must be a graph tensor.')
  K._GRAPH_LEARNING_PHASES[graph] = value  # pylint: disable=protected-access


def _append_activity_regularizer_loss(
    layer, call_fn_with_losses, activity_regularizer_fn):
  """Appends activity regularizer loss to losses returned by the wrapped fn."""
  def fn(*args):
    outputs, losses = call_fn_with_losses(*args)
    losses.append(activity_regularizer_fn(outputs))
    return outputs, losses
  return def_function.Function(
      fn,
      '{}_layer_call_and_return_all_conditional_losses'.format(layer.name),
      input_signature=call_fn_with_losses.input_signature,
      # TODO(kathywu): Investigate autograph error.
      autograph=False)


def _wrap_unconditional_loss(loss_fn, index):
  """Wraps callable/unconditonal loss, returning a serializable function."""
  # Extract original loss function from partial function
  fn = loss_fn.args[0] if isinstance(loss_fn, functools.partial) else loss_fn
  if isinstance(fn, def_function.Function):
    return fn
  else:
    return def_function.Function(
        fn, 'loss_fn_{}'.format(index), input_signature=[])


def _wrap_activity_regularizer(layer):
  """Wraps the activity regularizer."""
  if isinstance(layer.activity_regularizer, def_function.Function):
    return layer.activity_regularizer
  return def_function.Function(
      layer.activity_regularizer,
      '{}_activity_regularizer'.format(layer.name),
      input_signature=[tensor_spec.TensorSpec(None, layer.dtype or K.floatx())])


def load_from_saved_model_v2(path, compile=True):  # pylint: disable=redefined-builtin
  """Loads Keras objects from a SavedModel.

  Any Keras layer or model saved to the SavedModel will be loaded back
  as Keras objects. Other objects are loaded as regular trackable objects (same
  as `tf.saved_model.load`).

  Currently, Keras saving/loading only retains the Keras object's weights,
  losses, and call function.

  The loaded model can be re-compiled, but the original optimizer, compiled loss
  functions, and metrics are not retained. This is temporary, and `model.save`
  will soon be able to serialize compiled models.

  Args:
    path: Path to SavedModel.
    compile: If true, compile the model after loading it.

  Returns:
    Object loaded from SavedModel.
  """
  # TODO(kathywu): Add saving/loading of optimizer, compiled losses and metrics.
  # TODO(kathywu): Add code to load from objects that contain all endpoints
  model = load.load_internal(path, loader_cls=KerasObjectLoader)

  if isinstance(model, RevivedModel) and compile:
    # TODO(kathywu): Use compiled objects from SavedModel, instead of
    # creating new objects from the training config.
    if model._training_config is not None:  # pylint: disable=protected-access
      model.compile(**saving_utils.compile_args_from_training_config(
          model._training_config))  # pylint: disable=protected-access

  return model

PUBLIC_ATTRIBUTES = CommonEndpoints.all_functions.union(
    CommonEndpoints.all_checkpointable_objects)
PUBLIC_ATTRIBUTES.add(_KERAS_ATTR)


class KerasObjectLoader(load.Loader):
  """Loader that recreates Keras objects."""

  def __init__(self, *args, **kwargs):
    super(KerasObjectLoader, self).__init__(*args, **kwargs)
    self._finalize()

  def _finalize(self):
    # pylint: disable=protected-access
    for node in self._nodes:
      if isinstance(node, RevivedModel):
        input_signature = (
            node.keras_api.call_and_return_conditional_losses.input_signature[0]
            )
        if isinstance(node, RevivedSequential):
          with trackable.no_automatic_dependency_tracking_scope(node):
            node._layers = []
          for layer in node.keras_api.layers:
            node.add(layer)

        if not node.inputs:
          # Since this revived object is technically a subclassed model (even if
          # the original model is functional/sequential), inputs should be set.
          node._set_inputs(input_signature)
      if isinstance(node, RevivedLayer):
        losses = node._serialized_attributes.get('regularization_losses', [])
        for loss in losses:
          node.add_loss(loss)

        # Use wrapped activity regularizer function if the layer's activity
        # regularizer wasn't created during initialization.
        if node.activity_regularizer is None:
          node.activity_regularizer = getattr(node.keras_api,
                                              'activity_regularizer_fn', None)

        # Now that the node object has been fully loaded and restored from the,
        # checkpoint, the object no longer needs to track objects added from
        # SerializedAttributes. (Note that saving a training checkpoint still
        # functions correctly, because layers and variables are tracked
        # separately by the Layer object.)
        # TODO(kathywu): Instead of outright deleting these nodes (which would
        # make restoring from a different checkpoint tricky), mark them as extra
        # dependencies that are OK to overwrite.
        for name in PUBLIC_ATTRIBUTES:
          delete_tracking(node, name)

    # pylint: enable=protected-access

  def _recreate_base_user_object(self, proto):
    revived_classes = {
        '_tf_keras_layer': (RevivedLayer, base_layer.Layer),
        '_tf_keras_network': (RevivedNetwork, network_lib.Network),
        '_tf_keras_model': (RevivedModel, training_lib.Model),
        '_tf_keras_sequential': (RevivedSequential, models_lib.Sequential)
    }

    parent_classes = revived_classes.get(proto.identifier, None)

    if parent_classes is not None:
      parent_classes = revived_classes[proto.identifier]
      metadata = json.loads(proto.metadata)
      revived_cls = type(
          compat.as_str(metadata['class_name']),
          parent_classes,
          {'__setattr__': parent_classes[1].__setattr__})
      obj = revived_cls._init_from_metadata(metadata)  # pylint: disable=protected-access
      return obj, revived_cls._revive_setter  # pylint: disable=protected-access

    return super(KerasObjectLoader, self)._recreate_base_user_object(proto)


# TODO(kathywu): Centrally define keys and functions for both  serialization and
# deserialization.
class RevivedLayer(object):
  """Keras layer loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived layer from metadata stored in the SavedModel proto."""
    init_args = dict(
        name=metadata['name'],
        trainable=metadata['trainable'])
    if metadata.get('dtype') is not None:
      init_args['dtype'] = metadata['dtype']
    if metadata.get('batch_input_shape') is not None:
      init_args['batch_input_shape'] = metadata['batch_input_shape']

    revived_obj = cls(**init_args)

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      # pylint:disable=protected-access
      revived_obj._expects_training_arg = metadata['expects_training_arg']
      if metadata.get('config') is not None:
        revived_obj._config = metadata['config']
      if metadata.get('input_spec') is not None:
        revived_obj.input_spec = recursively_deserialize_keras_object(
            metadata['input_spec'],
            module_objects={'InputSpec': input_spec.InputSpec})
      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])

      # Store attributes revived from SerializedAttributes in a un-tracked
      # dictionary. The attributes are the ones listed in CommonEndpoints or
      # "keras_api" for keras-specific attributes.
      revived_obj._serialized_attributes = {}
      # pylint:enable=protected-access

    return revived_obj

  def _revive_setter(self, name, value):
    """Reattaches attributes from the SavedModel to the newly revived object."""
    if name in PUBLIC_ATTRIBUTES:
      if isinstance(value, trackable.Trackable):
        self._track_trackable(value, name=name)
      self._serialized_attributes[name] = value
    else:
      setattr(self, name, value)

  @property
  def keras_api(self):
    return self._serialized_attributes[_KERAS_ATTR]

  def get_config(self):
    if hasattr(self, '_config'):
      return self._config
    else:
      raise NotImplementedError

  def call(self, inputs, *args, **kwargs):
    """Calls the revived layer and add conditional losses."""
    call_fn = _use_wrapped_call(
        self, self.keras_api.call_and_return_conditional_losses)
    return call_fn(inputs, *args, **kwargs)


def recursively_deserialize_keras_object(config, module_objects=None):
  """Deserialize Keras object from a nested structure."""
  if isinstance(config, dict):
    if 'class_name' in config:
      return deserialize_keras_object(config, module_objects=module_objects)
    else:
      return {key: recursively_deserialize_keras_object(config[key],
                                                        module_objects)
              for key in config}
  if isinstance(config, (tuple, list)):
    return [recursively_deserialize_keras_object(x, module_objects)
            for x in config]
  else:
    raise ValueError('Unable to decode config: {}'.format(config))


class RevivedNetwork(RevivedLayer):
  """Keras network of layers loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived network from metadata stored in the SavedModel proto."""
    # TODO(kathywu): Refactor logic here so that RevivedNetwork uses the
    revived_obj = cls(name=metadata['name'])

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      # pylint:disable=protected-access
      if metadata.get('dtype') is not None:
        revived_obj._dtype = metadata['dtype']
      revived_obj.trainable = metadata['trainable']

      revived_obj._expects_training_arg = metadata['expects_training_arg']
      if metadata.get('config') is not None:
        revived_obj._config = metadata['config']

      if metadata.get('activity_regularizer') is not None:
        revived_obj.activity_regularizer = regularizers.deserialize(
            metadata['activity_regularizer'])

      # Store attributes revived from SerializedAttributes in a un-tracked
      # dictionary. The attributes are the ones listed in CommonEndpoints or
      # "keras_api" for keras-specific attributes.
      revived_obj._serialized_attributes = {}
      # pylint:enable=protected-access

    return revived_obj


class RevivedModel(RevivedNetwork):
  """Keras model loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived model from metadata stored in the SavedModel proto."""
    revived_obj = super(RevivedModel, cls)._init_from_metadata(metadata)

    with trackable.no_automatic_dependency_tracking_scope(revived_obj):
      revived_obj._training_config = metadata.get('training_config')  # pylint:disable=protected-access

    return revived_obj


class RevivedSequential(RevivedModel):
  """Keras sequential model loaded from a SavedModel."""

  @classmethod
  def _init_from_metadata(cls, metadata):
    """Create revived Sequential model from SavedModel metadata."""
    revived_obj = super(RevivedSequential, cls)._init_from_metadata(metadata)
    return revived_obj

  def call(self, *args, **kwargs):
    return models_lib.Sequential.call(self, *args, **kwargs)


def save(model, filepath, overwrite, include_optimizer):
  """Saves a model as a SavedModel to the filepath.

  Args:
    model: Keras model instance to be saved.
    filepath: String path to save the model.
    overwrite: whether to overwrite the existing filepath.
    include_optimizer: If True, save the model's optimizer state.

  Raises:
    ValueError: if the model's inputs have not been defined.
  """
  # If file exists and should not be overwritten.
  if not overwrite and os.path.exists(filepath):
    proceed = ask_to_proceed_with_overwrite(filepath)
    if not proceed:
      return

  if _should_skip_serialization(model):
    saving_utils.raise_model_input_error(model)

  if not include_optimizer:
    orig_optimizer = model.optimizer
    model.optimizer = None

  save_lib.save(model, filepath)

  if not include_optimizer:
    model.optimizer = orig_optimizer
