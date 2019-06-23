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
"""Deprecated experimental Keras SavedModel implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_from_json
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import model_utils
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
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
sequential = LazyLoader(
    "sequential", globals(),
    "tensorflow.python.keras.engine.sequential")
# pylint:enable=g-inconsistent-quotes


@deprecation.deprecated(
    date=None,
    instructions=('Please use `model.save(..., save_format="tf")` or '
                  '`tf.keras.models.save_model(..., save_format="tf")`.'))
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


@deprecation.deprecated(
    date=None,
    instructions=('The experimental save and load functions have been  '
                  'deprecated. Please switch to `tf.keras.models.load_model`.'))
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
