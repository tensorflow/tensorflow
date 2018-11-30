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
# pylint: disable=protected-access
"""Utility functions to save/load keras Model to/from SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

from tensorflow.python.client import session
from tensorflow.python.estimator import keras as estimator_keras_util
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export as export_helpers
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import models as models_lib
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.checkpointable import util as checkpointable_utils
from tensorflow.python.util import compat


def save_keras_model(
    model, saved_model_path, custom_objects=None, as_text=None):
  """Save a `tf.keras.Model` into Tensorflow SavedModel format.

  `save_model` generates new files/folders under the `saved_model_path` folder:
  1) an asset folder containing the json string of the model's
     configuration (topology).
  2) a checkpoint containing the model weights.
  3) a saved_model.pb file containing the model's MetaGraphs. The prediction
     graph is always exported. The evaluaton and training graphs are exported
     if the following conditions are met:
     - Evaluation: model loss is defined.
     - Training: model is compiled with an optimizer defined under `tf.train`.
       This is because `tf.keras.optimizers.Optimizer` instances cannot be
       saved to checkpoints.

  Model Requirements:
  - Model must be a sequential model or functional model. Subclassed models can
    not be saved via this function, unless you provide an implementation for
    get_config() and from_config().
  - All variables must be saveable by the model. In general, this condition is
    met through the use of layers defined in the keras library. However,
    there is currently a bug with variables created in Lambda layer functions
    not being saved correctly (see
    https://github.com/keras-team/keras/issues/9740).

  Note that each mode is exported in separate graphs, so different modes do not
  share variables. To use the train graph with evaluation or prediction graphs,
  create a new checkpoint if variable values have been updated.

  Example:

  ```python
  import tensorflow as tf

  # Create a tf.keras model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=[10]))
  model.summary()

  # Save the tf.keras model in the SavedModel format.
  saved_to_path = tf.contrib.saved_model.save_keras_model(
        model, '/tmp/my_simple_tf_keras_saved_model')

  # Load the saved keras model back.
  model_prime = tf.contrib.saved_model.load_keras_model(saved_to_path)
  model_prime.summary()
  ```

  Args:
    model: A `tf.keras.Model` to be saved.
    saved_model_path: a string specifying the path to the SavedModel directory.
      The SavedModel will be saved to a timestamped folder created within this
      directory.
    custom_objects: Optional dictionary mapping string names to custom classes
      or functions (e.g. custom loss functions).
    as_text: whether to write the `SavedModel` proto in text format.

  Returns:
    String path to the SavedModel folder, a subdirectory of `saved_model_path`.

  Raises:
    NotImplementedError: If the model is a subclassed model.
    ValueError: If a Sequential model does not have input shapes defined by the
      user, and is not built.
  """
  if not model._is_graph_network:
    if isinstance(model, sequential.Sequential):
      # If input shape is not directly set in the model, the exported model
      # will assume that the inputs have the same shape as the shape the model
      # was built model with.
      if not model.built:
        raise ValueError(
            'Sequential model must be built before it can be exported.')
    else:
      raise NotImplementedError(
          'Exporting subclassed models is not yet supported.')

  export_dir = export_helpers.get_timestamped_export_dir(saved_model_path)
  temp_export_dir = export_helpers.get_temp_export_dir(export_dir)

  builder = saved_model_builder._SavedModelBuilder(temp_export_dir)

  # Manually save variables to export them in an object-based checkpoint. This
  # skips the `builder.add_meta_graph_and_variables()` step, which saves a
  # named-based checkpoint.
  # TODO(b/113134168): Add fn to Builder to save with object-based saver.
  # TODO(b/113178242): This should only export the model json structure. Only
  # one save is needed once the weights can be copied from the model to clone.
  checkpoint_path = _export_model_json_and_variables(model, temp_export_dir)

  # Export each mode. Use ModeKeys enums defined for `Estimator` to ensure that
  # Keras models and `Estimator`s are exported with the same format.
  # Every time a mode is exported, the code checks to see if new variables have
  # been created (e.g. optimizer slot variables). If that is the case, the
  # checkpoint is re-saved to include the new variables.
  export_args = {'builder': builder,
                 'model': model,
                 'custom_objects': custom_objects,
                 'checkpoint_path': checkpoint_path}

  has_saved_vars = False
  if model.optimizer:
    if isinstance(model.optimizer, optimizers.TFOptimizer):
      _export_mode(model_fn_lib.ModeKeys.TRAIN, has_saved_vars, **export_args)
      has_saved_vars = True
      _export_mode(model_fn_lib.ModeKeys.EVAL, has_saved_vars, **export_args)
    else:
      logging.warning(
          'Model was compiled with an optimizer, but the optimizer is not from '
          '`tf.train` (e.g. `tf.train.AdagradOptimizer`). Only the serving '
          'graph was exported. The train and evaluate graphs were not added to '
          'the SavedModel.')
  _export_mode(model_fn_lib.ModeKeys.PREDICT, has_saved_vars, **export_args)

  builder.save(as_text)

  gfile.Rename(temp_export_dir, export_dir)
  return export_dir


def _export_model_json_and_variables(model, saved_model_path):
  """Save model variables and json structure into SavedModel subdirectories."""
  # Save model configuration as a json string under assets folder.
  model_json = model.to_json()
  model_json_filepath = os.path.join(
      saved_model_utils.get_or_create_assets_dir(saved_model_path),
      compat.as_text(constants.SAVED_MODEL_FILENAME_JSON))
  file_io.write_string_to_file(model_json_filepath, model_json)

  # Save model weights in checkpoint format under variables folder.
  saved_model_utils.get_or_create_variables_dir(saved_model_path)
  checkpoint_prefix = saved_model_utils.get_variables_path(saved_model_path)
  model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True)
  return checkpoint_prefix


def _get_var_list(model):
  """Return list of all checkpointed saveable objects in the model."""
  return checkpointable_utils.named_saveables(model)


def _export_mode(
    mode, has_saved_vars, builder, model, custom_objects, checkpoint_path):
  """Export a model, and optionally save new vars from the clone model.

  Args:
    mode: A `tf.estimator.ModeKeys` string.
    has_saved_vars: A `boolean` indicating whether the SavedModel has already
      exported variables.
    builder: A `SavedModelBuilder` object.
    model: A `tf.keras.Model` object.
    custom_objects: A dictionary mapping string names to custom classes
      or functions.
    checkpoint_path: String path to checkpoint.

  Raises:
    ValueError: If the train/eval mode is being exported, but the model does
      not have an optimizer.
  """
  compile_clone = (mode != model_fn_lib.ModeKeys.PREDICT)
  if compile_clone and not model.optimizer:
    raise ValueError(
        'Model does not have an optimizer. Cannot export mode %s' % mode)

  model_graph = ops.get_default_graph()
  with ops.Graph().as_default() as g:

    K.set_learning_phase(mode == model_fn_lib.ModeKeys.TRAIN)

    # Clone the model into blank graph. This will create placeholders for inputs
    # and targets.
    clone = models_lib.clone_and_build_model(
        model, custom_objects=custom_objects, compile_clone=compile_clone)

    # Make sure that iterations variable is added to the global step collection,
    # to ensure that, when the SavedModel graph is loaded, the iterations
    # variable is returned by `tf.train.get_global_step()`. This is required for
    # compatibility with the SavedModelEstimator.
    if compile_clone:
      g.add_to_collection(ops.GraphKeys.GLOBAL_STEP, clone.optimizer.iterations)

    # Extract update and train ops from train/test/predict functions.
    train_op = None
    if mode == model_fn_lib.ModeKeys.TRAIN:
      clone._make_train_function()
      train_op = clone.train_function.updates_op
    elif mode == model_fn_lib.ModeKeys.EVAL:
      clone._make_test_function()
    else:
      clone._make_predict_function()
    g.get_collection_ref(ops.GraphKeys.UPDATE_OPS).extend(clone.state_updates)

    clone_var_list = checkpointable_utils.named_saveables(clone)

    with session.Session().as_default():
      if has_saved_vars:
        # Confirm all variables in the clone have an entry in the checkpoint.
        status = clone.load_weights(checkpoint_path)
        status.assert_existing_objects_matched()
      else:
        # Confirm that variables between the clone and model match up exactly,
        # not counting optimizer objects. Optimizer objects are ignored because
        # if the model has not trained, the slot variables will not have been
        # created yet.
        # TODO(b/113179535): Replace with checkpointable equivalence.
        _assert_same_non_optimizer_objects(model, model_graph, clone, g)

        # TODO(b/113178242): Use value transfer for checkpointable objects.
        clone.load_weights(checkpoint_path)

        # Add graph and variables to SavedModel.
        # TODO(b/113134168): Switch to add_meta_graph_and_variables.
        clone.save_weights(checkpoint_path, save_format='tf', overwrite=True)
        builder._has_saved_variables = True

    # Add graph to the SavedModel builder.
    builder.add_meta_graph(
        model_fn_lib.EXPORT_TAG_MAP[mode],
        signature_def_map=_create_signature_def_map(clone, mode),
        saver=saver_lib.Saver(clone_var_list),
        init_op=variables.local_variables_initializer(),
        train_op=train_op)
    return None


def _create_signature_def_map(model, mode):
  """Create a SignatureDef map from a Keras model."""
  inputs_dict = {name: x for name, x in zip(model.input_names, model.inputs)}
  if model.optimizer:
    targets_dict = {x.name.split(':')[0]: x
                    for x in model.targets if x is not None}
    inputs_dict.update(targets_dict)
  outputs_dict = {name: x
                  for name, x in zip(model.output_names, model.outputs)}
  metrics = estimator_keras_util._convert_keras_metrics_to_estimator(model)

  # Add metric variables to the `LOCAL_VARIABLES` collection. Metric variables
  # are by default not added to any collections. We are doing this here, so
  # that metric variables get initialized.
  local_vars = set(ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
  vars_to_add = set()
  if metrics is not None:
    for key, value in six.iteritems(metrics):
      if isinstance(value, Metric):
        vars_to_add.update(value.variables)
        # Convert Metric instances to (value_tensor, update_op) tuple.
        metrics[key] = (value.result(), value.updates[0])
  # Remove variables that are in the local variables collection already.
  vars_to_add = vars_to_add.difference(local_vars)
  for v in vars_to_add:
    ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)

  export_outputs = model_fn_lib.export_outputs_for_mode(
      mode,
      predictions=outputs_dict,
      loss=model.total_loss if model.optimizer else None,
      metrics=metrics)
  return export_helpers.build_all_signature_defs(
      inputs_dict,
      export_outputs=export_outputs,
      serving_only=(mode == model_fn_lib.ModeKeys.PREDICT))


def _assert_same_non_optimizer_objects(model, model_graph, clone, clone_graph):  # pylint: disable=unused-argument
  """Assert model and clone contain the same checkpointable objects."""

  # TODO(fchollet, kathywu): make sure this works in eager mode.
  return True


def load_keras_model(saved_model_path):
  """Load a keras.Model from SavedModel.

  load_model reinstantiates model state by:
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
  saved_to_path = tf.contrib.saved_model.save_keras_model(
        model, '/tmp/my_simple_tf_keras_saved_model')

  # Load the saved keras model back.
  model_prime = tf.contrib.saved_model.load_keras_model(saved_to_path)
  model_prime.summary()
  ```

  Args:
    saved_model_path: a string specifying the path to an existing SavedModel.

  Returns:
    a keras.Model instance.
  """
  # restore model topology from json string
  model_json_filepath = os.path.join(
      compat.as_bytes(saved_model_path),
      compat.as_bytes(constants.ASSETS_DIRECTORY),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_JSON))
  model_json = file_io.read_file_to_string(model_json_filepath)
  model = model_from_json(model_json)

  # restore model weights
  checkpoint_prefix = os.path.join(
      compat.as_text(saved_model_path),
      compat.as_text(constants.VARIABLES_DIRECTORY),
      compat.as_text(constants.VARIABLES_FILENAME))
  model.load_weights(checkpoint_prefix)
  return model
