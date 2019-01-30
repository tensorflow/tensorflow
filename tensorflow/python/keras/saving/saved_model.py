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
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.saving import model_from_json
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import model_utils
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import mode_keys
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.checkpointable import util as checkpointable_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.experimental.export')
def export(
    model, saved_model_path, custom_objects=None, as_text=None,
    input_signature=None, serving_only=False):
  """Saves a `tf.keras.Model` into Tensorflow SavedModel format.

  `save_model` generates new files/folders under the `saved_model_path` folder:
  1) a checkpoint containing the model weights.
  2) a saved_model.pb file containing the model's MetaGraphs. The prediction
     graph is always exported. The evaluation and training graphs are exported
     if the following conditions are met:
     - Evaluation: model loss is defined.
     - Training: model is compiled with an optimizer defined under `tf.train`.
       This is because `tf.keras.optimizers.Optimizer` instances cannot be
       saved to checkpoints.
  3) Model's json configuration, if model.get_config() has been implemented.
     This file can be used to reload the model using
     tf.keras.models.model_from_json(). Note that if any custom objects were
     used, they should be passed to the `custom_object` argument when loading
     the model.

  Model limitations:
  - Sequential and functional models can always be saved.
  - Subclassed models can only be saved when `serving_only=True`. This is due to
    the current implementation copying the model in order to export the training
    and evaluation graphs. Because the topology of subclassed models cannot be
    determined, the subclassed models cannot be cloned. Subclassed models will
    be entirely exportable in the future.

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
  saved_to_path = tf.keras.experimental.export(
        model, '/tmp/my_simple_tf_keras_saved_model')

  # Load the saved keras model back.
  model_prime = tf.keras.experimental.load_from_saved_model(saved_to_path)
  model_prime.summary()
  ```

  Args:
    model: A `tf.keras.Model` to be saved. If the model is subclassed, the flag
      `serving_only` must be set to True.
    saved_model_path: a string specifying the path to the SavedModel directory.
      The SavedModel will be saved to a timestamped folder created within this
      directory.
    custom_objects: Optional dictionary mapping string names to custom classes
      or functions (e.g. custom loss functions).
    as_text: whether to write the `SavedModel` proto in text format. Currently
      unavailable in serving-only mode.
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects, used
      to specify the expected model inputs. `input_signature`'s nested structure
      should match the expected nested structure of the inputs to the model. If
      this is not set, this function will attempt to infer the input shapes and
      dtypes from the model. Note that if the model is subclassed, the tensor
      inputs to the call function should be nested in the first argument (this
      is a general requirement for using subclassed models with Keras functions
      .fit(), .predict(), etc.).
    serving_only: Export only the outputs produced from calling the model in
      predict mode. The losses, optimizer, and other training configurations are
      not saved. If the SavedModel will only be used for serving (rather than
      retraining), or if the model is subclassed, this can be set to True.

  Returns:
    String path to the SavedModel folder, a subdirectory of `saved_model_path`.

  Raises:
    NotImplementedError: If the model is a subclassed model, and serving_only is
      False.
    ValueError: If the input signature cannot be inferred from the model.
  """
  export_dir = model_utils.get_timestamped_export_dir(saved_model_path)

  if serving_only:
    save_lib.save(
        model, export_dir,
        signatures=saving_utils.trace_model_call(model, input_signature))
  else:
    _save_v1_format(model, export_dir, custom_objects, as_text, input_signature)

  try:
    _export_model_json(model, export_dir)
  except NotImplementedError:
    logging.warning('Skipped saving model JSON, subclassed model does not have '
                    'get_config() defined.')

  return export_dir


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
  from tensorflow.python.keras.engine import sequential  # pylint: disable=g-import-not-at-top

  if not model._is_graph_network:
    if isinstance(model, sequential.Sequential):
      # If input shape is not directly set in the model, the exported model
      # will infer the expected shapes of the input from the model.
      if not model.built and input_signature is None:
        raise ValueError(
            'Sequential model\'s input shape is unknown. Please build the '
            'model, or use the input_signature argument to specify the '
            'model inputs.')
    else:
      raise NotImplementedError(
          'Subclassed models can only be exported for serving. Please set '
          'argument serving_only=True.')

  builder = saved_model_builder._SavedModelBuilder(path)

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
    # TODO(kathywu): Verify this works with v2 optimizer.
    if isinstance(model.optimizer, optimizers.TFOptimizer):
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
  return checkpointable_utils.named_saveables(model)


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
  from tensorflow.python.keras import models as models_lib  # pylint: disable=g-import-not-at-top
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
    # variable is returned by `tf.train.get_global_step()`. This is required for
    # compatibility with the SavedModelEstimator.
    if compile_clone:
      g.add_to_collection(ops.GraphKeys.GLOBAL_STEP, clone.optimizer.iterations)

    # Extract update and train ops from train/test/predict functions.
    train_op = None
    if mode == mode_keys.ModeKeys.TRAIN:
      clone._make_train_function()
      train_op = clone.train_function.updates_op
    elif mode == mode_keys.ModeKeys.TEST:
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
        model_utils.EXPORT_TAG_MAP[mode],
        signature_def_map=_create_signature_def_map(clone, mode),
        saver=saver_lib.Saver(clone_var_list),
        init_op=variables.local_variables_initializer(),
        train_op=train_op)
    return None


def _create_signature_def_map(model, mode):
  """Creates a SignatureDef map from a Keras model."""
  inputs_dict = {name: x for name, x in zip(model.input_names, model.inputs)}
  if model.optimizer:
    targets_dict = {x.name.split(':')[0]: x
                    for x in model.targets if x is not None}
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
    from tensorflow.python.keras.metrics import Metric  # pylint: disable=g-import-not-at-top
    for key, value in six.iteritems(metrics):
      if isinstance(value, Metric):
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
  """Asserts model and clone contain the same checkpointable objects."""

  # TODO(fchollet, kathywu): make sure this works in eager mode.
  return True


@keras_export('keras.experimental.load_from_saved_model')
def load_from_saved_model(saved_model_path):
  """Loads a keras.Model from a SavedModel created by keras export().

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
  saved_to_path = tf.keras.experimental.export(
        model, '/tmp/my_simple_tf_keras_saved_model')

  # Load the saved keras model back.
  model_prime = tf.keras.experimental.load_from_saved_model(saved_to_path)
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
