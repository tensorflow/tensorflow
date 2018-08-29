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

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat


def save_model(model, saved_model_path):
  """Save a `tf.keras.Model` into Tensorflow SavedModel format.

  `save_model` generates such files/folders under the `saved_model_path` folder:
  1) an asset folder containing the json string of the model's
  configuration(topology).
  2) a checkpoint containing the model weights.

  Note that subclassed models can not be saved via this function, unless you
  provide an implementation for get_config() and from_config().
  Also note that `tf.keras.optimizers.Optimizer` instances can not currently be
  saved to checkpoints. Use optimizers from `tf.train`.

  Args:
    model: A `tf.keras.Model` to be saved.
    saved_model_path: a string specifying the path to the SavedModel directory.

  Raises:
    NotImplementedError: If the passed in model is a subclassed model.
  """
  if not model._is_graph_network:
    raise NotImplementedError

  # save model configuration as a json string under assets folder.
  model_json = model.to_json()
  assets_destination_dir = os.path.join(
      compat.as_bytes(saved_model_path),
      compat.as_bytes(constants.ASSETS_DIRECTORY))

  if not file_io.file_exists(assets_destination_dir):
    file_io.recursive_create_dir(assets_destination_dir)

  model_json_filepath = os.path.join(
      compat.as_bytes(assets_destination_dir),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_JSON))
  file_io.write_string_to_file(model_json_filepath, model_json)

  # save model weights in checkpoint format.
  checkpoint_destination_dir = os.path.join(
      compat.as_bytes(saved_model_path),
      compat.as_bytes(constants.VARIABLES_DIRECTORY))

  if not file_io.file_exists(checkpoint_destination_dir):
    file_io.recursive_create_dir(checkpoint_destination_dir)

  checkpoint_prefix = os.path.join(
      compat.as_text(checkpoint_destination_dir),
      compat.as_text(constants.VARIABLES_FILENAME))
  model.save_weights(checkpoint_prefix, save_format='tf', overwrite=True)


def load_model(saved_model_path):
  """Load a keras.Model from SavedModel.

  load_model reinstantiates model state by:
  1) loading model topology from json (this will eventually come
     from metagraph).
  2) loading model weights from checkpoint.

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
