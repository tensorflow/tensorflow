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
"""Keras SavedModel serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.util.lazy_loader import LazyLoader

# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")


def save(model, filepath, overwrite, include_optimizer, signatures=None,
         options=None):
  """Saves a model as a SavedModel to the filepath.

  Args:
    model: Keras model instance to be saved.
    filepath: String path to save the model.
    overwrite: whether to overwrite the existing filepath.
    include_optimizer: If True, save the model's optimizer state.
    signatures: Signatures to save with the SavedModel. Applicable to the 'tf'
      format only. Please see the `signatures` argument in `tf.saved_model.save`
      for details.
    options: Optional`tf.saved_model.SaveOptions` object that specifies
      options for saving to SavedModel.

  Raises:
    ValueError: if the model's inputs have not been defined.
  """
  # If file exists and should not be overwritten.
  if not overwrite and os.path.exists(filepath):
    proceed = ask_to_proceed_with_overwrite(filepath)
    if not proceed:
      return

  if save_impl.should_skip_serialization(model):
    saving_utils.raise_model_input_error(model)

  if not include_optimizer:
    orig_optimizer = model.optimizer
    model.optimizer = None

  # Trace all functions and signatures with `training=0` instead of using the
  # default learning phase placeholder.
  with K.learning_phase_scope(0):
    # When saving a model involving batch norm layer within a strategy scope,
    # the replica context is not available when calling `add_update()`, and thus
    # we use the default replica context here.
    with distribution_strategy_context._get_default_replica_context():  # pylint: disable=protected-access
      save_lib.save(model, filepath, signatures, options)

  if not include_optimizer:
    model.optimizer = orig_optimizer
