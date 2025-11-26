# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Model loading utilities for tf_model_summary CLI."""

import os

from tensorflow.python.keras.saving import save as keras_save
from tensorflow.python.platform import tf_logging as logging


class ModelLoadError(Exception):
  """Raised when a model cannot be loaded."""
  pass


def detect_format(model_path):
  """Detect the format of a model file or directory.

  Args:
    model_path: Path to the model file or directory.

  Returns:
    A string indicating the format: 'savedmodel', 'h5', 'keras', or 'unknown'.
  """
  if not os.path.exists(model_path):
    return 'not_found'

  # Check if it's a directory (SavedModel)
  if os.path.isdir(model_path):
    # SavedModel should contain saved_model.pb or saved_model.pbtxt
    pb_path = os.path.join(model_path, 'saved_model.pb')
    pbtxt_path = os.path.join(model_path, 'saved_model.pbtxt')
    if os.path.exists(pb_path) or os.path.exists(pbtxt_path):
      return 'savedmodel'
    return 'unknown'

  # Check file extension for HDF5/Keras formats
  if model_path.endswith('.h5') or model_path.endswith('.hdf5'):
    return 'h5'
  if model_path.endswith('.keras'):
    return 'keras'

  # Try to detect HDF5 format by file signature
  try:
    import h5py
    if h5py.is_hdf5(model_path):
      return 'h5'
  except ImportError:
    pass

  return 'unknown'


def load_model(model_path, compile_model=False):
  """Load a TensorFlow/Keras model from disk.

  Args:
    model_path: Path to the model file or directory.
    compile_model: Whether to compile the model after loading.

  Returns:
    A loaded Keras model.

  Raises:
    ModelLoadError: If the model cannot be loaded.
  """
  detected_format = detect_format(model_path)

  if detected_format == 'not_found':
    raise ModelLoadError(
        f"Error: Path does not exist: '{model_path}'\n"
        "Please provide a valid path to a SavedModel directory, "
        ".h5 file, or .keras file."
    )

  if detected_format == 'unknown':
    raise ModelLoadError(
        f"Error: Could not determine model format for: '{model_path}'\n"
        "Supported formats:\n"
        "  - SavedModel (directory with saved_model.pb)\n"
        "  - HDF5 (.h5, .hdf5 files)\n"
        "  - Keras (.keras files)"
    )

  logging.info(f"Detected format: {detected_format}")
  logging.info(f"Loading model from: {model_path}")

  try:
    model = keras_save.load_model(model_path, compile=compile_model)
    return model
  except Exception as e:
    error_msg = str(e)

    # Provide helpful context based on common errors
    if 'custom_objects' in error_msg.lower():
      raise ModelLoadError(
          f"Error: Failed to load model from '{model_path}'\n"
          f"The model contains custom layers or objects that could not be "
          f"deserialized.\n"
          f"Original error: {error_msg}"
      )
    elif 'No file or directory found' in error_msg:
      raise ModelLoadError(
          f"Error: Could not load model at '{model_path}'\n"
          f"The path exists but does not contain a valid model.\n"
          f"Original error: {error_msg}"
      )
    else:
      raise ModelLoadError(
          f"Error: Failed to load model from '{model_path}'\n"
          f"Original error: {error_msg}"
      )
