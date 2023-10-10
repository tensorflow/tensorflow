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
"""Extending CheckpointReader for TensorFlow."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export


def error_translator(e):
  """Translate the tensor_slice_reader.cc errors."""
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  error_message = str(e)
  if 'not found in checkpoint' in error_message or (
      'Failed to find any '
      'matching files for') in error_message:
    raise errors_impl.NotFoundError(None, None, error_message)
  elif 'Sliced checkpoints are not supported' in error_message or (
      'Data type '
      'not '
      'supported') in error_message:
    raise errors_impl.UnimplementedError(None, None, error_message)
  elif 'Failed to get matching files on' in error_message:
    raise errors_impl.InvalidArgumentError(None, None, error_message)
  elif 'Unable to open table file' in error_message:
    raise errors_impl.DataLossError(None, None, error_message)
  elif 'Failed to find the saved tensor slices' in error_message or (
      'not convertible to numpy dtype' in error_message):
    raise errors_impl.InternalError(None, None, error_message)
  else:
    raise errors_impl.OpError(None, None, error_message, errors_impl.UNKNOWN)


def get_variable_to_dtype_map(self):
  return {
      name: dtypes.DType(type_enum)
      for name, type_enum in self._GetVariableToDataTypeMap().items()  # pylint: disable=protected-access
  }

CheckpointReader.get_variable_to_dtype_map = get_variable_to_dtype_map


def has_tensor(self, tensor_str):
  return self._HasTensor(compat.as_bytes(tensor_str))  # pylint: disable=protected-access

CheckpointReader.has_tensor = has_tensor


def get_tensor(self, tensor_str):
  """Get the tensor from the Checkpoint object."""
  try:
    return CheckpointReader.CheckpointReader_GetTensor(
        self, compat.as_bytes(tensor_str))
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  except RuntimeError as e:
    error_translator(e)


CheckpointReader.get_tensor = get_tensor


# Disable invalid name to keep backwards compatibility with that function.
# It was previously exported from py_checkpoint_reader.i which did not conform
# to pylint checks.
# pylint: disable=invalid-name
@tf_export(v1=['train.NewCheckpointReader'])
def NewCheckpointReader(filepattern):
  """A function that returns a CheckPointReader.

  Args:
    filepattern: The filename.

  Returns:
    A CheckpointReader object.
  """
  try:
    return CheckpointReader(compat.as_bytes(filepattern))
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  except RuntimeError as e:
    error_translator(e)
