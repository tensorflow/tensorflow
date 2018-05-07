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
"""Python TF-Lite interpreter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.lazy_loader import LazyLoader

# Lazy load since some of the performance benchmark skylark rules
# break dependencies. Must use double quotes to match code internal rewrite
# rule.
# pylint: disable=g-inconsistent-quotes
_interpreter_wrapper = LazyLoader(
    "_interpreter_wrapper", globals(),
    "tensorflow.contrib.lite.python.interpreter_wrapper."
    "tensorflow_wrap_interpreter_wrapper")
# pylint: enable=g-inconsistent-quotes

del LazyLoader


class Interpreter(object):
  """Interpreter inferace for TF-Lite Models."""

  def __init__(self, model_path=None, model_content=None):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
    if model_path and not model_content:
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromFile(
              model_path))
      if not self._interpreter:
        raise ValueError('Failed to open {}'.format(model_path))
    elif model_content and not model_path:
      self._interpreter = (
          _interpreter_wrapper.InterpreterWrapper_CreateWrapperCPPFromBuffer(
              model_content, len(model_content)))
      if not self._interpreter:
        raise ValueError(
            'Failed to create model from {} bytes'.format(len(model_content)))
    elif not model_path and not model_path:
      raise ValueError('`model_path` or `model_content` must be specified.')
    else:
      raise ValueError('Can\'t both provide `model_path` and `model_content`')

  def allocate_tensors(self):
    if not self._interpreter.AllocateTensors():
      raise ValueError('Failed to allocate tensors')

  def _get_tensor_details(self, tensor_index):
    """Gets tensor details.

    Args:
      tensor_index: Tensor index of tensor to query.

    Returns:
      a dictionary containing the name, index, shape and type of the tensor.

    Raises:
      ValueError: If tensor_index is invalid.
    """
    tensor_index = int(tensor_index)
    tensor_name = self._interpreter.TensorName(tensor_index)
    tensor_size = self._interpreter.TensorSize(tensor_index)
    tensor_type = self._interpreter.TensorType(tensor_index)
    tensor_quantization = self._interpreter.TensorQuantization(tensor_index)

    if not tensor_name or not tensor_type:
      raise ValueError('Could not get tensor details')

    details = {
        'name': tensor_name,
        'index': tensor_index,
        'shape': tensor_size,
        'dtype': tensor_type,
        'quantization': tensor_quantization,
    }

    return details

  def get_input_details(self):
    """Gets model input details.

    Returns:
      A list of input details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.InputIndices()
    ]

  def set_tensor(self, tensor_index, value):
    """Sets the value of the input.

    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
                    the 'index' field in get_input_details.
      value: Value of tensor to set.

    Raises:
      ValueError: If the interpreter could not set the tensor.
    """
    if not self._interpreter.SetTensor(tensor_index, value):
      raise ValueError('Failed to set tensor')

  def resize_tensor_input(self, input_index, tensor_size):
    """Resizes an input tensor.

    Args:
      input_index: Tensor index of input to set. This value can be gotten from
                   the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.

    Raises:
      ValueError: If the interpreter could not resize the input tensor.
    """
    if not self._interpreter.ResizeInputTensor(input_index, tensor_size):
      raise ValueError('Failed to resize input')

  def get_output_details(self):
    """Gets model output details.

    Returns:
      A list of output details.
    """
    return [
        self._get_tensor_details(i) for i in self._interpreter.OutputIndices()
    ]

  def get_tensor(self, tensor_index):
    """Sets the value of the input.

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
                    the 'index' field in get_output_details.

    Returns:
      a numpy array.
    """
    return self._interpreter.GetTensor(tensor_index)

  def invoke(self):
    if not self._interpreter.Invoke():
      raise ValueError('Failed to invoke TFLite model')
