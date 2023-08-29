# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrapper for post training quantization with calibration."""
import numpy as np

from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework import dtypes
from tensorflow.python.util.lazy_loader import LazyLoader

# Lazy load since some of the performance benchmark skylark rules
# break dependencies. Must use double quotes to match code internal rewrite
# rule.
_calibration_wrapper = LazyLoader(
    "_calibration_wrapper",
    globals(),
    (
        "tensorflow.lite.python.optimize."
        "_pywrap_tensorflow_lite_calibration_wrapper"
    ),
)


def add_intermediate_tensors(model_content):
  """Adds intermediate tensors to fused op if needed."""
  return _calibration_wrapper.AddIntermediateTensors(model_content)


class Calibrator:
  """Calibrates a floating point model and then quantizes it.

  This is an internal class, not a public interface.
  """

  def __init__(
      self,
      model_content,
      custom_op_registerers_by_name=None,
      custom_op_registerers_by_func=None,
  ):
    """Constructor.

    Args:
      model_content: Content of a TF-Lite Flatbuffer file.
      custom_op_registerers_by_name: List of str (symbol names) that take a
        pointer to a MutableOpResolver and register custom ops.
      custom_op_registerers_by_func: List of functions that take a pointer to a
        MutableOpResolver and register custom ops.

    Raises:
      ValueError: If the calibrator was unable to open the model.
    """
    if not model_content:
      raise ValueError("`model_content` must be specified.")
    if custom_op_registerers_by_name is None:
      custom_op_registerers_by_name = []
    if custom_op_registerers_by_func is None:
      custom_op_registerers_by_func = []
    try:
      self._calibrator = _calibration_wrapper.CalibrationWrapper(
          model_content,
          custom_op_registerers_by_name,
          custom_op_registerers_by_func,
      )
      self._model_content = model_content
    except Exception as e:
      raise ValueError("Failed to parse the model: %s." % e)
    if not self._calibrator:
      raise ValueError("Failed to parse the model.")
    self._interpreter = None

  def _create_input_array_from_dict(self, signature_key, inputs):
    input_array = []
    signature_runner = self._interpreter.get_signature_runner(signature_key)
    input_details = sorted(
        signature_runner.get_input_details().items(),
        key=lambda item: item[1]["index"],
    )
    for input_name, _ in input_details:
      input_array.append(inputs[input_name])
    return input_array

  def _feed_tensors(self, dataset_gen, resize_input):
    """Feed tensors to the calibrator."""
    initialized = {}

    for sample in dataset_gen():
      if isinstance(sample, tuple):
        if not isinstance(sample[1], dict):
          raise ValueError(
              "You need to provide either a dictionary with input "
              "names and values in the second arugment in the "
              "tuple"
          )
        # Convert signature based inputs to the tensor index based data.
        if self._interpreter is None:
          self._interpreter = Interpreter(model_content=self._model_content)
        signature_key = sample[0]
        input_array = self._create_input_array_from_dict(
            signature_key, sample[1]
        )
      elif isinstance(sample, dict):
        # Convert signature based inputs to the tensor index based data.
        if self._interpreter is None:
          self._interpreter = Interpreter(model_content=self._model_content)
        signature_key = None
        input_array = self._create_input_array_from_dict(None, sample)
      elif isinstance(sample, list):
        signature_key = None
        input_array = sample
      else:
        raise ValueError(
            "You need to provide either a dictionary with input "
            "names and values, a tuple with signature key and a "
            "dictionary with input names and values, or an array "
            "with input values in the order of input tensors of "
            "the graph in the representative_dataset function. "
            "Unsupported value from dataset: {}.".format(sample)
        )

      if signature_key not in initialized:
        initialized[signature_key] = True
        if resize_input:
          if signature_key is not None:
            self._calibrator.Prepare(
                [list(s.shape) for s in input_array], signature_key
            )
          else:
            self._calibrator.Prepare([list(s.shape) for s in input_array])
        else:
          if signature_key is not None:
            self._calibrator.Prepare(signature_key)
          else:
            self._calibrator.Prepare()
      if signature_key is not None:
        self._calibrator.FeedTensor(input_array, signature_key)
      else:
        self._calibrator.FeedTensor(input_array)

  @convert_phase(
      Component.OPTIMIZE_TFLITE_MODEL,
      SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER,
  )
  def calibrate_and_quantize(
      self,
      dataset_gen,
      input_type,
      output_type,
      allow_float,
      activations_type=dtypes.int8,
      bias_type=dtypes.int32,
      resize_input=True,
      disable_per_channel=False,
  ):
    """Calibrates the model with specified generator and then quantizes it.

    The input shapes of the calibrator are resized with the calibration data if
    `resize_input` is set.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      activations_type: A tf.dtype representing the desired type for
        activations.
      bias_type: A tf.dtype representing the desired type for bias.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
      disable_per_channel: A boolean. True if disabling per-channel
        quantization.
    """
    self._feed_tensors(dataset_gen, resize_input)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
        allow_float,
        np.dtype(activations_type.as_numpy_dtype()).num,
        np.dtype(bias_type.as_numpy_dtype()).num,
        disable_per_channel,
    )

  @convert_phase(
      Component.OPTIMIZE_TFLITE_MODEL,
      SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER,
  )
  def calibrate_and_quantize_single(
      self,
      dataset_gen,
      input_type,
      output_type,
      allow_float,
      op_output_name,
      resize_input=True,
  ):
    """Calibrates the model with specified generator and then quantizes it.

    Only the single op with output op_output_name will be quantized.
    The input shapes of the calibrator are resized with the calibration data.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      op_output_name: A string, only this op will be quantized.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
    """
    self._feed_tensors(dataset_gen, resize_input)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
        allow_float,
        op_output_name,
    )

  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.CALIBRATE)
  def calibrate(self, dataset_gen):
    """Calibrates the model with specified generator.

    Returns:
      A model with min and max calibration stats.

    Args:
      dataset_gen: A generator that generates calibration samples.
    """
    self._feed_tensors(dataset_gen, resize_input=True)
    return self._calibrator.Calibrate()
