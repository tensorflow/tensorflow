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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.lite.python import lite_constants

# Lazy load since some of the performance benchmark skylark rules
# break dependencies. Must use double quotes to match code internal rewrite
# rule.
_calibration_wrapper = LazyLoader(
    "_calibration_wrapper", globals(),
    "tensorflow.lite.python.optimize."
    "_pywrap_tensorflow_lite_calibration_wrapper")


class Calibrator(object):
  """Calibrates a floating point model and then quantizes it.

  This is an internal class, not a public interface.
  """

  def __init__(self, model_content):
    """Constructor.

    Args:
      model_content: Content of a TF-Lite Flatbuffer file.

    Raises:
      ValueError: If the calibrator was unable to open the model.
    """
    if not model_content:
      raise ValueError("`model_content` must be specified.")
    try:
      self._calibrator = (
          _calibration_wrapper.CalibrationWrapper(model_content))
    except Exception as e:
      raise ValueError("Failed to parse the model: %s." % e)
    if not self._calibrator:
      raise ValueError("Failed to parse the model.")

  def calibrate_and_quantize(self,
                             dataset_gen,
                             input_type,
                             output_type,
                             allow_float,
                             activations_type=lite_constants.INT8,
                             resize_input=True):
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
                   computation, useful when targeting an integer-only backend.
                   If False, an error will be thrown if an operation cannot be
                   quantized, otherwise the model will fallback to float ops.
      activations_type: A tf.dtype representing the desired type for
                   activations.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
    """
    initialized = False
    for sample in dataset_gen():
      if not initialized:
        initialized = True
        if resize_input:
          self._calibrator.Prepare([list(s.shape) for s in sample])
        else:
          self._calibrator.Prepare()
      self._calibrator.FeedTensor(sample)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num, allow_float,
        np.dtype(activations_type.as_numpy_dtype()).num)

  def calibrate_and_quantize_single(self,
                                    dataset_gen,
                                    input_type,
                                    output_type,
                                    allow_float,
                                    op_output_name,
                                    resize_input=True):
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
    initialized = False
    for sample in dataset_gen():
      if not initialized:
        initialized = True
        if resize_input:
          self._calibrator.Prepare([list(s.shape) for s in sample])
        else:
          self._calibrator.Prepare()
      self._calibrator.FeedTensor(sample)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num, allow_float, op_output_name)

  def calibrate(self, dataset_gen):
    """Calibrates the model with specified generator.

    Returns:
      A model with min and max calibration stats.

    Args:
      dataset_gen: A generator that generates calibration samples.
    """
    initialized = False
    for sample in dataset_gen():
      if not initialized:
        initialized = True
        self._calibrator.Prepare([list(s.shape) for s in sample])
      self._calibrator.FeedTensor(sample)
    return self._calibrator.Calibrate()
