# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Tensorflow Lite quantization."""

import enum
import functools

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.lite.python import lite as _lite
from tensorflow.lite.testing.model_coverage import model_coverage_lib as model_coverage
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


FLAGS = flags.FLAGS
flags.DEFINE_boolean("update_goldens", False, "Update stored golden files.")

# Random number generator seed and number of steps to use for calibration
REPRESENTATIVE_DATA_SEED = 0
NUM_CALIBRATION_STEPS = 100


@enum.unique
class QuantizationType(enum.Enum):
  FLOAT16 = "float16"
  FULL_INTEGER = "int"
  FULL_INTEGER_16X8 = "int16x8"

ALL_QUANTIZATION_TYPES = list(QuantizationType)


def parameterize_by_quantization(*quantization_types):
  """Decorates a test to parameterize it by a list of quantization type.

  Example:
    @parameterize_by_quantization(QuantizationType.FLOAT16, ..)
    def test_mytest(self, quantization_type):
      ..

  Args:
    *quantization_types: The list of QuantizationType to parameterize with.
  Returns:
    A test parameterized by the passed quantization types.
  """
  def decorator(to_be_wrapped):
    @parameterized.named_parameters(
        (quant_type.value, quant_type) for quant_type in quantization_types)
    def wrapper(*args, **kwargs):
      return to_be_wrapped(*args, **kwargs)

    return wrapper

  return decorator


def representative_dataset_gen(shape, dtype):
  """Generates a pseudo random representtive dataset.

  The random number generator is seeded with the same value before each call.
  Args:
    shape: Input shape of the model.
    dtype: Type (numpy.dtype) of the data to generate.
  Yields:
    Arrays of data to be used as representative dataset.
  """
  np.random.seed(REPRESENTATIVE_DATA_SEED)
  for _ in range(NUM_CALIBRATION_STEPS):
    data = np.random.rand(*(shape[1:])).astype(dtype)
    yield [data]


class ModelQuantizationTest(parameterized.TestCase):

  def _test_quantization_goldens(self, quantization_type, converter, input_data,
                                 golden_name):
    converter.experimental_new_quantizer = True
    converter.optimizations = [_lite.Optimize.DEFAULT]

    if quantization_type == QuantizationType.FLOAT16:
      converter.target_spec.supported_types = [dtypes.float16]
    elif quantization_type in (QuantizationType.FULL_INTEGER,
                               QuantizationType.FULL_INTEGER_16X8):
      converter.representative_dataset = functools.partial(
          representative_dataset_gen,
          shape=np.shape(input_data),
          dtype=np.float32)

      # QuantizationType.FULL_INTEGER (int8 quantization with float fallback):
      # keep target_spec.supported_ops as default

      if quantization_type == QuantizationType.FULL_INTEGER_16X8:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet
            .EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]

    tflite_model = converter.convert()
    model_coverage.compare_model_golden(tflite_model, input_data, golden_name,
                                        FLAGS.update_goldens)

  @parameterize_by_quantization(*ALL_QUANTIZATION_TYPES)
  def test_mobilenet_v1(self, quantization_type):
    frozengraph_file = model_coverage.get_filepath(
        "mobilenet/mobilenet_v1_1.0_224_frozen.pb")
    converter = _lite.TFLiteConverter.from_frozen_graph(
        frozengraph_file,
        input_arrays=["input"],
        output_arrays=["MobilenetV1/Predictions/Reshape_1"],
        input_shapes={"input": (1, 224, 224, 3)})
    img_array = keras.applications.inception_v3.preprocess_input(
        model_coverage.get_image(224))

    self._test_quantization_goldens(
        quantization_type,
        converter,
        input_data=[img_array],
        golden_name="mobilenet_v1_%s" % quantization_type.value)

  @parameterize_by_quantization(*ALL_QUANTIZATION_TYPES)
  def test_mobilenet_v2(self, quantization_type):
    saved_model_dir = model_coverage.get_filepath(
        "keras_applications/mobilenet_v2_tf2")
    converter = _lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
    img_array = keras.applications.inception_v3.preprocess_input(
        model_coverage.get_image(224))

    self._test_quantization_goldens(
        quantization_type,
        converter,
        input_data=[img_array],
        golden_name="mobilenet_v2_%s" % quantization_type.value)


if __name__ == "__main__":
  test.main()
