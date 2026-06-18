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
"""This tool analyzes a TensorFlow Lite graph."""

import os

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("tflite_runtime", "analyzer")):
  # This file is part of tensorflow package.
  from tensorflow.compiler.mlir.lite.python import wrap_converter
  from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper
  from tensorflow.python.util.tf_export import tf_export as _tf_export
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import _pywrap_analyzer_wrapper as _analyzer_wrapper

  def _tf_export(*x, **kwargs):
    del x, kwargs
    return lambda x: x


@_tf_export("lite.experimental.Analyzer")
class ModelAnalyzer():
  """Provides a collection of TFLite model analyzer tools.

  Example:

  ```python
  model = tf.keras.applications.MobileNetV3Large()
  fb_model = tf.lite.TFLiteConverterV2.from_keras_model(model).convert()
  tf.lite.experimental.Analyzer.analyze(model_content=fb_model)
  # === TFLite ModelAnalyzer ===
  #
  # Your TFLite model has ‘1’ subgraph(s). In the subgraph description below,
  # T# represents the Tensor numbers. For example, in Subgraph#0, the MUL op
  # takes tensor #0 and tensor #19 as input and produces tensor #136 as output.
  #
  # Subgraph#0 main(T#0) -> [T#263]
  #   Op#0 MUL(T#0, T#19) -> [T#136]
  #   Op#1 ADD(T#136, T#18) -> [T#137]
  #   Op#2 CONV_2D(T#137, T#44, T#93) -> [T#138]
  #   Op#3 HARD_SWISH(T#138) -> [T#139]
  #   Op#4 DEPTHWISE_CONV_2D(T#139, T#94, T#24) -> [T#140]
  #   ...
  ```

  WARNING: Experimental interface, subject to change.
  """

  @staticmethod
  def analyze(model_path=None,
              model_content=None,
              gpu_compatibility=False,
              **kwargs):
    """Analyzes the given tflite_model with dumping model structure.

    This tool provides a way to understand users' TFLite flatbuffer model by
    dumping internal graph structure. It also provides additional features
    like checking GPU delegate compatibility.

    WARNING: Experimental interface, subject to change.
             The output format is not guaranteed to stay stable, so don't
             write scripts to this.

    Args:
      model_path: TFLite flatbuffer model path.
      model_content: TFLite flatbuffer model object.
      gpu_compatibility: Whether to check GPU delegate compatibility.
      **kwargs: Experimental keyword arguments to analyze API.

    Returns:
      Print analyzed report via console output.
    """
    if not model_path and not model_content:
      raise ValueError("neither `model_path` nor `model_content` is provided")
    if model_path:
      print(f"=== {model_path} ===\n")
      tflite_model = model_path
      input_is_filepath = True
    else:
      print("=== TFLite ModelAnalyzer ===\n")
      tflite_model = model_content
      input_is_filepath = False

    if kwargs.get("experimental_use_mlir", False):
      print(
          wrap_converter.wrapped_flat_buffer_file_to_mlir(
              tflite_model, input_is_filepath
          )
      )
    else:
      print(
          _analyzer_wrapper.ModelAnalyzer(tflite_model, input_is_filepath,
                                          gpu_compatibility))
