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
"""Creates TOCO options to process a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import traceback

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.lite.testing import zip_test_utils


def toco_options(data_types,
                 input_arrays,
                 output_arrays,
                 shapes,
                 extra_toco_options=None):
  """Create TOCO options to process a model.

  Args:
    data_types: input and inference types used by TOCO.
    input_arrays: names of the input tensors
    output_arrays: name of the output tensors
    shapes: shapes of the input tensors
    extra_toco_options: additional toco options

  Returns:
    the options in a string.
  """
  if extra_toco_options is None:
    extra_toco_options = zip_test_utils.ExtraTocoOptions()

  shape_str = ":".join([",".join(str(y) for y in x) for x in shapes if x])
  inference_type = "FLOAT"
  if data_types[0] == "QUANTIZED_UINT8":
    inference_type = "QUANTIZED_UINT8"
  s = (" --input_data_types=%s" % ",".join(data_types) +
       " --inference_type=%s" % inference_type +
       " --input_format=TENSORFLOW_GRAPHDEF" + " --output_format=TFLITE" +
       " --input_arrays=%s" % ",".join(input_arrays) +
       " --output_arrays=%s" % ",".join(output_arrays))
  if shape_str:
    s += (" --input_shapes=%s" % shape_str)
  if extra_toco_options.drop_control_dependency:
    s += " --drop_control_dependency"
  if extra_toco_options.allow_custom_ops:
    s += " --allow_custom_ops"
  if extra_toco_options.rnn_states:
    s += (" --rnn_states='" + extra_toco_options.rnn_states + "'")
  if extra_toco_options.split_tflite_lstm_inputs is not None:
    if extra_toco_options.split_tflite_lstm_inputs:
      s += " --split_tflite_lstm_inputs=true"
    else:
      s += " --split_tflite_lstm_inputs=false"
  return s


def toco_convert(options, graph_def, input_tensors, output_tensors, **kwargs):
  """Convert a model's graph def into a tflite model.

  NOTE: this currently shells out to the toco binary, but we would like
  convert to Python API tooling in the future.

  Args:
    options: An Options instance.
    graph_def: A GraphDef object.
    input_tensors: List of input tensor tuples `(name, shape, type)`.
    output_tensors: List of output tensors (names).
    **kwargs: Extra options to be passed.

  Returns:
    output tflite model, log_txt from conversion
    or None, log_txt if it did not convert properly.
  """
  # Convert ophint ops if presented.
  graph_def = tf.compat.v1.lite.experimental.convert_op_hints_to_stubs(
      graph_def=graph_def)
  graph_def_str = graph_def.SerializeToString()

  extra_toco_options = kwargs.get("extra_toco_options",
                                  zip_test_utils.ExtraTocoOptions())
  test_params = kwargs.get("test_params", {})
  input_arrays = [x[0] for x in input_tensors]
  data_types = [zip_test_utils.TF_TYPE_INFO[x[2]][1] for x in input_tensors]

  fully_quantize = test_params.get("fully_quantize", False)
  dynamic_range_quantize = test_params.get("dynamic_range_quantize", False)
  if dynamic_range_quantize or fully_quantize:
    with tempfile.NamedTemporaryFile() as graphdef_file:
      graphdef_file.write(graph_def_str)
      graphdef_file.flush()

      input_shapes = zip_test_utils.get_input_shapes_map(input_tensors)
      converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
          graphdef_file.name, input_arrays, output_tensors, input_shapes)

      converter.experimental_new_quantizer = options.mlir_quantizer
      converter.optimizations = [tf.lite.Optimize.DEFAULT]

      if fully_quantize:
        # Read the input range for the representative dataset from parameters.
        min_value, max_value = test_params.get("input_range", (-1, 1))

        def representative_dataset(input_tensors):
          calibration_inputs = []
          for _, shape, _ in input_tensors:
            if shape:
              dims = [dim.value for dim in shape.dims]
              calibration_inputs.append(
                  np.random.uniform(min_value, max_value,
                                    tuple(dims)).astype(np.float32))
          return calibration_inputs

        def representative_dataset_gen():
          for _ in range(100):
            yield representative_dataset(input_tensors)

        if test_params.get("quant_16x8", False):
          converter.target_spec.supported_ops = [
              tf.lite.OpsSet.\
              EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
          ]
        else:
          converter.target_spec.supported_ops = [
              tf.lite.OpsSet.TFLITE_BUILTINS_INT8
          ]

        converter.representative_dataset = representative_dataset_gen
        if extra_toco_options.inference_input_type:
          converter.inference_input_type = (
              extra_toco_options.inference_input_type)
        if extra_toco_options.inference_output_type:
          converter.inference_output_type = (
              extra_toco_options.inference_output_type)
        else:
          if test_params.get("quant_16x8", False):
            converter.inference_output_type = tf.int16
          else:
            converter.inference_output_type = tf.int8

      try:
        tflite_model = converter.convert()
        return tflite_model, ""
      except Exception as e:
        log = "{0}\n{1}".format(str(e), traceback.format_exc())
        return None, log

  else:
    opts = toco_options(
        data_types=data_types,
        input_arrays=input_arrays,
        shapes=[x[1] for x in input_tensors],
        output_arrays=output_tensors,
        extra_toco_options=extra_toco_options)

    with tempfile.NamedTemporaryFile() as graphdef_file, \
         tempfile.NamedTemporaryFile() as output_file, \
         tempfile.NamedTemporaryFile("w+") as stdout_file:
      graphdef_file.write(graph_def_str)
      graphdef_file.flush()

      if options.run_with_flex:
        opts += " --enable_select_tf_ops --force_select_tf_ops"
      cmd = ("%s --input_file=%s --output_file=%s %s > %s 2>&1" %
             (options.toco, graphdef_file.name, output_file.name, opts,
              stdout_file.name))
      exit_code = os.system(cmd)
      log = (
          cmd + "exited with code %d" % exit_code + "\n------------------\n" +
          stdout_file.read())
      return (None if exit_code != 0 else output_file.read()), log
