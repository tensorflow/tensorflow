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
"""Utils for make_zip tests."""
import functools
import io
import itertools
import operator
import os
import re
import string
import sys
import tempfile
import traceback
import zipfile

import ml_dtypes
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.lite.testing import _pywrap_string_util
from tensorflow.lite.testing import generate_examples_report as report_lib
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import signature_constants

# pylint: disable=g-import-not-at-top

# A map from names to functions which make test cases.
_MAKE_TEST_FUNCTIONS_MAP = {}


# A decorator to register the make test functions.
# Usage:
# All the make_*_test should be registered. Example:
#   @register_make_test_function()
#   def make_conv_tests(options):
#     # ...
# If a function is decorated by other decorators, it's required to specify the
# name explicitly. Example:
#   @register_make_test_function(name="make_unidirectional_sequence_lstm_tests")
#   @test_util.enable_control_flow_v2
#   def make_unidirectional_sequence_lstm_tests(options):
#     # ...
def register_make_test_function(name=None):

  def decorate(function, name=name):
    if name is None:
      name = function.__name__
    _MAKE_TEST_FUNCTIONS_MAP[name] = function

  return decorate


def get_test_function(test_function_name):
  """Get the test function according to the test function name."""

  if test_function_name not in _MAKE_TEST_FUNCTIONS_MAP:
    return None
  return _MAKE_TEST_FUNCTIONS_MAP[test_function_name]


RANDOM_SEED = 342

MAP_TF_TO_NUMPY_TYPE = {
    tf.float32: np.float32,
    tf.float16: np.float16,
    tf.float64: np.float64,
    tf.complex64: np.complex64,
    tf.complex128: np.complex128,
    tf.int32: np.int32,
    tf.uint32: np.uint32,
    tf.uint8: np.uint8,
    tf.int8: np.int8,
    tf.uint16: np.uint16,
    tf.int16: np.int16,
    tf.int64: np.int64,
    tf.bool: np.bool_,
    tf.string: np.bytes_,
}


class ExtraConvertOptions:
  """Additional options for conversion, besides input, output, shape."""

  def __init__(self):
    # Whether to ignore control dependency nodes.
    self.drop_control_dependency = False
    # Allow custom ops in the conversion.
    self.allow_custom_ops = False
    # Rnn states that are used to support rnn / lstm cells.
    self.rnn_states = None
    # Split the LSTM inputs from 5 inputs to 18 inputs for TFLite.
    self.split_tflite_lstm_inputs = None
    # The inference input type passed to TFLiteConvert.
    self.inference_input_type = None
    # The inference output type passed to TFLiteConvert.
    self.inference_output_type = None


def create_tensor_data(dtype, shape, min_value=-100, max_value=100):
  """Build tensor data spreading the range [min_value, max_value)."""

  if dtype in MAP_TF_TO_NUMPY_TYPE:
    dtype = MAP_TF_TO_NUMPY_TYPE[dtype]

  if dtype in (tf.float32, tf.float16, tf.float64):
    value = (max_value - min_value) * np.random.random_sample(shape) + min_value
  elif dtype in (tf.complex64, tf.complex128):
    real = (max_value - min_value) * np.random.random_sample(shape) + min_value
    imag = (max_value - min_value) * np.random.random_sample(shape) + min_value
    value = real + imag * 1j
  elif dtype in (tf.uint32, tf.int32, tf.uint8, tf.int8, tf.int64, tf.uint16,
                 tf.int16):
    value = np.random.randint(min_value, max_value + 1, shape)
  elif dtype == tf.bool:
    value = np.random.choice([True, False], size=shape)
  elif dtype == np.bytes_:
    # Not the best strings, but they will do for some basic testing.
    letters = list(string.ascii_uppercase)
    return np.random.choice(letters, size=shape).astype(dtype)
  elif dtype == tf.bfloat16:
    value = (max_value - min_value) * np.random.random_sample(shape) + min_value
    # There is no bfloat16 type in numpy. Uses ml_dtypes.bfloat16 for Eigen.
    dtype = ml_dtypes.bfloat16
  else:
    raise ValueError("Unsupported dtype: %s" % dtype)
  return np.dtype(dtype).type(value) if np.isscalar(value) else value.astype(
      dtype)


def create_scalar_data(dtype, min_value=-100, max_value=100):
  """Build scalar tensor data range from min_value to max_value exclusively."""

  if dtype in MAP_TF_TO_NUMPY_TYPE:
    dtype = MAP_TF_TO_NUMPY_TYPE[dtype]

  if dtype in (tf.float32, tf.float16, tf.float64):
    value = (max_value - min_value) * np.random.random() + min_value
  elif dtype in (tf.int32, tf.uint8, tf.int64, tf.int16):
    value = np.random.randint(min_value, max_value + 1)
  elif dtype == tf.bool:
    value = np.random.choice([True, False])
  elif dtype == np.bytes_:
    l = np.random.randint(1, 6)
    value = "".join(np.random.choice(list(string.ascii_uppercase), size=l))
  elif dtype == tf.bfloat16:
    value = (max_value - min_value) * np.random.random() + min_value
    # There is no bfloat16 type in numpy. Uses ml_dtypes.bfloat16 for Eigen.
    dtype = ml_dtypes.bfloat16
  else:
    raise ValueError("Unsupported dtype: %s" % dtype)
  return np.array(value, dtype=dtype)


def freeze_graph(session, outputs):
  """Freeze the current graph.

  Args:
    session: Tensorflow sessions containing the graph
    outputs: List of output tensors

  Returns:
    The frozen graph_def.
  """
  return convert_to_constants.convert_variables_to_constants(
      session, session.graph.as_graph_def(), [x.op.name for x in outputs])


def format_result(t):
  """Convert a tensor to a format that can be used in test specs."""
  if t.dtype.kind not in [np.dtype(np.bytes_).kind, np.dtype(np.object_).kind]:
    # Output 9 digits after the point to ensure the precision is good enough.
    # bfloat16 promotes the value to string, not float. so we need to
    # convert it to float explicitly.
    if t.dtype == ml_dtypes.bfloat16:
      values = ["{:.9f}".format(float(value)) for value in list(t.flatten())]
    else:
      values = ["{:.9f}".format(value) for value in list(t.flatten())]
    return ",".join(values)
  else:
    # SerializeAsHexString returns bytes in PY3, so decode if appropriate.
    return _pywrap_string_util.SerializeAsHexString(t.flatten()).decode("utf-8")


def write_examples(fp, examples):
  """Given a list `examples`, write a text format representation.

  The file format is csv like with a simple repeated pattern. We would ike
  to use proto here, but we can't yet due to interfacing with the Android
  team using this format.

  Args:
    fp: File-like object to write to.
    examples: Example dictionary consisting of keys "inputs" and "outputs"
  """

  def write_tensor(fp, name, x):
    """Write tensor in file format supported by TFLITE example."""
    fp.write("name,%s\n" % name)
    fp.write("dtype,%s\n" % x.dtype)
    fp.write("shape," + ",".join(map(str, x.shape)) + "\n")
    fp.write("values," + format_result(x) + "\n")

  fp.write("test_cases,%d\n" % len(examples))
  for example in examples:
    fp.write("inputs,%d\n" % len(example["inputs"]))
    for name, value in example["inputs"].items():
      if value is not None:
        write_tensor(fp, name, value)
    fp.write("outputs,%d\n" % len(example["outputs"]))
    for name, value in example["outputs"].items():
      write_tensor(fp, name, value)


class TextFormatWriter:
  """Utility class for writing ProtoBuf like messages."""

  def __init__(self, fp, name=None, parent=None):
    self.fp = fp
    self.indent = parent.indent if parent else 0
    self.name = name

  def __enter__(self):
    if self.name:
      self.write(self.name + " {")
      self.indent += 2
    return self

  def __exit__(self, *exc_info):
    if self.name:
      self.indent -= 2
      self.write("}")
    return True

  def write(self, data):
    self.fp.write(" " * self.indent + data + "\n")

  def write_field(self, key, val):
    self.write(key + ": \"" + val + "\"")

  def sub_message(self, name):
    return TextFormatWriter(self.fp, name, self)


def write_test_cases(fp, model_name, examples):
  """Given a dictionary of `examples`, write a text format representation.

  The file format is protocol-buffer-like, even though we don't use proto due
  to the needs of the Android team.

  Args:
    fp: File-like object to write to.
    model_name: Filename where the model was written to, relative to filename.
    examples: Example dictionary consisting of keys "inputs" and "outputs"

  Raises:
    RuntimeError: Example dictionary does not have input / output names.
  """

  writer = TextFormatWriter(fp)
  writer.write_field("load_model", os.path.basename(model_name))
  for example in examples:
    inputs = []
    for name in example["inputs"].keys():
      if name:
        inputs.append(name)
    outputs = []
    for name in example["outputs"].keys():
      if name:
        outputs.append(name)
    if not (inputs and outputs):
      raise RuntimeError("Empty input / output names.")

    # Reshape message
    with writer.sub_message("reshape") as reshape:
      for name, value in example["inputs"].items():
        with reshape.sub_message("input") as input_msg:
          input_msg.write_field("key", name)
          input_msg.write_field("value", ",".join(map(str, value.shape)))

    # Invoke message
    with writer.sub_message("invoke") as invoke:
      for name, value in example["inputs"].items():
        with invoke.sub_message("input") as input_msg:
          input_msg.write_field("key", name)
          input_msg.write_field("value", format_result(value))
      # Expectations
      for name, value in example["outputs"].items():
        with invoke.sub_message("output") as output_msg:
          output_msg.write_field("key", name)
          output_msg.write_field("value", format_result(value))
        with invoke.sub_message("output_shape") as output_shape:
          output_shape.write_field("key", name)
          output_shape.write_field("value",
                                   ",".join([str(dim) for dim in value.shape]))


def get_input_shapes_map(input_tensors):
  """Gets a map of input names to shapes.

  Args:
    input_tensors: List of input tensor tuples `(name, shape, type)`.

  Returns:
    {string : list of integers}.
  """
  input_arrays = [tensor[0] for tensor in input_tensors]
  input_shapes_list = []

  for _, shape, _ in input_tensors:
    dims = None
    if shape:
      dims = [dim.value for dim in shape.dims]
    input_shapes_list.append(dims)

  input_shapes = {
      name: shape
      for name, shape in zip(input_arrays, input_shapes_list)
      if shape
  }
  return input_shapes


def _normalize_input_name(input_name):
  """Remove :i suffix from input tensor names."""
  return input_name.split(":")[0]


def _normalize_output_name(output_name):
  """Remove :0 suffix from output tensor names."""
  return output_name.split(":")[0] if output_name.endswith(
      ":0") else output_name


def _get_tensor_info(tensors, default_name_prefix, normalize_func):
  """Get the list of tensor name and info."""
  tensor_names = []
  tensor_info_map = {}
  for idx, tensor in enumerate(tensors):
    if not tensor.name:
      tensor.name = default_name_prefix + str(idx)
    tensor_info = tf.compat.v1.saved_model.utils.build_tensor_info(tensor)
    tensor_name = normalize_func(tensor.name)
    tensor_info_map[tensor_name] = tensor_info
    tensor_names.append(tensor_name)
  return tensor_names, tensor_info_map


# How many test cases we may have in a zip file. Too many test cases will
# slow down the test data generation process.
_MAX_TESTS_PER_ZIP = 500


def make_zip_of_tests(options,
                      test_parameters,
                      make_graph,
                      make_test_inputs,
                      extra_convert_options=ExtraConvertOptions(),
                      use_frozen_graph=False,
                      expected_tf_failures=0):
  """Helper to make a zip file of a bunch of TensorFlow models.

  This does a cartesian product of the dictionary of test_parameters and
  calls make_graph() for each item in the cartesian product set.
  If the graph is built successfully, then make_test_inputs() is called to
  build expected input/output value pairs. The model is then converted to
  tflite, and the examples are serialized with the tflite model into a zip
  file (2 files per item in the cartesian product set).

  Args:
    options: An Options instance.
    test_parameters: Dictionary mapping to lists for each parameter.
      e.g. `{"strides": [[1,3,3,1], [1,2,2,1]], "foo": [1.2, 1.3]}`
    make_graph: function that takes current parameters and returns tuple
      `[input1, input2, ...], [output1, output2, ...]`
    make_test_inputs: function taking `curr_params`, `session`, `input_tensors`,
      `output_tensors` and returns tuple `(input_values, output_values)`.
    extra_convert_options: Additional convert options.
    use_frozen_graph: Whether or not freeze graph before convertion.
    expected_tf_failures: Number of times tensorflow is expected to fail in
      executing the input graphs. In some cases it is OK for TensorFlow to fail
      because the one or more combination of parameters is invalid.

  Raises:
    RuntimeError: if there are converter errors that can't be ignored.
  """
  zip_path = os.path.join(options.output_path, options.zip_to_output)
  parameter_count = 0
  for parameters in test_parameters:
    parameter_count += functools.reduce(
        operator.mul, [len(values) for values in parameters.values()])

  all_parameter_count = parameter_count
  if options.multi_gen_state:
    all_parameter_count += options.multi_gen_state.parameter_count
  if not options.no_tests_limit and all_parameter_count > _MAX_TESTS_PER_ZIP:
    raise RuntimeError(
        "Too many parameter combinations for generating '%s'.\n"
        "There are at least %d combinations while the upper limit is %d.\n"
        "Having too many combinations will slow down the tests.\n"
        "Please consider splitting the test into multiple functions.\n" %
        (zip_path, all_parameter_count, _MAX_TESTS_PER_ZIP))
  if options.multi_gen_state:
    options.multi_gen_state.parameter_count = all_parameter_count

  # TODO(aselle): Make this allow multiple inputs outputs.
  if options.multi_gen_state:
    archive = options.multi_gen_state.archive
  else:
    archive = zipfile.PyZipFile(zip_path, "w")
  zip_manifest = []
  convert_report = []
  converter_errors = 0

  processed_labels = set()

  if options.make_tf_ptq_tests:
    # For cases with fully_quantize is True, also generates a case with
    # fully_quantize is False. Marks these cases as suitable for PTQ tests.
    parameter_count = 0
    for parameters in test_parameters:
      if True in parameters.get("fully_quantize", []):
        parameters.update({"fully_quantize": [True, False], "tf_ptq": [True]})
        # TODO(b/199054047): Support 16x8 quantization in TF Quantization.
        parameters.update({"quant_16x8": [False]})
        parameter_count += functools.reduce(
            operator.mul, [len(values) for values in parameters.values()])

  if options.make_edgetpu_tests:
    extra_convert_options.inference_input_type = tf.uint8
    extra_convert_options.inference_output_type = tf.uint8
    # Only count parameters when fully_quantize is True.
    parameter_count = 0
    for parameters in test_parameters:
      if True in parameters.get("fully_quantize",
                                []) and False in parameters.get(
                                    "quant_16x8", [False]):
        parameter_count += functools.reduce(operator.mul, [
            len(values)
            for key, values in parameters.items()
            if key != "fully_quantize" and key != "quant_16x8"
        ])

  label_base_path = zip_path
  if options.multi_gen_state:
    label_base_path = options.multi_gen_state.label_base_path

  i = 1
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      label = label_base_path.replace(".zip", "_") + (",".join(
          "%s=%r" % z for z in sorted(zip(keys, curr))).replace(" ", ""))
      if label[0] == "/":
        label = label[1:]

      zip_path_label = label
      if len(os.path.basename(zip_path_label)) > 245:
        zip_path_label = label_base_path.replace(".zip", "_") + str(i)

      i += 1
      if label in processed_labels:
        # Do not populate data for the same label more than once. It will cause
        # errors when unzipping.
        continue
      processed_labels.add(label)

      param_dict = dict(zip(keys, curr))

      if options.make_tf_ptq_tests and not param_dict.get("tf_ptq", False):
        continue

      if options.make_edgetpu_tests and (not param_dict.get(
          "fully_quantize", False) or param_dict.get("quant_16x8", False)):
        continue

      def generate_inputs_outputs(tflite_model_binary,
                                  min_value=0,
                                  max_value=255):
        """Generate input values and output values of the given tflite model.

        Args:
          tflite_model_binary: A serialized flatbuffer as a string.
          min_value: min value for the input tensor.
          max_value: max value for the input tensor.

        Returns:
          (input_values, output_values): Maps of input values and output values
          built.
        """
        interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_values = {}
        for input_detail in input_details:
          input_value = create_tensor_data(
              input_detail["dtype"],
              input_detail["shape"],
              min_value=min_value,
              max_value=max_value)
          interpreter.set_tensor(input_detail["index"], input_value)
          input_values.update(
              {_normalize_input_name(input_detail["name"]): input_value})

        interpreter.invoke()

        output_details = interpreter.get_output_details()
        output_values = {}
        for output_detail in output_details:
          output_values.update({
              _normalize_output_name(output_detail["name"]):
                  interpreter.get_tensor(output_detail["index"])
          })

        return input_values, output_values

      def build_example(label, param_dict_real, zip_path_label):
        """Build the model with parameter values set in param_dict_real.

        Args:
          label: Label of the model
          param_dict_real: Parameter dictionary (arguments to the factories
            make_graph and make_test_inputs)
          zip_path_label: Filename in the zip

        Returns:
          (tflite_model_binary, report) where tflite_model_binary is the
          serialized flatbuffer as a string and report is a dictionary with
          keys `tflite_converter_log` (log of conversion), `tf_log` (log of tf
          conversion), `converter` (a string of success status of the
          conversion), `tf` (a string success status of the conversion).
        """

        np.random.seed(RANDOM_SEED)
        report = {
            "tflite_converter": report_lib.NOTRUN,
            "tf": report_lib.FAILED
        }

        # Build graph
        report["tf_log"] = ""
        report["tflite_converter_log"] = ""
        tf.compat.v1.reset_default_graph()

        with tf.Graph().as_default():
          with tf.device("/cpu:0"):
            try:
              inputs, outputs = make_graph(param_dict_real)
              inputs = [x for x in inputs if x is not None]
            except (tf.errors.UnimplementedError,
                    tf.errors.InvalidArgumentError, ValueError):
              report["tf_log"] += traceback.format_exc()
              return None, report

          sess = tf.compat.v1.Session()
          try:
            baseline_inputs, baseline_outputs = (
                make_test_inputs(param_dict_real, sess, inputs, outputs))
            baseline_inputs = [x for x in baseline_inputs if x is not None]
            # Converts baseline inputs/outputs to maps. The signature input and
            # output names are set to be the same as the tensor names.
            input_names = [_normalize_input_name(x.name) for x in inputs]
            output_names = [_normalize_output_name(x.name) for x in outputs]
            baseline_input_map = dict(zip(input_names, baseline_inputs))
            baseline_output_map = dict(zip(output_names, baseline_outputs))
          except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError,
                  ValueError):
            report["tf_log"] += traceback.format_exc()
            return None, report
          report["tflite_converter"] = report_lib.FAILED
          report["tf"] = report_lib.SUCCESS

          # Builds a saved model with the default signature key.
          input_names, tensor_info_inputs = _get_tensor_info(
              inputs, "input_", _normalize_input_name)
          output_tensors, tensor_info_outputs = _get_tensor_info(
              outputs, "output_", _normalize_output_name)
          input_tensors = [
              (name, t.shape, t.dtype) for name, t in zip(input_names, inputs)
          ]

          inference_signature = (
              tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                  inputs=tensor_info_inputs,
                  outputs=tensor_info_outputs,
                  method_name="op_test"))
          saved_model_dir = tempfile.mkdtemp("op_test")
          saved_model_tags = [tf.saved_model.SERVING]
          signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
          builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
              saved_model_dir)
          builder.add_meta_graph_and_variables(
              sess,
              saved_model_tags,
              signature_def_map={
                  signature_key: inference_signature,
              },
              strip_default_attrs=True)
          builder.save(as_text=False)
          # pylint: disable=g-long-ternary
          graph_def = freeze_graph(
              sess,
              tf.compat.v1.global_variables() + inputs +
              outputs) if use_frozen_graph else sess.graph_def

        if "split_tflite_lstm_inputs" in param_dict_real:
          extra_convert_options.split_tflite_lstm_inputs = param_dict_real[
              "split_tflite_lstm_inputs"]
        tflite_model_binary, converter_log = options.tflite_convert_function(
            options,
            saved_model_dir,
            input_tensors,
            output_tensors,
            extra_convert_options=extra_convert_options,
            test_params=param_dict_real)
        report["tflite_converter"] = (
            report_lib.SUCCESS
            if tflite_model_binary is not None else report_lib.FAILED)
        report["tflite_converter_log"] = converter_log

        if options.save_graphdefs:
          zipinfo = zipfile.ZipInfo(zip_path_label + ".pbtxt")
          archive.writestr(zipinfo, text_format.MessageToString(graph_def),
                           zipfile.ZIP_DEFLATED)

        if tflite_model_binary:
          if options.make_edgetpu_tests:
            # Set proper min max values according to input dtype.
            baseline_input_map, baseline_output_map = generate_inputs_outputs(
                tflite_model_binary, min_value=0, max_value=255)
          zipinfo = zipfile.ZipInfo(zip_path_label + ".bin")
          if sys.byteorder == "big":
            tflite_model_binary = flatbuffer_utils.byte_swap_tflite_buffer(
                tflite_model_binary, "big", "little"
            )
          archive.writestr(zipinfo, tflite_model_binary, zipfile.ZIP_DEFLATED)

          example = {
              "inputs": baseline_input_map,
              "outputs": baseline_output_map
          }

          example_fp = io.StringIO()
          write_examples(example_fp, [example])
          zipinfo = zipfile.ZipInfo(zip_path_label + ".inputs")
          archive.writestr(zipinfo, example_fp.getvalue(), zipfile.ZIP_DEFLATED)

          example_fp2 = io.StringIO()
          write_test_cases(example_fp2, zip_path_label + ".bin", [example])
          zipinfo = zipfile.ZipInfo(zip_path_label + "_tests.txt")
          archive.writestr(zipinfo, example_fp2.getvalue(),
                           zipfile.ZIP_DEFLATED)

          zip_manifest_label = zip_path_label + " " + label
          if zip_path_label == label:
            zip_manifest_label = zip_path_label

          zip_manifest.append(zip_manifest_label + "\n")

        return tflite_model_binary, report

      _, report = build_example(label, param_dict, zip_path_label)

      if report["tflite_converter"] == report_lib.FAILED:
        ignore_error = False
        if not options.known_bugs_are_errors:
          for pattern, bug_number in options.known_bugs.items():
            if re.search(pattern, label):
              print("Ignored converter error due to bug %s" % bug_number)
              ignore_error = True
        if not ignore_error:
          converter_errors += 1
          print("-----------------\nconverter error!\n%s\n-----------------\n" %
                report["tflite_converter_log"])

      convert_report.append((param_dict, report))

  if not options.no_conversion_report:
    report_io = io.StringIO()
    report_lib.make_report_table(report_io, zip_path, convert_report)
    if options.multi_gen_state:
      zipinfo = zipfile.ZipInfo("report_" + options.multi_gen_state.test_name +
                                ".html")
      archive.writestr(zipinfo, report_io.getvalue())
    else:
      zipinfo = zipfile.ZipInfo("report.html")
      archive.writestr(zipinfo, report_io.getvalue())

  if options.multi_gen_state:
    options.multi_gen_state.zip_manifest.extend(zip_manifest)
  else:
    zipinfo = zipfile.ZipInfo("manifest.txt")
    archive.writestr(zipinfo, "".join(zip_manifest), zipfile.ZIP_DEFLATED)

  # Log statistics of what succeeded
  total_conversions = len(convert_report)
  tf_success = sum(
      1 for x in convert_report if x[1]["tf"] == report_lib.SUCCESS)
  converter_success = sum(1 for x in convert_report
                          if x[1]["tflite_converter"] == report_lib.SUCCESS)
  percent = 0
  if tf_success > 0:
    percent = float(converter_success) / float(tf_success) * 100.
  tf.compat.v1.logging.info(
      ("Archive %s Considered %d graphs, %d TF evaluated graphs "
       " and %d converted graphs (%.1f%%"), zip_path, total_conversions,
      tf_success, converter_success, percent)

  tf_failures = parameter_count - tf_success

  if tf_failures / parameter_count > 0.8:
    raise RuntimeError(("Test for '%s' is not very useful. "
                        "TensorFlow fails in %d percent of the cases.") %
                       (zip_path, int(100 * tf_failures / parameter_count)))

  if tf_failures != expected_tf_failures and not (options.make_edgetpu_tests or
                                                  options.make_tf_ptq_tests):
    raise RuntimeError(("Expected TF to fail %d times while generating '%s', "
                        "but that happened %d times") %
                       (expected_tf_failures, zip_path, tf_failures))

  if not options.ignore_converter_errors and converter_errors > 0:
    raise RuntimeError("Found %d errors while generating models" %
                       converter_errors)
