# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Generate a series of TensorFlow graphs that become tflite test cases.

Usage:

generate_examples <output directory>

bazel run //tensorflow/lite/testing:generate_examples

To more easily debug failures use (or override) the --save_graphdefs flag to
place text proto graphdefs into the generated zip files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import operator
import os
import random
import re
import sys
import tempfile
import traceback
import zipfile
import numpy as np
from six import StringIO
from six.moves import xrange

# TODO(aselle): Disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# pylint: disable=g-import-not-at-top
import tensorflow as tf
from google.protobuf import text_format
# TODO(aselle): switch to TensorFlow's resource_loader
from tensorflow.lite.testing import generate_examples_report as report_lib
from tensorflow.python.framework import graph_util as tf_graph_util
from tensorflow.python.ops import rnn

parser = argparse.ArgumentParser(description="Script to generate TFLite tests.")
parser.add_argument("output_path",
                    help="Directory where the outputs will be go.")
parser.add_argument(
    "--zip_to_output",
    type=str,
    help="Particular zip to output.",
    required=True)
parser.add_argument("--toco",
                    type=str,
                    help="Path to toco tool.",
                    required=True)
parser.add_argument(
    "--known_bugs_are_errors",
    action="store_true",
    help=("If a particular model is affected by a known bug,"
          " count it as a toco error."))
parser.add_argument(
    "--ignore_toco_errors",
    action="store_true",
    help="Raise an exception if any toco error is encountered.")
parser.add_argument(
    "--save_graphdefs",
    action="store_true",
    help="Include intermediate graphdefs in the output zip files.")
parser.add_argument(
    "--run_with_flex",
    action="store_true",
    help="Whether the TFLite Flex converter is being used.")

RANDOM_SEED = 342
TEST_INPUT_DEPTH = 3


# A map from regular expression to bug number. Any test failure with label
# matching the expression will be considered due to the corresponding bug.
KNOWN_BUGS = {
    # TOCO doesn't support scalars as input.
    # Concat doesn't work with a single input tensor
    r"concat.*num_tensors=1": "67378344",
    # Transposition in MatMul is not fully supported.
    "fully_connected.*transpose_a=True": "67586970",
    # Softmax graphs are too complex.
    r"softmax.*dim=0": "67749831",
    # BatchToSpaceND only supports 4D tensors.
    r"batch_to_space_nd.*input_shape=\[8,2,2,2,1,1\]": "70594733",
    # Div will use floordiv.
    r"div.*int32": "72051395",
}


class ExtraTocoOptions(object):
  """Additional toco options besides input, output, shape."""

  def __init__(self):
    # Whether to ignore control dependency nodes.
    self.drop_control_dependency = False
    # Allow custom ops in the toco conversion.
    self.allow_custom_ops = False
    # Rnn states that are used to support rnn / lstm cells.
    self.rnn_states = None
    # Split the LSTM inputs from 5 inoputs to 18 inputs for TFLite.
    self.split_tflite_lstm_inputs = None


def toco_options(data_types,
                 input_arrays,
                 output_arrays,
                 shapes,
                 extra_toco_options=ExtraTocoOptions()):
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
  shape_str = ":".join([",".join(str(y) for y in x) for x in shapes if x])
  inference_type = "FLOAT"
  # TODO(ahentz): if we get multi-input quantization to work we need this
  # to change
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


def write_examples(fp, examples):
  """Given a list `examples`, write a text format representation.

  The file format is csv like with a simple repeated pattern. We would ike
  to use proto here, but we can't yet due to interfacing with the Android
  team using this format.

  Args:
    fp: File-like object to write to.
    examples: Example dictionary consiting of keys "inputs" and "outputs"
  """

  def write_tensor(fp, x):
    """Write tensor in file format supported by TFLITE example."""
    fp.write("dtype,%s\n" % x.dtype)
    fp.write("shape," + ",".join(map(str, x.shape)) + "\n")
    # Output 9 digits after the point to ensure the precision is good enough.
    values = ["{:.9f}".format(value) for value in list(x.flatten())]
    fp.write("values," + ",".join(values) + "\n")

  fp.write("test_cases,%d\n" % len(examples))
  for example in examples:
    fp.write("inputs,%d\n" % len(example["inputs"]))
    for i in example["inputs"]:
      write_tensor(fp, i)
    fp.write("outputs,%d\n" % len(example["outputs"]))
    for i in example["outputs"]:
      write_tensor(fp, i)


def write_test_cases(fp, model_name, examples):
  """Given a dictionary of `examples`, write a text format representation.

  The file format is protocol-buffer-like, even though we don't use proto due
  to the needs of the Android team.

  Args:
    fp: File-like object to write to.
    model_name: Filename where the model was written to, relative to filename.
    examples: Example dictionary consiting of keys "inputs" and "outputs"
  """

  fp.write("load_model: %s\n" % os.path.basename(model_name))
  for example in examples:
    fp.write("reshape {\n")
    for t in example["inputs"]:
      fp.write("  input: \"" + ",".join(map(str, t.shape)) + "\"\n")
    fp.write("}\n")
    fp.write("invoke {\n")

    for t in example["inputs"]:
      values = ["{:.9f}".format(value) for value in list(t.flatten())]
      fp.write("  input: \"" + ",".join(values) + "\"\n")
    for t in example["outputs"]:
      values = ["{:.9f}".format(value) for value in list(t.flatten())]
      fp.write("  output: \"" + ",".join(values) + "\"\n")
    fp.write("}\n")


_TF_TYPE_INFO = {
    tf.float32: (np.float32, "FLOAT"),
    tf.float16: (np.float16, "FLOAT"),
    tf.int32: (np.int32, "INT32"),
    tf.uint8: (np.uint8, "QUANTIZED_UINT8"),
    tf.int16: (np.int16, "QUANTIZED_INT16"),
    tf.int64: (np.int64, "INT64"),
    tf.bool: (np.bool, "BOOL"),
}


def create_tensor_data(dtype, shape, min_value=-100, max_value=100):
  """Build tensor data spreading the range [min_value, max_value)."""

  if dtype in _TF_TYPE_INFO:
    dtype = _TF_TYPE_INFO[dtype][0]

  if dtype in (tf.float32, tf.float16):
    value = (max_value-min_value)*np.random.random_sample(shape)+min_value
  elif dtype in (tf.int32, tf.uint8, tf.int64, tf.int16):
    value = np.random.randint(min_value, max_value+1, shape)
  elif dtype == tf.bool:
    value = np.random.choice([True, False], size=shape)
  return np.dtype(dtype).type(value) if np.isscalar(value) else value.astype(
      dtype)


def create_scalar_data(dtype, min_value=-100, max_value=100):
  """Build scalar tensor data range from min_value to max_value exclusively."""

  if dtype in _TF_TYPE_INFO:
    dtype = _TF_TYPE_INFO[dtype][0]

  if dtype in (tf.float32, tf.float16):
    value = (max_value - min_value) * np.random.random() + min_value
  elif dtype in (tf.int32, tf.uint8, tf.int64, tf.int16):
    value = np.random.randint(min_value, max_value + 1)
  return np.array(value, dtype=dtype)


def freeze_graph(session, outputs):
  """Freeze the current graph.

  Args:
    session: Tensorflow sessions containing the graph
    outputs: List of output tensors

  Returns:
    The frozen graph_def.
  """
  return tf_graph_util.convert_variables_to_constants(
      session, session.graph.as_graph_def(), [x.op.name for x in outputs])


def make_control_dep_tests(zip_path):
  """Make a set of tests that use control dependencies."""

  test_parameters = [{
      "input_shape": [[], [1, 1, 1, 1], [1, 15, 14, 1], [3, 15, 14, 3]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    filter_value = tf.zeros((3, 3, TEST_INPUT_DEPTH, 8), tf.float32)
    assert_op = tf.assert_greater_equal(input_tensor, input_tensor - 1)
    with tf.control_dependencies([assert_op]):
      out = tf.nn.conv2d(input_tensor, filter_value,
                         strides=(1, 1, 1, 1), padding="SAME")
      return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(tf.float32, parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.drop_control_dependency = True
  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs,
                    extra_toco_options)


def toco_convert(graph_def_str, input_tensors, output_tensors,
                 extra_toco_options):
  """Convert a model's graph def into a tflite model.

  NOTE: this currently shells out to the toco binary, but we would like
  convert to Python API tooling in the future.

  Args:
    graph_def_str: Graph def proto in serialized string format.
    input_tensors: List of input tensor tuples `(name, shape, type)`.
    output_tensors: List of output tensors (names).
    extra_toco_options: Additional toco options.

  Returns:
    output tflite model, log_txt from conversion
    or None, log_txt if it did not convert properly.
  """
  input_arrays = [x[0] for x in input_tensors]
  data_types = [_TF_TYPE_INFO[x[2]][1] for x in input_tensors]
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

    # TODO(aselle): Switch this to subprocess at some point.
    if "pb2lite" in bin_path and FLAGS.run_with_flex:
      opts = ("--input_arrays={0} --output_arrays={1}".format(
          ",".join(input_arrays), ",".join(output_tensors)))
    elif FLAGS.run_with_flex:
      opts += " --enable_select_tf_ops --force_select_tf_ops"
    cmd = ("%s --input_file=%s --output_file=%s %s > %s 2>&1" %
           (bin_path, graphdef_file.name, output_file.name, opts,
            stdout_file.name))
    exit_code = os.system(cmd)
    log = (
        cmd + "exited with code %d" % exit_code + "\n------------------\n" +
        stdout_file.read())
    return (None if exit_code != 0 else output_file.read()), log


def normalize_output_name(output_name):
  """Remove :0 suffix from tensor names."""
  return output_name.split(":")[0] if output_name.endswith(
      ":0") else output_name


# How many test cases we may have in a zip file. Too many test cases will
# slow down the test data generation process.
_MAX_TESTS_PER_ZIP = 500


def make_zip_of_tests(zip_path,
                      test_parameters,
                      make_graph,
                      make_test_inputs,
                      extra_toco_options=ExtraTocoOptions(),
                      use_frozen_graph=False,
                      expected_tf_success=None):
  """Helper to make a zip file of a bunch of TensorFlow models.

  This does a cartestian product of the dictionary of test_parameters and
  calls make_graph() for each item in the cartestian product set.
  If the graph is built successfully, then make_test_inputs() is called to
  build expected input/output value pairs. The model is then converted to tflite
  with toco, and the examples are serialized with the tflite model into a zip
  file (2 files per item in the cartesian product set).

  Args:
    zip_path: Path of zip file to write
    test_parameters: Dictionary mapping to lists for each parameter.
      e.g. `{"strides": [[1,3,3,1], [1,2,2,1]], "foo": [1.2, 1.3]}`
    make_graph: function that takes current parameters and returns tuple
      `[input1, input2, ...], [output1, output2, ...]`
    make_test_inputs: function taking `curr_params`, `session`, `input_tensors`,
      `output_tensors` and returns tuple `(input_values, output_values)`.
    extra_toco_options: Additional toco options.
    use_frozen_graph: Whether or not freeze graph before toco converter.
    expected_tf_success: Number of times tensorflow is supposed to succeed in
      executing the input graphs. `None` means "unknown".

  Raises:
    RuntimeError: if there are toco errors that can't be ignored.
  """
  parameter_count = 0
  for parameters in test_parameters:
    parameter_count += functools.reduce(
        operator.mul, [len(values) for values in parameters.values()])

  if parameter_count > _MAX_TESTS_PER_ZIP:
    raise RuntimeError(
        "Too many parameter combinations for generating '%s'.\n"
        "There are %d combinations while the upper limit is %d.\n"
        "Having too many combinations will slow down the tests.\n"
        "Please consider splitting the test into multiple functions.\n"
        % (zip_path, parameter_count, _MAX_TESTS_PER_ZIP))

  # TODO(aselle): Make this allow multiple inputs outputs.
  archive = zipfile.PyZipFile(zip_path, "w")
  zip_manifest = []
  convert_report = []
  toco_errors = 0

  processed_labels = set()
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      label = zip_path.replace(".zip", "_") + (",".join(
          "%s=%r" % z for z in sorted(zip(keys, curr))).replace(" ", ""))
      if label[0] == "/":
        label = label[1:]
      if label in processed_labels:
        # Do not populate data for the same label more than once. It will cause
        # errors when unzipping.
        continue
      processed_labels.add(label)

      param_dict = dict(zip(keys, curr))

      def build_example(label, param_dict_real):
        """Build the model with parameter values set in param_dict_real.

        Args:
          label: Label of the model (i.e. the filename in the zip).
          param_dict_real: Parameter dictionary (arguments to the factories
            make_graph and make_test_inputs)
        Returns:
          (tflite_model_binary, report) where tflite_model_binary is the
          serialized flatbuffer as a string and report is a dictionary with
          keys `toco_log` (log of toco conversion), `tf_log` (log of tf
          conversion), `toco` (a string of success status of the conversion),
          `tf` (a string success status of the conversion).
        """

        np.random.seed(RANDOM_SEED)
        report = {"toco": report_lib.NOTRUN, "tf": report_lib.FAILED}

        # Build graph
        report["tf_log"] = ""
        report["toco_log"] = ""
        tf.reset_default_graph()

        with tf.device("/cpu:0"):
          try:
            inputs, outputs = make_graph(param_dict_real)
          except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError,
                  ValueError):
            report["tf_log"] += traceback.format_exc()
            return None, report

        sess = tf.Session()
        try:
          baseline_inputs, baseline_outputs = (make_test_inputs(
              param_dict_real, sess, inputs, outputs))
        except (tf.errors.UnimplementedError, tf.errors.InvalidArgumentError,
                ValueError):
          report["tf_log"] += traceback.format_exc()
          return None, report
        report["toco"] = report_lib.FAILED
        report["tf"] = report_lib.SUCCESS
        # Convert graph to toco
        input_tensors = [(input_tensor.name.split(":")[0],
                          input_tensor.get_shape(), input_tensor.dtype)
                         for input_tensor in inputs]
        output_tensors = [normalize_output_name(out.name) for out in outputs]
        graph_def = freeze_graph(
            sess,
            tf.global_variables() + inputs +
            outputs) if use_frozen_graph else sess.graph_def

        if "split_tflite_lstm_inputs" in param_dict_real:
          extra_toco_options.split_tflite_lstm_inputs = param_dict_real[
              "split_tflite_lstm_inputs"]

        tflite_model_binary, toco_log = toco_convert(
            graph_def.SerializeToString(), input_tensors, output_tensors,
            extra_toco_options)
        report["toco"] = (report_lib.SUCCESS if tflite_model_binary is not None
                          else report_lib.FAILED)
        report["toco_log"] = toco_log

        if True or FLAGS.save_graphdefs:
          archive.writestr(label + ".pbtxt",
                           text_format.MessageToString(graph_def),
                           zipfile.ZIP_DEFLATED)

        if tflite_model_binary:
          archive.writestr(label + ".bin", tflite_model_binary,
                           zipfile.ZIP_DEFLATED)
          example = {"inputs": baseline_inputs, "outputs": baseline_outputs}

          example_fp = StringIO()
          write_examples(example_fp, [example])
          archive.writestr(label + ".inputs",
                           example_fp.getvalue(), zipfile.ZIP_DEFLATED)

          example_fp2 = StringIO()
          write_test_cases(example_fp2, label + ".bin", [example])
          archive.writestr(label + "_tests.txt",
                           example_fp2.getvalue(), zipfile.ZIP_DEFLATED)

          zip_manifest.append(label + "\n")

        return tflite_model_binary, report

      _, report = build_example(label, param_dict)

      if report["toco"] == report_lib.FAILED:
        ignore_error = False
        if not FLAGS.known_bugs_are_errors:
          for pattern, bug_number in KNOWN_BUGS.items():
            if re.search(pattern, label):
              print("Ignored TOCO error due to bug %s" % bug_number)
              ignore_error = True
        if not ignore_error:
          toco_errors += 1
          print("-----------------\ntoco error!\n%s\n-----------------\n" %
                report["toco_log"])

      convert_report.append((param_dict, report))

  report_io = StringIO()
  report_lib.make_report_table(report_io, zip_path, convert_report)
  archive.writestr("report.html", report_io.getvalue())

  archive.writestr("manifest.txt", "".join(zip_manifest), zipfile.ZIP_DEFLATED)

  # Log statistics of what succeeded
  total_conversions = len(convert_report)
  tf_success = sum(1 for x in convert_report
                   if x[1]["tf"] == report_lib.SUCCESS)
  toco_success = sum(1 for x in convert_report
                     if x[1]["toco"] == report_lib.SUCCESS)
  percent = 0
  if tf_success > 0:
    percent = float(toco_success) / float(tf_success) * 100.
  tf.logging.info(("Archive %s Considered %d graphs, %d TF evaluated graphs "
                   " and %d TOCO converted graphs (%.1f%%"), zip_path,
                  total_conversions, tf_success, toco_success, percent)

  if expected_tf_success is not None and tf_success != expected_tf_success:
    raise RuntimeError(
        "Expected TF to succeed %d times, but that happened %d times" %
        (expected_tf_success, tf_success))

  if not FLAGS.ignore_toco_errors and toco_errors > 0:
    raise RuntimeError(
        "Found %d errors while generating toco models" % toco_errors)


def make_pool_tests(pool_op_in):
  """Make a set of tests to do average pooling.

  Args:
    pool_op_in: TensorFlow pooling operation to test  i.e. `tf.nn.avg_pool`.

  Returns:
    A function representing the true generator (after curried pool_op_in).
  """

  pool_op = pool_op_in

  def f(zip_path):
    """Actual function that generates examples.

    Args:
      zip_path: path to write zip to.
    """

    # Chose a set of parameters
    test_parameters = [{
        "ksize": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1], [1, 10, 11, 1]],
        "strides": [[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 2, 1], [1, 10, 11, 1]],
        # TODO(aselle): should add in a degenerate shape (e.g. [1, 0, 1, 1]).
        "input_shape": [[], [1, 1, 1, 1], [1, 15, 14, 1], [3, 15, 14, 3]],
        "padding": ["SAME", "VALID"],
        "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
    }]

    def build_graph(parameters):
      input_tensor = tf.placeholder(
          dtype=tf.float32, name="input", shape=parameters["input_shape"])
      out = pool_op(
          input_tensor,
          ksize=parameters["ksize"],
          strides=parameters["strides"],
          data_format=parameters["data_format"],
          padding=parameters["padding"])
      return [input_tensor], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      input_values = create_tensor_data(tf.float32, parameters["input_shape"])
      return [input_values], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_values])))

    make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)
  return f


def make_l2_pool_tests(zip_path):
  make_pool_tests(make_l2_pool)(zip_path)


def make_avg_pool_tests(zip_path):
  make_pool_tests(tf.nn.avg_pool)(zip_path)


def make_max_pool_tests(zip_path):
  make_pool_tests(tf.nn.max_pool)(zip_path)


def make_abs_tests(zip_path):
  """Make a set of tests to do relu."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.abs(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-10, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_relu_tests(zip_path):
  """Make a set of tests to do relu."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.relu(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-4, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_relu1_tests(zip_path):
  """Make a set of tests to do relu1."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3],
                      [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    # Note that the following is not supported:
    #   out = tf.maximum(-1.0, tf.minimum(input_tensor, 1.0))
    out = tf.minimum(1.0, tf.maximum(input_tensor, -1.0))
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-3, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_relu6_tests(zip_path):
  """Make a set of tests to do relu6."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3],
                      [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.relu(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-3, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_prelu_tests(zip_path):
  """Make a set of tests to do PReLU."""

  test_parameters = [
      {
          # The canonical case for image processing is having a 4D `input`
          # (NHWC)and `shared_axes`=[1, 2], so the alpha parameter is per
          # channel.
          "input_shape": [[1, 10, 10, 3], [3, 3, 3, 3]],
          "shared_axes": [[1, 2], [1]],
      },
      {
          # 2D-3D example. Share the 2nd axis.
          "input_shape": [[20, 20], [20, 20, 20]],
          "shared_axes": [[1]],
      }
  ]

  def build_graph(parameters):
    """Build the graph for the test case."""

    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    prelu = tf.keras.layers.PReLU(shared_axes=parameters["shared_axes"])
    out = prelu(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build the inputs for the test case."""

    input_shape = parameters["input_shape"]
    input_values = create_tensor_data(
        np.float32, input_shape, min_value=-10, max_value=10)
    shared_axes = parameters["shared_axes"]

    alpha_shape = []
    for dim in range(1, len(input_shape)):
      alpha_shape.append(1 if dim in shared_axes else input_shape[dim])

    alpha_values = create_tensor_data(np.float32, alpha_shape)

    # There should be only 1 trainable variable tensor.
    variables = tf.all_variables()
    assert len(variables) == 1
    sess.run(variables[0].assign(alpha_values))

    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(
      zip_path,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)


def make_leaky_relu_tests(zip_path):
  """Make a set of tests to do LeakyRelu."""

  test_parameters = [
      {
          "input_shape": [[], [1], [5], [1, 10, 10, 3], [3, 3, 3, 3]],
          "alpha": [0.1, 1.0, 2.0, -0.1, -1.0, -2.0],
      },
  ]

  def build_graph(parameters):
    """Build the graph for the test case."""

    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.leaky_relu(input_tensor, alpha=parameters["alpha"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build the inputs for the test case."""
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-3, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


# This function tests various TensorFLow functions that generates Const op,
# including `tf.ones`, `tf.zeros` and random functions.
def make_constant_tests(zip_path):
  """Make a set of tests to do constant ops."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32],
      "input_shape": [[], [1], [2], [1, 1, 1, 1], [2, 2, 2, 2]],
      "constant_is_also_output": [True, False],
  }]

  def build_graph(parameters):
    dummy_input = tf.placeholder(
        dtype=parameters["dtype"],
        name="input1",
        shape=parameters["input_shape"])
    constant = tf.constant(
        create_tensor_data(parameters["dtype"], parameters["input_shape"]))
    out = [tf.maximum(dummy_input, constant)]
    if parameters["constant_is_also_output"]:
      out.append(constant)

    return [dummy_input], out

  def build_inputs(parameters, sess, inputs, outputs):
    dummy_input = np.zeros(
        parameters["input_shape"], dtype=_TF_TYPE_INFO[parameters["dtype"]][0])
    return [dummy_input], sess.run(outputs, feed_dict={inputs[0]: dummy_input})

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs,
                    expected_tf_success=20)


def make_binary_op_tests(zip_path, binary_operator):
  """Make a set of tests to do binary ops with and without broadcast."""

  test_parameters = [
      # Avoid creating all combinations to keep the test size small.
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [True],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[5]],
          "input_shape_2": [[5]],
          "activation": [False, True],
      },
      {
          "dtype": [tf.float32, tf.int32, tf.int64],
          "input_shape_1": [[1, 3, 4, 3]],
          "input_shape_2": [[3]],
          "activation": [True, False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape_1": [[3]],
          "input_shape_2": [[1, 3, 4, 3]],
          "activation": [True, False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[]],
          "input_shape_2": [[]],
          "activation": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape_1": [[0]],
          "input_shape_2": [[1]],
          "activation": [False],
      }
  ]

  def build_graph(parameters):
    """Builds the graph given the current parameters."""
    input1 = tf.placeholder(
        dtype=parameters["dtype"],
        name="input1",
        shape=parameters["input_shape_1"])
    input2 = tf.placeholder(
        dtype=parameters["dtype"],
        name="input2",
        shape=parameters["input_shape_2"])
    out = binary_operator(input1, input2)
    if parameters["activation"]:
      out = tf.nn.relu(out)
    return [input1, input2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Builds operand inputs for op."""
    input1 = create_tensor_data(parameters["dtype"],
                                parameters["input_shape_1"])
    input2 = create_tensor_data(parameters["dtype"],
                                parameters["input_shape_2"])
    return [input1, input2], sess.run(
        outputs, feed_dict={
            inputs[0]: input1,
            inputs[1]: input2
        })

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_reduce_tests(reduce_op,
                      min_value=-10,
                      max_value=10,
                      boolean_tensor_only=False):
  """Make a set of tests to do reduce operation.

  Args:
    reduce_op: TensorFlow reduce operation to test, i.e. `tf.reduce_mean`.
    min_value: min value for created tensor data.
    max_value: max value for created tensor data.
    boolean_tensor_only: If true, will only generate tensor with boolean value.

  Returns:
    a function representing the true generator with `reduce_op_in` curried.
  """

  def f(zip_path):
    """Actual function that generates examples."""

    test_parameters = [
        {
            "input_dtype": [tf.float32, tf.int32, tf.int64],
            "input_shape": [[3, 3, 2, 4]],
            "axis": [
                0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2], [1, 0], [2, 0],
                [2, 1], [2, 1, 0], [2, 0, 1], -1, -2, -3, [1, -1], [0, -1],
                [-1, 0], [-1, -2, -3], [0, 0, 0], [2, 2, 0], [1, 0, -3, -3]
            ],
            "const_axis": [True, False],
            "keepdims": [True, False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[1, 8, 8, 3]],
            "axis": [
                0, 1, 2, 3, [1, 2], [0, 3], [1, 2, 3], [0, 1, 2,
                                                        3], [3, 2, 1, 0],
                [3, 1, 0, 2], [2, 0], [3, 0], [3, 1], [1, 0], -1, -2, -3, -4,
                [0, -2], [2, 3, -1, 0], [3, 1, 2, -3], [3, -4], [2, 2, 2],
                [2, 2, 3], [-3, -3, -4], [-3, 2, 1]
            ],
            "const_axis": [True, False],
            "keepdims": [True, False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1, 8, 8, 3], [3, 2, 4]],
            "axis": [[]],  # shape is: [0]
            "const_axis": [False],
            "keepdims": [True, False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1, 8, 8, 3], [3, 2, 4]],
            "axis": [None],  # shape is: []
            "const_axis": [True],
            "keepdims": [True, False],
        }
    ]

    def build_graph(parameters):
      """Build the mean op testing graph."""
      dtype = parameters["input_dtype"]
      if boolean_tensor_only:
        dtype = tf.bool
      input_tensor = tf.placeholder(
          dtype=dtype, name="input", shape=parameters["input_shape"])

      # Get axis as either a placeholder or constants.
      if parameters["const_axis"]:
        axis = parameters["axis"]
        input_tensors = [input_tensor]
      else:
        if isinstance(parameters["axis"], list):
          shape = [len(parameters["axis"])]
        else:
          shape = []  # shape for None or integers.
        axis = tf.placeholder(dtype=tf.int32, name="axis", shape=shape)
        input_tensors = [input_tensor, axis]

      out = reduce_op(
          input_tensor, axis=axis, keepdims=parameters["keepdims"])
      return input_tensors, [out]

    def build_inputs(parameters, sess, inputs, outputs):
      dtype = parameters["input_dtype"]
      if boolean_tensor_only:
        dtype = tf.bool
      values = [
          create_tensor_data(
              dtype,
              parameters["input_shape"],
              min_value=min_value,
              max_value=max_value)
      ]
      if not parameters["const_axis"]:
        values.append(np.array(parameters["axis"]))
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

    make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)

  return f


def make_mean_tests(zip_path):
  """Make a set of tests to do mean."""
  return make_reduce_tests(tf.reduce_mean)(zip_path)


def make_sum_tests(zip_path):
  """Make a set of tests to do sum."""
  return make_reduce_tests(tf.reduce_sum)(zip_path)


def make_reduce_prod_tests(zip_path):
  """Make a set of tests to do prod."""
  # set min max value to be -2, 2 to avoid overflow.
  return make_reduce_tests(tf.reduce_prod, -2, 2)(zip_path)


def make_reduce_max_tests(zip_path):
  """Make a set of tests to do max."""
  return make_reduce_tests(tf.reduce_max)(zip_path)


def make_reduce_min_tests(zip_path):
  """Make a set of tests to do min."""
  return make_reduce_tests(tf.reduce_min)(zip_path)


def make_reduce_any_tests(zip_path):
  """Make a set of tests to do any."""
  return make_reduce_tests(tf.reduce_any, boolean_tensor_only=True)(zip_path)


def make_exp_tests(zip_path):
  """Make a set of tests to do exp."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
  }]

  def build_graph(parameters):
    """Build the exp op testing graph."""
    input_tensor = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.exp(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["input_dtype"], parameters["input_shape"],
                           min_value=-100, max_value=9)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_log_softmax_tests(zip_path):
  """Make a set of tests to do log_softmax."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[1, 100], [4, 2], [5, 224]],
  }]

  def build_graph(parameters):
    """Build the log_softmax op testing graph."""
    input_tensor = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.nn.log_softmax(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=-100,
            max_value=9)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_maximum_tests(zip_path):
  """Make a set of tests to do maximum."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape_1": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
      "input_shape_2": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
  }]

  def build_graph(parameters):
    """Build the maximum op testing graph."""
    input_tensor_1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input_1",
        shape=parameters["input_shape_1"])
    input_tensor_2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input_2",
        shape=parameters["input_shape_2"])

    out = tf.maximum(input_tensor_1, input_tensor_2)
    return [input_tensor_1, input_tensor_2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape_1"]),
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape_2"])
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_minimum_tests(zip_path):
  """Make a set of tests to do minimum."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape_1": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
      "input_shape_2": [[], [3], [1, 100], [4, 2, 3], [5, 224, 224, 3]],
  }]

  def build_graph(parameters):
    """Build the minimum op testing graph."""
    input_tensor_1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input_1",
        shape=parameters["input_shape_1"])
    input_tensor_2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input_2",
        shape=parameters["input_shape_2"])

    out = tf.minimum(input_tensor_1, input_tensor_2)
    return [input_tensor_1, input_tensor_2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape_1"]),
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape_2"])
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_binary_op_tests_func(binary_operator):
  """Return a function that does a test on a binary operator."""
  return lambda zip_path: make_binary_op_tests(zip_path, binary_operator)


def make_add_tests(zip_path):
  make_binary_op_tests(zip_path, tf.add)


def make_div_tests(zip_path):
  make_binary_op_tests(zip_path, tf.div)


def make_sub_tests(zip_path):
  make_binary_op_tests(zip_path, tf.subtract)


def make_mul_tests(zip_path):
  make_binary_op_tests(zip_path, tf.multiply)


def make_pow_tests(zip_path):
  make_binary_op_tests(zip_path, tf.pow)


def make_floor_div_tests(zip_path):
  make_binary_op_tests(zip_path, tf.floor_div)


def make_floor_mod_tests(zip_path):
  make_binary_op_tests(zip_path, tf.floormod)


def make_squared_difference_tests(zip_path):
  make_binary_op_tests(zip_path, tf.squared_difference)


def make_gather_tests(zip_path):
  """Make a set of tests to do gather."""

  test_parameters = [{
      # TODO(mgubin): add string tests when they are supported by Toco.
      # TODO(mgubin): add tests for Nd indices when they are supported by
      # TfLite.
      "params_dtype": [tf.float32, tf.int32, tf.int64],
      "params_shape": [[10], [1, 2, 20]],
      "indices_dtype": [tf.int32, tf.int64],
      "indices_shape": [[3], [5]],
      "axis": [-1, 0, 1],
  }]

  def build_graph(parameters):
    """Build the gather op testing graph."""
    params = tf.placeholder(
        dtype=parameters["params_dtype"],
        name="params",
        shape=parameters["params_shape"])
    indices = tf.placeholder(
        dtype=parameters["indices_dtype"],
        name="indices",
        shape=parameters["indices_shape"])
    axis = min(len(parameters["params_shape"]), parameters["axis"])
    out = tf.gather(params, indices, axis=axis)
    return [params, indices], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    params = create_tensor_data(parameters["params_dtype"],
                                parameters["params_shape"])
    indices = create_tensor_data(parameters["indices_dtype"],
                                 parameters["indices_shape"], 0,
                                 parameters["params_shape"][0] - 1)
    return [params, indices], sess.run(
        outputs, feed_dict=dict(zip(inputs, [params, indices])))

  # Note that TF can't execute with index=1 and params_shape=[10].
  make_zip_of_tests(
      zip_path,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_success=60)


def make_gather_with_constant_tests(zip_path):
  """Make a set of test which feed a constant to gather toco."""

  test_parameters = [{
      "input_shape": [[3]],
      "reference_shape": [[2]],
  }, {
      "input_shape": [[2, 3]],
      "reference_shape": [[2, 3]],
  }]

  def build_graph(parameters):
    """Build a graph where the inputs to Gather are constants."""
    reference = tf.placeholder(
        dtype=tf.int32, shape=parameters["reference_shape"])
    gather_input = tf.constant(
        create_tensor_data(tf.int32, parameters["input_shape"]))
    gather_indices = tf.constant([0, 1], tf.int32)
    out = tf.equal(reference, tf.gather(gather_input, gather_indices))
    return [reference], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    reference_values = np.zeros(parameters["reference_shape"], dtype=np.int32)
    return [reference_values], sess.run(
        outputs, feed_dict={inputs[0]: reference_values})

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs,
                    expected_tf_success=2)


def make_global_batch_norm_tests(zip_path):
  """Make a set of tests to do batch_norm_with_global_normalization."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 6, 2], [3, 4, 5, 4]],
      "epsilon": [0.1, 0.0001],
      "scale_after": [True, False],
  }]

  def build_graph(parameters):
    """Build the global batch norm testing graph."""
    input_shape = parameters["input_shape"]
    scale_shape = input_shape[3]

    scale = create_tensor_data(parameters["dtype"], scale_shape)
    offset = create_tensor_data(parameters["dtype"], scale_shape)
    mean = create_tensor_data(parameters["dtype"], scale_shape)
    variance = create_tensor_data(parameters["dtype"], scale_shape)

    x = create_tensor_data(parameters["dtype"], parameters["input_shape"])
    x_norm = tf.nn.batch_norm_with_global_normalization(
        x, mean, variance, scale, offset,
        parameters["epsilon"], parameters["scale_after"])

    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.add(input_tensor, x_norm)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_fused_batch_norm_tests(zip_path):
  """Make a set of tests to do fused_batch_norm."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 6, 2]],
      "epsilon": [0.001, 0.1],
  }]

  def build_graph(parameters):
    """Build the testing graph for fused batch normalization."""
    input_shape = parameters["input_shape"]
    scale_shape = input_shape[3]

    scale = create_tensor_data(parameters["dtype"], scale_shape)
    offset = create_tensor_data(parameters["dtype"], scale_shape)
    mean = create_tensor_data(parameters["dtype"], scale_shape)
    variance = create_tensor_data(parameters["dtype"], scale_shape)

    x = create_tensor_data(parameters["dtype"], parameters["input_shape"])
    [x_norm, _, _] = tf.nn.fused_batch_norm(
        x, scale, offset, mean, variance,
        parameters["epsilon"], data_format="NHWC", is_training=False)

    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.add(input_tensor, x_norm)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_conv_tests(zip_path):
  """Make a set of tests to do convolution."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 3], [4, 6, 6, 1]],
      "filter_shape": [[1, 1], [2, 3], [3, 3]],
      "strides": [[1, 1, 1, 1], [1, 2, 3, 1]],
      "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
      "padding": ["SAME", "VALID"],
      "data_format": ["NHWC"],  # TODO(aselle): NCHW  would be good
      "constant_filter": [True, False],
      "channel_multiplier": [1, 2],
  }]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    # Get filter input either as a placeholder or constants. Also get a list of
    # the input tensors that are represented as placeholders.
    if parameters["constant_filter"]:
      filter_input = create_tensor_data(np.float32, filter_shape)
      input_tensors = [input_tensor]
    else:
      filter_input = tf.placeholder(
          dtype=tf.float32, name="filter", shape=filter_shape)
      input_tensors = [input_tensor, filter_input]

    out = tf.nn.conv2d(
        input_tensor,
        filter_input,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [create_tensor_data(np.float32, input_shape)]
    if not parameters["constant_filter"]:
      values.append(create_tensor_data(np.float32, filter_shape))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


# Note: This is a regression test for a bug (b/112436267) that Toco incorrectly
# fuses weights when multiple Conv2D/FULLY_CONNECTED ops share the same constant
# weight tensor.
def make_conv_with_shared_weights_tests(zip_path):
  """Make a test where 2 Conv ops shared the same constant weight tensor."""

  test_parameters = [{
      "input_shape": [[1, 10, 10, 3]],
      "filter_shape": [[3, 3]],
      "strides": [[1, 1, 1, 1]],
      "dilations": [[1, 1, 1, 1]],
      "padding": ["SAME"],
      "data_format": ["NHWC"],
      "channel_multiplier": [1],
  }]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)
    input_tensors = [input_tensor]

    # Construct a constant weights tensor which will be used by both Conv2D.
    filter_tensor = tf.constant(
        create_tensor_data(np.float32, filter_shape), dtype=tf.float32)

    # Ensure that FuseBinaryIntoFollowingAffine works with an input which
    # is shared by multiple affine ops.
    conv_input = input_tensor + 0.1

    # Construct 2 Conv2D operations which use exactly the same input and
    # weights.
    result1 = tf.nn.conv2d(
        conv_input,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    result2 = tf.nn.conv2d(
        conv_input,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    # Add MUL ops after Conv2D ops. These MUL ops should be fused into the
    # weights of Conv2D.
    result1 = result1 * 2
    result2 = result2 * 3
    # Add the 2 results up.
    out = result1 + result2
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, unused_filter_shape = get_tensor_shapes(parameters)
    values = [create_tensor_data(np.float32, input_shape)]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


# Note: This is a regression test for a bug (b/112303004) that Toco incorrectly
# transforms Conv into DepthwiseConv when two Conv ops share the same constant
# weight tensor.
def make_conv_to_depthwiseconv_with_shared_weights_tests(zip_path):
  """Make a test where 2 Conv ops shared the same constant weight tensor."""

  test_parameters = [{
      "input_shape": [[1, 10, 10, 1]],
      "filter_shape": [[3, 3]],
      "strides": [[1, 1, 1, 1]],
      "dilations": [[1, 1, 1, 1]],
      "padding": ["SAME"],
      "data_format": ["NHWC"],
      "channel_multiplier": [3],
  }]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    # Construct a constant weights tensor which will be used by both Conv2D.
    filter_tensor = tf.constant(
        create_tensor_data(np.float32, filter_shape), dtype=tf.float32)
    input_tensors = [input_tensor]

    # Construct 2 Conv2D operations which use exactly the same input and
    # weights.
    result1 = tf.nn.conv2d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    result2 = tf.nn.conv2d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    # Add the 2 results up.
    out = result1 + result2
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, unused_filter_shape = get_tensor_shapes(parameters)
    values = [create_tensor_data(np.float32, input_shape)]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_depthwiseconv_tests(zip_path):
  """Make a set of tests to do convolution."""

  # Tensorflow only supports equal strides
  test_parameters = [
      {
          "input_shape": [[1, 3, 4, 3], [1, 10, 10, 3]],
          "filter_size": [[1, 1], [1, 2], [3, 3]],
          "strides": [[1, 1, 1, 1], [1, 3, 3, 1]],
          "dilations": [[1, 1, 1, 1], [1, 3, 2, 1], [1, 2, 2, 1]],
          "channel_multiplier": [1, 2],
          "rate": [[1, 1]],
          "padding": ["SAME", "VALID"],
          "data_format": ["NHWC"],
          "constant_filter": [True, False],
      },
      {
          "input_shape": [[1, 3, 4, 3]],
          "filter_size": [[1, 1]],
          "strides": [[1, 1, 2, 1]],  # TF needs [1, x, x, 1]
          "dilations": [[1, 1, 1, 1], [1, 2, 2, 1]],
          "channel_multiplier": [2],
          "rate": [[2, 2]],  #  Only [1, 1] is supported
          "padding": ["SAME"],
          "data_format": ["NHWC"],
          "constant_filter": [True, False],
      }
  ]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_size"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a depthwise conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    # Get filter input either as a placeholder or constants. Also get a list of
    # the input tensors that are represented as placeholders.
    if parameters["constant_filter"]:
      filter_input = create_tensor_data(np.float32, filter_shape)
      input_tensors = [input_tensor]
    else:
      filter_input = tf.placeholder(
          dtype=tf.float32, name="filter", shape=filter_shape)
      input_tensors = [input_tensor, filter_input]

    out = tf.nn.depthwise_conv2d(
        input_tensor,
        filter_input,
        strides=parameters["strides"],
        rate=parameters["rate"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [create_tensor_data(np.float32, input_shape)]
    if not parameters["constant_filter"]:
      values.append(create_tensor_data(np.float32, filter_shape))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_split_tests(zip_path):
  """Make a set of tests to do tf.split."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]],
      "num_or_size_splits": [1, 2, 3, 4, 5],
      "axis": [0, 1, 2, 3, -4, -3, -2, -1],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.split(
        input_tensor, parameters["num_or_size_splits"], parameters["axis"])
    return [input_tensor], [out[0]]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [create_tensor_data(np.float32, parameters["input_shape"])]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_splitv_tests(zip_path):
  """Make a set of tests to do tf.split_v."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]],
      "size_splits": [[2, 2], [1, 3], [4, 2], [5, 3],
                      [-1, 1], [-1, 2], [-1, 4]],
      "axis": [0, 1, 2, 3, -4, -3, -2, -1],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.split(input_tensor, parameters["size_splits"], parameters["axis"])
    return [input_tensor], [out[0]]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [create_tensor_data(np.float32, parameters["input_shape"])]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_concat_tests(zip_path):
  """Make a set of tests to do concatenation."""

  test_parameters = [{
      "base_shape": [[1, 3, 4, 3], [3, 4]],
      "num_tensors": [1, 2, 3, 4, 5, 6],
      "axis": [0, 1, 2, 3, -3, -2, -1],
      "type": [tf.float32, tf.uint8, tf.int32, tf.int64],
  }]

  def get_shape(parameters, delta):
    """Return a tweaked version of 'base_shape'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    if axis < 0:
      axis += len(shape)
    if axis < len(shape):
      shape[axis] += delta
    return shape

  def build_graph(parameters):
    all_tensors = []
    for n in range(0, parameters["num_tensors"]):
      input_tensor = tf.placeholder(dtype=parameters["type"],
                                    name=("input%d" % n),
                                    shape=get_shape(parameters, n))
      all_tensors.append(input_tensor)
    out = tf.concat(all_tensors, parameters["axis"])
    return all_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    all_values = []
    for n in range(0, parameters["num_tensors"]):
      input_values = create_tensor_data(
          parameters["type"], get_shape(parameters, n))
      all_values.append(input_values)
    return all_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, all_values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_fully_connected_tests(zip_path):
  """Make a set of tests to do fully_connected."""

  test_parameters = [{
      "shape1": [[3, 3]],
      "shape2": [[3, 3]],
      "transpose_a": [True, False],
      "transpose_b": [True, False],
      "constant_filter": [True, False],
  }, {
      "shape1": [[4, 4], [1, 4], [4]],
      "shape2": [[4, 4], [4, 1], [4]],
      "transpose_a": [False],
      "transpose_b": [False],
      "constant_filter": [True, False],
  }, {
      "shape1": [[40, 37]],
      "shape2": [[37, 40]],
      "transpose_a": [False],
      "transpose_b": [False],
      "constant_filter": [True, False],
  }, {
      "shape1": [[40, 37]],
      "shape2": [[40, 37]],
      "transpose_a": [False],
      "transpose_b": [True],
      "constant_filter": [True, False],
  }]

  def build_graph(parameters):
    """Build a matmul graph given `parameters`."""
    input_tensor1 = tf.placeholder(dtype=tf.float32, name="input1",
                                   shape=parameters["shape1"])

    # Get input_tensor2 either as a placeholder or constants. Also get a list of
    # the input tensors that are represented as placeholders.
    if parameters["constant_filter"]:
      input_tensor2 = create_tensor_data(np.float32, parameters["shape2"])
      input_tensors = [input_tensor1]
    else:
      input_tensor2 = tf.placeholder(
          dtype=tf.float32, name="input2", shape=parameters["shape2"])
      input_tensors = [input_tensor1, input_tensor2]

    out = tf.matmul(input_tensor1, input_tensor2,
                    transpose_a=parameters["transpose_a"],
                    transpose_b=parameters["transpose_b"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input_values1) or 2
    # tensors (input_values1, input_values2) based on whether the second input
    # is a constant or variable input.
    values = [create_tensor_data(np.float32, shape=parameters["shape1"])]
    if not parameters["constant_filter"]:
      values.append(create_tensor_data(np.float32, parameters["shape2"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_l2norm_tests(zip_path):
  """Make a set of tests to do l2norm."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[5, 7], [1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3],
                      [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "dim": [0, 1, 2, 3, [2, 3], -2],
      "epsilon": [None, 1e-12, 1e-3],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    if parameters["epsilon"]:
      out = tf.nn.l2_normalize(
          input_tensor, parameters["dim"], epsilon=parameters["epsilon"])
    else:
      out = tf.nn.l2_normalize(input_tensor, parameters["dim"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-4, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_local_response_norm_tests(zip_path):
  """Make a set of tests to do local_response_norm."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[1, 1, 1, 1], [1, 3, 4, 3], [3, 15, 14, 3]],
      "depth_radius": [None, 0, 1, 3, 5],
      "bias": [None, 0.3, -0.1],
      "alpha": [None, 2, -3],
      "beta": [None, 0.25, 2],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.nn.local_response_normalization(
        input_tensor, depth_radius=parameters["depth_radius"],
        bias=parameters["bias"], alpha=parameters["alpha"],
        beta=parameters["beta"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        np.float32, parameters["input_shape"], min_value=-4, max_value=10)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_pad_tests(zip_path):
  """Make a set of tests to do pad."""

  # TODO(nupurgarg): Add test for tf.uint8.
  test_parameters = [
      # 4D:
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 1, 2, 1], [2, 1, 1, 1]],
          "paddings": [[[0, 0], [0, 1], [2, 3], [0, 0]], [[0, 1], [0, 0],
                                                          [0, 0], [2, 3]]],
          "constant_paddings": [True, False],
      },
      # 2D:
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 2]],
          "paddings": [[[0, 1], [2, 3]]],
          "constant_paddings": [True, False],
      },
      # 1D:
      {
          "dtype": [tf.int32],
          "input_shape": [[1]],
          "paddings": [[[1, 2]]],
          "constant_paddings": [False],
      },
  ]

  def build_graph(parameters):
    """Build a pad graph given `parameters`."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])

    # Get paddings as either a placeholder or constants.
    if parameters["constant_paddings"]:
      paddings = parameters["paddings"]
      input_tensors = [input_tensor]
    else:
      shape = [len(parameters["paddings"]), 2]
      paddings = tf.placeholder(dtype=tf.int32, name="padding", shape=shape)
      input_tensors = [input_tensor, paddings]

    out = tf.pad(input_tensor, paddings=paddings)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_paddings"]:
      values.append(np.array(parameters["paddings"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_padv2_tests(zip_path):
  """Make a set of tests to do padv2."""

  # TODO(nupurgarg): Add test for tf.uint8.
  test_parameters = [
      # 4D:
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 1, 2, 1], [2, 1, 1, 1]],
          "paddings": [[[0, 0], [0, 1], [2, 3], [0, 0]], [[0, 1], [0, 0],
                                                          [0, 0], [2, 3]]],
          "constant_paddings": [True, False],
          "constant_values": [0, 2],
      },
      # 2D:
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 2]],
          "paddings": [[[0, 1], [2, 3]]],
          "constant_paddings": [True, False],
          "constant_values": [0, 2],
      },
      # 1D:
      {
          "dtype": [tf.int32],
          "input_shape": [[1]],
          "paddings": [[[0, 1]]],
          "constant_paddings": [False],
          "constant_values": [0, 2],
      },
  ]

  def build_graph(parameters):
    """Build a pad graph given `parameters`."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])

    # Get paddings as either a placeholder or constants.
    if parameters["constant_paddings"]:
      paddings = parameters["paddings"]
      input_tensors = [input_tensor]
    else:
      shape = [len(parameters["paddings"]), 2]
      paddings = tf.placeholder(dtype=tf.int32, name="padding", shape=shape)
      input_tensors = [input_tensor, paddings]

    out = tf.pad(input_tensor, paddings=paddings,
                 constant_values=parameters["constant_values"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_paddings"]:
      values.append(np.array(parameters["paddings"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_reshape_tests(zip_path):
  """Make a set of tests to do reshape."""

  # All shapes below are suitable for tensors with 420 elements.
  test_parameters = [{
      "dtype": [tf.float32, tf.int32],
      "input_shape": [[3, 4, 5, 7], [4, 105], [21, 5, 2, 2], [420]],
      "output_shape": [[15, 28], [420], [1, -1, 5, 7], [-1]],
      "constant_shape": [True, False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1]],
      "output_shape": [[]],
      "constant_shape": [True, False],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])

    # Get shape as either a placeholder or constants.
    if parameters["constant_shape"]:
      output_shape = parameters["output_shape"]
      input_tensors = [input_tensor]
    else:
      # The shape of the shape tensor.
      shape_tensor_shape = [len(parameters["output_shape"])]
      output_shape = tf.placeholder(
          dtype=tf.int32, name="output_shape", shape=shape_tensor_shape)
      input_tensors = [input_tensor, output_shape]
    out = tf.reshape(input_tensor, shape=output_shape)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_shape"]:
      values.append(np.array(parameters["output_shape"]))

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_shape_tests(zip_path):
  """Make a set of tests to do shape."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[], [0], [1, 1, 1, 3], [2, 3, 4, 5], [5, 5], [10]],
      "out_type": [tf.int32, tf.int64],
  }]

  def build_graph(parameters):
    """Build the shape op testing graph."""
    # Note that we intentionally leave out the shape from the input placeholder
    # to prevent the Shape operation from being optimized out during conversion.
    input_value = tf.placeholder(dtype=parameters["input_dtype"], name="input")
    out = tf.shape(input_value, out_type=parameters["out_type"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_one_hot_tests(zip_path):
  """Make a set of tests to do one_hot."""

  test_parameters = [{
      "indices_type": [tf.int32, tf.int64],
      "indices_shape": [[3], [4, 4], [1, 5], [5, 1]],
      "axis": [0, 1],
      "dtype": [tf.int32, tf.int64, tf.float32],
      "provide_optional_inputs": [True, False],
  }]

  def build_graph(parameters):
    indices = tf.placeholder(
        dtype=parameters["indices_type"],
        name="indices",
        shape=parameters["indices_shape"])
    depth = tf.placeholder(dtype=tf.int32, name="depth", shape=())

    if not parameters["provide_optional_inputs"]:
      out = tf.one_hot(indices=indices, depth=depth)
      return [indices, depth], [out]

    on_value = tf.placeholder(
        dtype=parameters["dtype"], name="on_value", shape=())
    off_value = tf.placeholder(
        dtype=parameters["dtype"], name="off_value", shape=())
    out = tf.one_hot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=parameters["axis"],
        dtype=parameters["dtype"])
    return [indices, depth, on_value, off_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [
        create_tensor_data(
            parameters["indices_type"],
            shape=parameters["indices_shape"],
            min_value=-1,
            max_value=10),
        create_tensor_data(tf.int32, shape=None, min_value=1, max_value=10),
    ]

    if parameters["provide_optional_inputs"]:
      input_values.append(
          create_tensor_data(
              parameters["dtype"], shape=None, min_value=1, max_value=10))
      input_values.append(
          create_tensor_data(
              parameters["dtype"], shape=None, min_value=-1, max_value=0))

    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_resize_bilinear_tests(zip_path):
  """Make a set of tests to do resize_bilinear."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 3, 4, 3], [1, 10, 2, 1]],
      "size": [[1, 1], [4, 3], [2, 2], [5, 6]],
      "align_corners": [None, True, False],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.image.resize_bilinear(input_tensor, size=parameters["size"],
                                   align_corners=parameters["align_corners"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_resize_nearest_neighbor_tests(zip_path):
  """Make a set of tests to do resize_nearest_neighbor."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 3, 4, 3], [1, 10, 2, 1]],
      "size": [[1, 1], [4, 3], [2, 2], [5, 6]],
      "align_corners": [False],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.image.resize_nearest_neighbor(
        input_tensor,
        size=parameters["size"],
        align_corners=parameters["align_corners"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_sigmoid_tests(zip_path):
  """Make a set of tests to do sigmoid."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 3, 4, 3], [4], [], [1, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.sigmoid(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_softmax_tests(zip_path):
  """Make a set of tests to do softmax."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[1, 3, 4, 3], [2, 3]],
      "dim": [-1, 0],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[4, 7]],
      "dim": [-1, 1],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.nn.softmax(input_tensor, dim=parameters["dim"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_space_to_depth_tests(zip_path):
  """Make a set of tests to do space_to_depth."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32, tf.uint8, tf.int64],
      "input_shape": [[2, 12, 24, 1]],
      "block_size": [2, 3, 4],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(dtype=parameters["dtype"], name="input",
                                  shape=parameters["input_shape"])
    out = tf.space_to_depth(input_tensor, block_size=parameters["block_size"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_space_to_batch_nd_tests(zip_path):
  """Make a set of tests to do space_to_batch_nd."""

  # TODO(nupurgarg): Add test for uint8.
  test_parameters = [
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 2, 2, 3], [2, 2, 4, 1]],
          "block_shape": [[1, 3], [2, 2]],
          "paddings": [[[0, 0], [0, 0]], [[0, 0], [2, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[2, 3, 7, 3]],
          "block_shape": [[1, 3], [2, 2]],
          "paddings": [[[0, 0], [2, 0]], [[1, 0], [1, 0]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
      # Non-4D use case: 1 bath dimension, 3 spatial dimensions, 2 others.
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 4, 4, 4, 1, 1]],
          "block_shape": [[2, 2, 2]],
          "paddings": [[[0, 0], [0, 0], [0, 0]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
  ]

  def build_graph(parameters):
    """Build a space_to_batch graph given `parameters`."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    input_tensors = [input_tensor]

    # Get block_shape either as a const or as a placeholder (tensor).
    if parameters["constant_block_shape"]:
      block_shape = parameters["block_shape"]
    else:
      shape = [len(parameters["block_shape"])]
      block_shape = tf.placeholder(dtype=tf.int32, name="shape", shape=shape)
      input_tensors.append(block_shape)

    # Get paddings either as a const or as a placeholder (tensor).
    if parameters["constant_paddings"]:
      paddings = parameters["paddings"]
    else:
      shape = [len(parameters["paddings"]), 2]
      paddings = tf.placeholder(dtype=tf.int32, name="paddings", shape=shape)
      input_tensors.append(paddings)

    out = tf.space_to_batch_nd(input_tensor, block_shape, paddings)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_block_shape"]:
      values.append(np.array(parameters["block_shape"]))
    if not parameters["constant_paddings"]:
      values.append(np.array(parameters["paddings"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_batch_to_space_nd_tests(zip_path):
  """Make a set of tests to do batch_to_space_nd."""

  test_parameters = [
      {
          "dtype": [tf.float32, tf.int64, tf.int32],
          "input_shape": [[12, 3, 3, 1]],
          "block_shape": [[1, 4], [2, 2], [3, 4]],
          "crops": [[[0, 0], [0, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True, False],
          "constant_crops": [True, False],
      },
      # Non-4D use case: 1 bath dimension, 3 spatial dimensions, 2 others.
      {
          "dtype": [tf.float32],
          "input_shape": [[8, 2, 2, 2, 1, 1]],
          "block_shape": [[2, 2, 2]],
          "crops": [[[0, 0], [0, 0], [0, 0]]],
          "constant_block_shape": [True, False],
          "constant_crops": [True, False],
      },
  ]

  def build_graph(parameters):
    """Build a batch_to_space graph given `parameters`."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    input_tensors = [input_tensor]

    # Get block_shape either as a const or as a placeholder (tensor).
    if parameters["constant_block_shape"]:
      block_shape = parameters["block_shape"]
    else:
      shape = [len(parameters["block_shape"])]
      block_shape = tf.placeholder(dtype=tf.int32, name="shape", shape=shape)
      input_tensors.append(block_shape)

    # Get crops either as a const or as a placeholder (tensor).
    if parameters["constant_crops"]:
      crops = parameters["crops"]
    else:
      shape = [len(parameters["crops"]), 2]
      crops = tf.placeholder(dtype=tf.int32, name="crops", shape=shape)
      input_tensors.append(crops)

    out = tf.batch_to_space_nd(input_tensor, block_shape, crops)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_block_shape"]:
      values.append(np.array(parameters["block_shape"]))
    if not parameters["constant_crops"]:
      values.append(np.array(parameters["crops"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_transpose_tests(zip_path):
  """Make a set of tests to do transpose."""

  # TODO(nupurgarg): Add test for uint8.
  test_parameters = [{
      "dtype": [tf.int32, tf.int64, tf.float32],
      "input_shape": [[2, 2, 3]],
      "perm": [[0, 1, 2], [0, 2, 1]],
      "constant_perm": [True, False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4]],
      "perm": [[0, 1, 2, 3], [3, 0, 1, 2]],
      "constant_perm": [True, False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 3, 4, 5]],
      "perm": [[4, 3, 2, 1, 0]],
      "constant_perm": [True, False],
  }]

  def build_graph(parameters):
    """Build a transpose graph given `parameters`."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])

    if parameters["constant_perm"]:
      perm = parameters["perm"]
      input_tensors = [input_tensor]
    else:
      shape = [len(parameters["perm"]), 2]
      perm = tf.placeholder(dtype=tf.int32, name="perm", shape=shape)
      input_tensors = [input_tensor, perm]

    out = tf.transpose(input_tensor, perm=perm)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_perm"]:
      values.append(np.array(parameters["perm"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_squeeze_tests(zip_path):
  """Make a set of tests to do squeeze."""

  test_parameters = [{
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1, 2, 1, 3, 1, 4, 1, 1]],
      "axis": [
          None, [], [0, 2], [4, 7], [-1, 0, 2, 0, 7, -6], [1], [2, 3, 2],
          [-1, -2, -4, -6, -8], [0, 2, 4, 6, 7], [7, 6, 4, 2, 0], [6, 6],
          [0, 1, 2, 3, 4, 5, 6, 7], [-2, -3, 1, 0, 7, -5]
      ],
  }, {
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1]],
      "axis": [None, [], [0], [-1]],
  }, {
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1, 1, 1, 1, 1]],
      "axis": [None, [], [0], [3, 0], [-2, 0, 3, 2]],
  }]

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.squeeze(input_tensor, axis=parameters["axis"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def _make_strided_slice_tests(zip_path, test_parameters):
  """Utility function to make strided_slice_tests based on parameters."""

  def build_graph(parameters):
    """Build graph for stride_slice test."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["constant_indices"]:
      begin = parameters["begin"]
      end = parameters["end"]
      strides = parameters["strides"]
      tensors = [input_tensor]
    else:
      begin = tf.placeholder(
          dtype=parameters["index_type"],
          name="begin",
          shape=[len(parameters["input_shape"])])
      end = tf.placeholder(
          dtype=parameters["index_type"],
          name="end",
          shape=[len(parameters["input_shape"])])
      strides = (
          tf.placeholder(
              dtype=parameters["index_type"],
              name="strides",
              shape=[len(parameters["input_shape"])])
          if parameters["strides"] is not None else None)
      tensors = [input_tensor, begin, end]
      if strides is not None:
        tensors.append(strides)
    out = tf.strided_slice(
        input_tensor,
        begin,
        end,
        strides,
        begin_mask=parameters["begin_mask"],
        end_mask=parameters["end_mask"])
    return tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build inputs for stride_slice test."""
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    index_type = _TF_TYPE_INFO[parameters["index_type"]][0]
    values = [input_values]
    if not parameters["constant_indices"]:
      begin_values = np.array(parameters["begin"]).astype(index_type)
      end_values = np.array(parameters["end"]).astype(index_type)
      stride_values = (
          np.array(parameters["strides"]).astype(index_type)
          if parameters["strides"] is not None else None)
      values.append(begin_values)
      values.append(end_values)
      if stride_values is not None:
        values.append(stride_values)

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_strided_slice_tests(zip_path):
  """Make a set of tests to do strided_slice."""

  # TODO(soroosh): add test/support for uint8.
  test_parameters = [
      # 4-D (basic cases with const/non-const indices).
      {
          "dtype": [tf.float32, tf.int32, tf.int64],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "strides": [None, [2, 1, 3, 1]],
          "begin": [[0, 0, 0, 0]],
          "end": [[12, 2, 2, 5]],
          "begin_mask": [None],
          "end_mask": [None],
          "shrink_axis_mask": [None],
          "constant_indices": [False, True],
      },
      # 4-D with non-trivial begin & end.
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0, 0, 0, 0], [1, 0, 1, 0]],
          "end": [[8, 2, 2, 3], [12, 2, 2, 5]],
          "strides": [None, [2, 1, 3, 1]],
          "begin_mask": [None, 8],
          "end_mask": [None, 3],
          "shrink_axis_mask": [None, 15, -1],
          "constant_indices": [True],
      },
      # Begin, end, strides dim are different from input shape
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0]],
          "end": [[1]],
          "strides": [None, [1]],
          "begin_mask": [0],
          "end_mask": [0],
          "shrink_axis_mask": [1],
          "constant_indices": [True],
      },
      # 2-D
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[2, 3]],
          "begin": [[0, 0]],
          "end": [[2, 2]],
          "strides": [None, [2, 2]],
          "begin_mask": [None, 1, 2],
          "end_mask": [None, 1, 2],
          "shrink_axis_mask": [None, 1, 2, 3, -1],
          "constant_indices": [False, True],
      },
      # Negative strides
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[2, 3]],
          "begin": [[0, -1]],
          "end": [[2, -3]],
          "strides": [[1, -1]],
          "begin_mask": [None, 1, 2],
          "end_mask": [None, 1, 2],
          "shrink_axis_mask": [None, 1, 2, 3, -1],
          "constant_indices": [False],
      },
  ]
  _make_strided_slice_tests(zip_path, test_parameters)


def make_strided_slice_1d_exhaustive_tests(zip_path):
  """Make a set of exhaustive tests for 1D strided_slice."""
  test_parameters = [
      # 1-D Exhaustive
      {
          "dtype": [tf.float32],
          "index_type": [tf.int32],
          "input_shape": [[3]],
          "begin": [[-2], [-1], [0], [1], [2]],
          "end": [[-2], [-1], [0], [1], [2]],
          "strides": [[-2], [-1], [1], [2]],
          "begin_mask": [0, 1],
          "end_mask": [0, 1],
          "shrink_axis_mask": [0],
          "constant_indices": [False],
      },
  ]
  _make_strided_slice_tests(zip_path, test_parameters)


def make_strided_slice_buggy_tests(zip_path):
  """Make a set of tests to show strided_slice yields incorrect results."""

  test_parameters = [{
      "unused_iteration_counter": [1],
  }]

  def build_graph(parameters):
    """Build the strided_slice op testing graph."""
    del parameters
    input_values = tf.placeholder(dtype=tf.float32, shape=[4, 2])
    data = tf.constant([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]], tf.float32)
    return [input_values], [input_values + data[:, :2]]

  def build_inputs(parameters, sess, inputs, outputs):
    del parameters
    input_values = np.zeros([4, 2], dtype=np.float32)
    return [input_values], sess.run(
        outputs, feed_dict={inputs[0]: input_values})

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_lstm_tests(zip_path):
  """Make a set of tests to do basic Lstm cell."""

  test_parameters = [
      {
          "dtype": [tf.float32],
          "num_batchs": [1],
          "time_step_size": [1],
          "input_vec_size": [3],
          "num_cells": [4],
          "split_tflite_lstm_inputs": [False],
      },
  ]

  def build_graph(parameters):
    """Build a simple graph with BasicLSTMCell."""

    num_batchs = parameters["num_batchs"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    num_cells = parameters["num_cells"]
    inputs_after_split = []
    for i in xrange(time_step_size):
      one_timestamp_input = tf.placeholder(
          dtype=parameters["dtype"],
          name="split_{}".format(i),
          shape=[num_batchs, input_vec_size])
      inputs_after_split.append(one_timestamp_input)
    # Currently lstm identifier has a few limitations: only supports
    # forget_bias == 0, inner state activiation == tanh.
    # TODO(zhixianyan): Add another test with forget_bias == 1.
    # TODO(zhixianyan): Add another test with relu as activation.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_cells, forget_bias=0.0, state_is_tuple=True)
    cell_outputs, _ = rnn.static_rnn(
        lstm_cell, inputs_after_split, dtype=tf.float32)
    out = cell_outputs[-1]
    return inputs_after_split, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Feed inputs, assign variables, and freeze graph."""

    with tf.variable_scope("", reuse=True):
      kernel = tf.get_variable("rnn/basic_lstm_cell/kernel")
      bias = tf.get_variable("rnn/basic_lstm_cell/bias")
      kernel_values = create_tensor_data(
          parameters["dtype"], [kernel.shape[0], kernel.shape[1]], -1, 1)
      bias_values = create_tensor_data(parameters["dtype"], [bias.shape[0]], 0,
                                       1)
      sess.run(tf.group(kernel.assign(kernel_values), bias.assign(bias_values)))

    num_batchs = parameters["num_batchs"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    input_values = []
    for _ in xrange(time_step_size):
      tensor_data = create_tensor_data(parameters["dtype"],
                                       [num_batchs, input_vec_size], 0, 1)
      input_values.append(tensor_data)
    out = sess.run(outputs, feed_dict=dict(zip(inputs, input_values)))
    return input_values, out

  # TODO(zhixianyan): Automatically generate rnn_states for lstm cell.
  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.rnn_states = (
      "{state_array:rnn/BasicLSTMCellZeroState/zeros,"
      "back_edge_source_array:rnn/basic_lstm_cell/Add_1,size:4},"
      "{state_array:rnn/BasicLSTMCellZeroState/zeros_1,"
      "back_edge_source_array:rnn/basic_lstm_cell/Mul_2,size:4}")

  make_zip_of_tests(
      zip_path,
      test_parameters,
      build_graph,
      build_inputs,
      extra_toco_options,
      use_frozen_graph=True)


def make_l2_pool(input_tensor, ksize, strides, padding, data_format):
  """Given an input perform a sequence of TensorFlow ops to produce l2pool."""
  return tf.sqrt(tf.nn.avg_pool(
      tf.square(input_tensor), ksize=ksize, strides=strides,
      padding=padding, data_format=data_format))


def make_topk_tests(zip_path):
  """Make a set of tests to do topk."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[10], [5, 20]],
      "input_k": [None, 1, 3],
  }]

  def build_graph(parameters):
    """Build the topk op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    if parameters["input_k"] is not None:
      k = tf.placeholder(dtype=tf.int32, name="input_k", shape=[])
      inputs = [input_value, k]
    else:
      k = tf.constant(3, name="k")
      inputs = [input_value]
    out = tf.nn.top_k(input_value, k)
    return inputs, [out[1]]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    if parameters["input_k"] is not None:
      k = np.array(parameters["input_k"], dtype=np.int32)
      return [input_value, k], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value, k])))
    else:
      return [input_value], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_arg_min_max_tests(zip_path):
  """Make a set of tests to do arg_max."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[], [1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5], [10]],
      "output_type": [tf.int32, tf.int64],
      "is_arg_max": [True],
  }]

  def build_graph(parameters):
    """Build the topk op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    axis = random.randint(0, max(len(parameters["input_shape"]) - 1, 0))
    if parameters["is_arg_max"]:
      out = tf.arg_max(input_value, axis, output_type=parameters["output_type"])
    else:
      out = tf.arg_min(input_value, axis, output_type=parameters["output_type"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_equal_tests(zip_path):
  """Make a set of tests to do equal."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([], []),
                           ([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the equal op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.equal(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_not_equal_tests(zip_path):
  """Make a set of tests to do not equal."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the not euqal op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.not_equal(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_greater_tests(zip_path):
  """Make a set of tests to do greater."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the greater op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.greater(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_greater_equal_tests(zip_path):
  """Make a set of tests to do greater_equal."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the greater_equal op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.greater_equal(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_less_tests(zip_path):
  """Make a set of tests to do less."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the less op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.less(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_less_equal_tests(zip_path):
  """Make a set of tests to do less_equal."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape_pair": [([1, 1, 1, 3], [1, 1, 1, 3]),
                           ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                           ([5, 5], [1]), ([10], [2, 4, 10])],
  }]

  def build_graph(parameters):
    """Build the less_equal op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape_pair"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_pair"][1])
    out = tf.less_equal(input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_pair"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_floor_tests(zip_path):
  """Make a set of tests to do floor."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    """Build the floor op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape"])
    out = tf.floor(input_value)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict={inputs[0]: input_value})

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_ceil_tests(zip_path):
  """Make a set of tests to do ceil."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    """Build the ceil op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input1",
        shape=parameters["input_shape"])
    out = tf.ceil(input_value)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict={inputs[0]: input_value})

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_neg_tests(zip_path):
  """Make a set of tests to do neg."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 3, 4, 3], [5]],
  }]

  def build_graph(parameters):
    """Build the neg op testing graph."""
    input_tensor = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.negative(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = create_tensor_data(parameters["input_dtype"],
                                parameters["input_shape"])
    return [values], sess.run(outputs, feed_dict=dict(zip(inputs, [values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_zeros_like_tests(zip_path):
  """Make a set of tests to do zeros_like."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    """Build the zeros_like op testing graph."""
    input_tensor = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    zeros = tf.zeros_like(input_tensor)
    # This maximum node is so that toco can perform the constants-propagation
    # through the above zeros_like, which it can't do if the output of the
    # zeros_like as an output of the whole graphs (graph outputs can't be
    # constants). If toco does not perform such constants-propagation then
    # the resulting tflite graph retains the zeros_like as a Fill op, which
    # is unsupported by TFLite, even as a custom op.
    out = tf.maximum(zeros, input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = create_tensor_data(parameters["input_dtype"],
                                parameters["input_shape"])
    return [values], sess.run(outputs, feed_dict=dict(zip(inputs, [values])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def _make_elementwise_tests(op):
  """Make a set of tests to do element-wise operations."""

  def f(zip_path):
    """Actual function that generates examples."""
    test_parameters = [{
        "input_dtype": [tf.float32],
        "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
    }]

    def build_graph(parameters):
      """Build the unary op testing graph."""
      input_value = tf.placeholder(
          dtype=parameters["input_dtype"],
          name="input1",
          shape=parameters["input_shape"])
      out = op(input_value)
      return [input_value], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      input_value = create_tensor_data(parameters["input_dtype"],
                                       parameters["input_shape"])
      return [input_value], sess.run(
          outputs, feed_dict={inputs[0]: input_value})

    make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)

  return f


def make_sin_tests(zip_path):
  """Make a set of tests to do sin."""
  return _make_elementwise_tests(tf.sin)(zip_path)


def make_log_tests(zip_path):
  """Make a set of tests to do log."""
  return _make_elementwise_tests(tf.log)(zip_path)


def make_sqrt_tests(zip_path):
  """Make a set of tests to do sqrt."""
  return _make_elementwise_tests(tf.sqrt)(zip_path)


def make_rsqrt_tests(zip_path):
  """Make a set of tests to do 1/sqrt."""
  return _make_elementwise_tests(tf.rsqrt)(zip_path)


def make_square_tests(zip_path):
  """Make a set of tests to do square."""
  return _make_elementwise_tests(tf.square)(zip_path)


def make_where_tests(zip_path):
  """Make a set of tests to do where."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape_set": [([1, 2, 3, 4], [1, 2, 3, 4]),],
  }]

  def build_graph(parameters):
    """Build the where op testing graph."""
    input_value1 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_set"][0])
    input_value2 = tf.placeholder(
        dtype=parameters["input_dtype"],
        name="input3",
        shape=parameters["input_shape_set"][1])
    less = tf.less(input_value1, input_value2)
    out = tf.where(less, input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_set"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_set"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_slice_tests(zip_path):
  """Make a set of tests to do slice."""

  # TODO(renjieliu): add test/support for uint8.
  test_parameters = [
      # 4-D
      {
          "dtype": [tf.float32, tf.int32, tf.int64],
          "index_type": [tf.int32, tf.int64],
          "input_shape": [[12, 2, 2, 5]],
          "begin": [[0, 0, 0, 0], [1, 0, 1, 0]],
          "size": [[8, 2, 2, 3], [11, 2, 1, 5]],
      },
      # 2-D
      {
          "dtype": [tf.float32, tf.int32, tf.int64],
          "index_type": [tf.int32, tf.int64],
          "input_shape": [[2, 3]],
          "begin": [[0, 0], [1, 0]],
          "size": [[2, 3], [2, 2]],
      },
  ]

  def build_graph(parameters):
    """Build graph for slice test."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    begin = tf.placeholder(
        dtype=parameters["index_type"],
        name="begin",
        shape=[len(parameters["input_shape"])])
    size = tf.placeholder(
        dtype=parameters["index_type"],
        name="size",
        shape=[len(parameters["input_shape"])])
    tensors = [input_tensor, begin, size]
    out = tf.slice(input_tensor, begin, size)
    return tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build inputs for slice test."""
    input_values = create_tensor_data(parameters["dtype"],
                                      parameters["input_shape"])
    index_type = _TF_TYPE_INFO[parameters["index_type"]][0]

    begin_values = np.array(parameters["begin"]).astype(index_type)
    size_values = np.array(parameters["size"]).astype(index_type)
    values = [input_values, begin_values, size_values]

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


# Since compute output_shape is fairly complicated for
# tf.nn.conv2d_backprop_input input_sizes argument, so we here first perform a
# "conv2d" operation to get the output, then we use the output to feed in
# tf.nn.conv2d_backprop_input.
# This test will depend on the "conv2d" operation's correctness.
def make_transpose_conv_tests(zip_path):
  """Make a set of tests to do transpose_conv."""

  # Tensorflow only supports equal strides
  test_parameters = [{
      "input_shape": [[1, 3, 4, 1], [1, 10, 10, 3], [3, 20, 20, 1]],
      "filter_size": [[1, 1], [1, 2], [3, 3]],
      "strides": [[1, 1, 1, 1], [1, 3, 3, 1]],
      "padding": ["SAME", "VALID"],
      "data_format": ["NHWC"],
      "channel_multiplier": [1, 2],
  }]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_size"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a transpose_conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    filter_input = tf.placeholder(
        dtype=tf.float32, name="filter", shape=filter_shape)

    conv_outputs = tf.nn.conv2d(
        input_tensor,
        filter_input,
        strides=parameters["strides"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    out = tf.nn.conv2d_backprop_input(
        input_shape,
        filter_input,
        conv_outputs,
        strides=parameters["strides"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    input_tensors = [input_tensor, filter_input]
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [
        create_tensor_data(np.float32, input_shape),
        create_tensor_data(np.float32, filter_shape)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_tile_tests(zip_path):
  """Make a set of tests to do tile."""
  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.bool],
      "input_shape": [[3, 2, 1], [2, 2, 2]],
      "multiplier_dtype": [tf.int32, tf.int64],
      "multiplier_shape": [[3]]
  }]

  def build_graph(parameters):
    """Build the tile op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_dtype"],
        shape=parameters["input_shape"],
        name="input")
    multiplier_value = tf.placeholder(
        dtype=parameters["multiplier_dtype"],
        shape=parameters["multiplier_shape"],
        name="multiplier")
    out = tf.tile(input_value, multiplier_value)
    return [input_value, multiplier_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    multipliers_value = create_tensor_data(
        parameters["multiplier_dtype"],
        parameters["multiplier_shape"],
        min_value=0)
    return [input_value, multipliers_value], sess.run(
        outputs,
        feed_dict={
            inputs[0]: input_value,
            inputs[1]: multipliers_value
        })

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_expand_dims_tests(zip_path):
  """Make a set of tests to do expand_dims."""

  test_parameters = [{
      "input_type": [tf.float32, tf.int32],
      "input_shape": [[3, 4], [10, 10, 3]],
      "axis_value": [0, 1, 2, -1, -2],
  }]

  def build_graph(parameters):
    """Build the where op testing graph."""
    input_value = tf.placeholder(
        dtype=parameters["input_type"],
        name="input",
        shape=parameters["input_shape"])
    axis_value = tf.placeholder(dtype=tf.int32, name="axis", shape=[1])
    out = tf.expand_dims(input_value, axis=axis_value)
    return [input_value, axis_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_type"],
                                     parameters["input_shape"])
    axis_value = np.array([parameters["axis_value"]], dtype=np.int32)
    return [input_value, axis_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value, axis_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_sparse_to_dense_tests(zip_path):
  """Make a set of tests to do sparse to dense."""

  test_parameters = [{
      "value_dtype": [tf.float32, tf.int32],
      "index_dtype": [tf.int32, tf.int64],
      "value_count": [1, 3, 6, 8],
      "dense_shape": [[15], [3, 10], [4, 4, 4, 4], [7, 10, 9]],
      "default_value": [0, -1],
      "value_is_scalar": [True, False],
  }]

  # Return a single value for 1-D dense shape, but a tuple for other shapes.
  def generate_index(dense_shape):
    if len(dense_shape) == 1:
      return np.random.randint(dense_shape[0])
    else:
      index = []
      for shape in dense_shape:
        index.append(np.random.randint(shape))
      return tuple(index)

  def build_graph(parameters):
    """Build the sparse_to_dense op testing graph."""
    dense_shape = parameters["dense_shape"]

    # Special handle for value_is_scalar case.
    # value_count must be 1.
    if parameters["value_is_scalar"] and parameters["value_count"] == 1:
      value = tf.placeholder(
          name="value", dtype=parameters["value_dtype"], shape=())
    else:
      value = tf.placeholder(
          name="value",
          dtype=parameters["value_dtype"],
          shape=[parameters["value_count"]])
    indices = set()
    while len(indices) < parameters["value_count"]:
      indices.add(generate_index(dense_shape))
    indices = tf.constant(tuple(indices), dtype=parameters["index_dtype"])
    # TODO(renjieliu): Add test for validate_indices case.
    out = tf.sparse_to_dense(
        indices,
        dense_shape,
        value,
        parameters["default_value"],
        validate_indices=False)

    return [value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    if parameters["value_is_scalar"] and parameters["value_count"] == 1:
      input_value = create_scalar_data(parameters["value_dtype"])
    else:
      input_value = create_tensor_data(parameters["value_dtype"],
                                       [parameters["value_count"]])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_pack_tests(zip_path):
  """Make a set of tests to do stack."""

  test_parameters = [
      # Avoid creating all combinations to keep the test size small.
      {
          "dtype": [tf.float32],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [1, 2, 3, 4, 5, 6],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
      },
      {
          "dtype": [tf.int32],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [6],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
      },
      {
          "dtype": [tf.int64],
          "base_shape": [[3, 4, 3], [3, 4], [5]],
          "num_tensors": [5],
          "axis": [0, 1, 2, 3],
          "additional_shape": [1, 2, 3],
      }
  ]

  def get_shape(parameters):
    """Return a tweaked version of 'base_shape'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    if axis < len(shape):
      shape[axis] += parameters["additional_shape"]
    return shape

  def build_graph(parameters):
    all_tensors = []
    for n in range(0, parameters["num_tensors"]):
      input_tensor = tf.placeholder(
          dtype=parameters["dtype"],
          name=("input%d" % n),
          shape=get_shape(parameters))
      all_tensors.append(input_tensor)
    out = tf.stack(all_tensors, parameters["axis"])
    return all_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    all_values = []
    for _ in range(0, parameters["num_tensors"]):
      input_values = create_tensor_data(np.float32, get_shape(parameters))
      all_values.append(input_values)
    return all_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, all_values)))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_unpack_tests(zip_path):
  """Make a set of tests to do unstack."""

  test_parameters = [{
      "base_shape": [[3, 4, 3], [3, 4], [5, 6, 7, 8]],
      "axis": [0, 1, 2, 3],
  }]

  def get_valid_axis(parameters):
    """Return a tweaked version of 'axis'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    while axis > len(shape) - 1:
      axis -= 1
    return axis

  def build_graph(parameters):
    input_tensor = tf.placeholder(
        dtype=tf.float32, name=("input"), shape=parameters["base_shape"])
    outs = tf.unstack(input_tensor, axis=get_valid_axis(parameters))
    return [input_tensor], [outs[0]]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(np.float32, shape=parameters["base_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_range_tests(zip_path):
  """Make a set of tests to do range."""

  test_parameters = [{
      "dtype": [tf.int32],
      "offset": [10, 100, 1000],
      "delta": [1, 2, 3, 4, -1, -2, -3, -4],
  }]

  def build_graph(parameters):
    """Build the range op testing graph."""
    input_tensor = tf.placeholder(
        dtype=parameters["dtype"], name=("start"), shape=[])
    if parameters["delta"] < 0:
      offset = parameters["offset"] * -1
    else:
      offset = parameters["offset"]
    delta = parameters["delta"]
    limit_tensor = input_tensor + offset
    delta_tensor = tf.constant(delta, dtype=tf.int32)
    out = tf.range(input_tensor, limit_tensor, delta_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_scalar_data(parameters["dtype"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_fill_tests(zip_path):
  """Make a set of tests to do fill."""

  test_parameters = [{
      "dims_dtype": [tf.int32, tf.int64],
      "dims_shape": [[], [1], [3], [3, 3]],
      "value_dtype": [tf.int32, tf.int64, tf.float32],
  }]

  def build_graph(parameters):
    """Build the fill op testing graph."""
    input1 = tf.placeholder(
        dtype=parameters["dims_dtype"],
        name="dims",
        shape=parameters["dims_shape"])
    input2 = tf.placeholder(
        dtype=parameters["value_dtype"], name="value", shape=[])
    out = tf.fill(input1, input2)
    return [input1, input2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input1 = create_tensor_data(parameters["dims_dtype"],
                                parameters["dims_shape"], 1)
    input2 = create_scalar_data(parameters["value_dtype"])
    return [input1, input2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input1, input2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def _make_logical_tests(op):
  """Make a set of tests to do logical operations."""

  def logical(zip_path):
    """Generate examples."""
    test_parameters = [{
        "input_shape_pair": [([], []), ([1, 1, 1, 3], [1, 1, 1, 3]),
                             ([2, 3, 4, 5], [2, 3, 4, 5]), ([2, 3, 3], [2, 3]),
                             ([5, 5], [1]), ([10], [2, 4, 10])],
    }]

    def build_graph(parameters):
      """Build the logical testing graph."""
      input_value1 = tf.placeholder(
          dtype=tf.bool, name="input1", shape=parameters["input_shape_pair"][0])
      input_value2 = tf.placeholder(
          dtype=tf.bool, name="input2", shape=parameters["input_shape_pair"][1])
      out = op(input_value1, input_value2)
      return [input_value1, input_value2], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      input_value1 = create_tensor_data(tf.bool,
                                        parameters["input_shape_pair"][0])
      input_value2 = create_tensor_data(tf.bool,
                                        parameters["input_shape_pair"][1])
      return [input_value1, input_value2], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

    make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)

  return logical


def make_logical_or_tests(zip_path):
  """Make a set of tests to do logical_or."""
  return _make_logical_tests(tf.logical_or)(zip_path)


def make_logical_and_tests(zip_path):
  """Make a set of tests to do logical_and."""
  return _make_logical_tests(tf.logical_and)(zip_path)


def make_logical_xor_tests(zip_path):
  """Make a set of tests to do logical_xor.

    Test logical_not as well.
  """
  return _make_logical_tests(tf.logical_xor)(zip_path)


def make_mirror_pad_tests(zip_path):
  """Make a set of tests to do mirror_pad."""

  test_parameters = [
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["REFLECT"],
          "type": ["const"]
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [1, 1]]],
          "mode": ["REFLECT"],
          "type": ["const"]
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["SYMMETRIC"],
          "type": ["placeholder"]
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["REFLECT"],
          "type": ["placeholder"]
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["SYMMETRIC"],
          "type": ["placeholder"]
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["SYMMETRIC"],
          "type": ["const"]
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["REFLECT"],
          "type": ["const"]
      },
  ]

  def build_graph(parameters):
    """Build the graph for the test case."""

    input_tensor = tf.placeholder(
        dtype=tf.int32, name="input", shape=parameters["input_shape"])
    if parameters["type"] != "const":
      padding_matrix = tf.placeholder(
          dtype=tf.int32,
          name="padding",
          shape=[len(parameters["input_shape"]), 2])
      input_tensors = [input_tensor, padding_matrix]
    else:
      padding_matrix = tf.constant(np.array(parameters["padding_matrix"]))
      input_tensors = [input_tensor]
    output = tf.pad(
        input_tensor, paddings=padding_matrix, mode=parameters["mode"])

    return input_tensors, [output]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [create_tensor_data(tf.int32, parameters["input_shape"])]
    if parameters["type"] != "const":
      input_values.append(np.array(parameters["padding_matrix"]))
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      zip_path,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_success=7)


def make_unroll_batch_matmul_tests(zip_path):
  """Make a set of tests to test unroll_batch_matmul."""

  test_parameters = [{"dtype": [tf.float32], "shape": [[(2, 2, 3), (2, 3, 2)]]}]

  def build_graph(parameters):
    """Build the batch_matmul op testing graph."""
    input_tensor1 = tf.placeholder(
        dtype=parameters["dtype"], shape=parameters["shape"][0])
    input_tensor2 = tf.placeholder(
        dtype=parameters["dtype"], shape=parameters["shape"][1])
    # Should be unrolled and replaced with fully_connected ops in the end.
    out = tf.matmul(input_tensor1, input_tensor2)
    return [input_tensor1, input_tensor2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][0])
    input_value2 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs)


def make_placeholder_with_default_tests(zip_path):
  """Make a set of tests to test placeholder_with_default."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32, tf.int64],
  }]

  def build_graph(parameters):
    """Build the placeholder_with_default testing graph."""
    const_node = tf.constant(
        [1, 2, 2, 0], shape=[2, 2], dtype=parameters["dtype"])
    input_tensor = tf.placeholder_with_default(
        const_node, shape=[2, 2], name="input")
    out = tf.equal(input_tensor, const_node, name="output")

    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    numpy_type = _TF_TYPE_INFO[parameters["dtype"]][0]
    input_value = np.array([[1, 0], [2, 1]], numpy_type)
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(zip_path, test_parameters, build_graph, build_inputs,
                    expected_tf_success=3)


# Toco binary path provided by the generate rule.
bin_path = None


def main(unused_args):
  global bin_path
  def mkdir_if_not_exist(x):
    if not os.path.isdir(x):
      os.mkdir(x)
      if not os.path.isdir(x):
        raise RuntimeError("Failed to create dir %r" % x)

  opstest_path = os.path.join(FLAGS.output_path)
  mkdir_if_not_exist(opstest_path)

  out = FLAGS.zip_to_output
  bin_path = FLAGS.toco
  # Some zip filenames contain a postfix identifying the conversion mode. The
  # list of valid conversion modes is defined in
  # generated_test_conversion_modes() in build_def.bzl.
  test_function = ("make_%s_tests" % (out.replace(".zip", "").replace(
      "pb2lite", "").replace("toco-flex", "").rstrip("_")))
  if test_function not in globals():
    raise RuntimeError("Can't find a test function to create %r. Tried %r" %
                       (out, test_function))

  # TODO(ahentz): accessing globals() is not very elegant. We should either
  # break this file into multiple tests or use decorator-based registration to
  # avoid using globals().
  globals()[test_function](os.path.join(opstest_path, out))


if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    print("Usage: %s <path out> <zip file to generate>")
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
