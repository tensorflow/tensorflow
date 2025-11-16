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
"""Generate a series of TensorFlow graphs that become tflite test cases.

Usage:

generate_examples <output directory>

bazel run //tensorflow/lite/testing:generate_examples

To more easily debug failures use (or override) the --save_graphdefs flag to
place text proto graphdefs into the generated zip files.
"""

import copy
import datetime
import os
import re
import zipfile

import tensorflow as tf

# TODO(aselle): Disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# pylint: disable=g-import-not-at-top
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# Placeholder for make_dense_image_warp_tests import
# Note: This is a regression test for a bug (b/122651451) that Toco incorrectly
# erases the reduction indices array while it's shared with other ops.
# For verifying https://github.com/tensorflow/tensorflow/issues/23599
from tensorflow.lite.testing.zip_test_utils import get_test_function


class MultiGenState:
  """State of multiple set generation process.

  This state class stores the information needed when generating the examples
  for multiple test set. The stored informations are open archive object to be
  shared, information on test target for current iteration of generation,
  accumulated generation results.
  """

  def __init__(self):
    # Open archive.
    self.archive = None
    # Test name for current generation.
    self.test_name = None
    # Label base path containing the test name.
    # Each of the test data path in the zip archive is derived from this path.
    # If this path is "a/b/c/d.zip", an example of generated test data path
    # is "a/b/c/d_input_type=tf.float32,input_shape=[2,2].inputs".
    # The test runner interpretes the test name of this path as "d".
    # Label base path also should finish with ".zip".
    self.label_base_path = None
    # Zip manifests.
    self.zip_manifest = []
    # Number of all parameters accumulated.
    self.parameter_count = 0


class Options:
  """All options for example generation."""

  def __init__(self):
    # Directory where the outputs will be go.
    self.output_path = None
    # Particular zip to output.
    self.zip_to_output = None
    # If a particular model is affected by a known bug count it as a converter
    # error.
    self.known_bugs_are_errors = False
    # Raise an exception if any converter error is encountered.
    self.ignore_converter_errors = False
    # Include intermediate graphdefs in the output zip files.
    self.save_graphdefs = False
    # Whether the TFLite Flex converter is being used.
    self.run_with_flex = False
    # Whether to generate test cases for edgetpu.
    self.make_edgetpu_tests = False
    # Whether to generate test cases for TF PTQ.
    self.make_tf_ptq_tests = False
    # For TF Quantization only: where conversion for HLO target.
    self.hlo_aware_conversion = True
    # The function to convert a TensorFLow model to TFLite model.
    # See the document for `mlir_convert` function for its required signature.
    self.tflite_convert_function = None
    # A map from regular expression to bug number. Any test failure with label
    # matching the expression will be considered due to the corresponding bug.
    self.known_bugs = {}
    # Make tests by setting TF forward compatibility horizon to the future.
    self.make_forward_compat_test = False
    # No limitation on the number of tests.
    self.no_tests_limit = False
    # Do not create conversion report.
    self.no_conversion_report = False
    # State of multiple test set generation. This stores state values those
    # should be kept and updated while generating examples over multiple
    # test sets.
    # TODO(juhoha): Separate the state from the options.
    self.multi_gen_state = None
    self.mlir_quantizer = False
    # The list of ops' name that should exist in the converted model.
    # This feature is currently only supported in MLIR conversion path.
    # Example of supported ops' name:
    # - "AVERAGE_POOL_2D" for builtin op.
    # - "NumericVerify" for custom op.
    self.expected_ops_in_converted_model = []
    # Whether to skip generating tests with high dimension input shape.
    self.skip_high_dimension_inputs = False
    # Whether to enable DynamicUpdateSlice op.
    self.enable_dynamic_update_slice = False
    # Whether to unrolling batch matmul.
    self.unfold_batchmatmul = False
    # Experimental low bit options
    self.experimental_low_bit_qat = False


def _prepare_dir(options):

  def mkdir_if_not_exist(x):
    if not os.path.isdir(x):
      os.mkdir(x)
      if not os.path.isdir(x):
        raise RuntimeError("Failed to create dir %r" % x)

  opstest_path = os.path.join(options.output_path)
  mkdir_if_not_exist(opstest_path)


def generate_examples(options):
  """Generate examples for a test set.

  Args:
    options: Options containing information to generate examples.

  Raises:
    RuntimeError: if the test function cannot be found.
  """
  _prepare_dir(options)

  out = options.zip_to_output
  # Some zip filenames contain a postfix identifying the conversion mode. The
  # list of valid conversion modes is defined in
  # generated_test_conversion_modes() in build_def.bzl.

  if options.multi_gen_state:
    test_name = options.multi_gen_state.test_name
  else:
    # Remove suffixes to extract the test name from the output name.
    test_name = re.sub(
        r"(_(|with-flex|forward-compat|edgetpu|mlir-quant))?(_xnnpack)?\.zip$",
        "",
        out,
        count=1)

  test_function_name = "make_%s_tests" % test_name
  test_function = get_test_function(test_function_name)
  if test_function is None:
    raise RuntimeError("Can't find a test function to create %r. Tried %r" %
                       (out, test_function_name))
  if options.make_forward_compat_test:
    future_date = datetime.date.today() + datetime.timedelta(days=30)
    with tf.compat.forward_compatibility_horizon(future_date.year,
                                                 future_date.month,
                                                 future_date.day):
      test_function(options)
  else:
    test_function(options)


def generate_multi_set_examples(options, test_sets):
  """Generate examples for test sets.

  Args:
    options: Options containing information to generate examples.
    test_sets: List of the name of test sets to generate examples.
  """
  _prepare_dir(options)

  multi_gen_state = MultiGenState()
  options.multi_gen_state = multi_gen_state

  zip_path = os.path.join(options.output_path, options.zip_to_output)
  with zipfile.PyZipFile(zip_path, "w") as archive:
    multi_gen_state.archive = archive

    for test_name in test_sets:
      # Some generation function can change the value of the options object.
      # To keep the original options for each run, we use shallow copy.
      new_options = copy.copy(options)

      # Remove suffix and set test_name to run proper test generation function.
      multi_gen_state.test_name = re.sub(
          r"(_(|with-flex|forward-compat|mlir-quant))?$",
          "",
          test_name,
          count=1)
      # Set label base path to write test data files with proper path.
      multi_gen_state.label_base_path = os.path.join(
          os.path.dirname(zip_path), test_name + ".zip")

      generate_examples(new_options)

    zipinfo = zipfile.ZipInfo("manifest.txt")
    archive.writestr(zipinfo, "".join(multi_gen_state.zip_manifest),
                     zipfile.ZIP_DEFLATED)
