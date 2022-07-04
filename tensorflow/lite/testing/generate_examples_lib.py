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

import tensorflow.compat.v1 as tf

# TODO(aselle): Disable GPU for now
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# pylint: disable=g-import-not-at-top
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
from tensorflow.lite.testing.op_tests.abs import make_abs_tests
from tensorflow.lite.testing.op_tests.add_n import make_add_n_tests
from tensorflow.lite.testing.op_tests.arg_min_max import make_arg_min_max_tests
from tensorflow.lite.testing.op_tests.batch_to_space_nd import make_batch_to_space_nd_tests
from tensorflow.lite.testing.op_tests.batchmatmul import make_batchmatmul_tests
from tensorflow.lite.testing.op_tests.binary_op import make_add_tests, make_div_tests, make_sub_tests, make_mul_tests, make_pow_tests, make_floor_div_tests, make_floor_mod_tests, make_squared_difference_tests
from tensorflow.lite.testing.op_tests.broadcast_args import make_broadcast_args_tests
from tensorflow.lite.testing.op_tests.broadcast_gradient_args import make_broadcast_gradient_args_tests
from tensorflow.lite.testing.op_tests.broadcast_to import make_broadcast_to_tests
from tensorflow.lite.testing.op_tests.cast import make_cast_tests
from tensorflow.lite.testing.op_tests.ceil import make_ceil_tests
from tensorflow.lite.testing.op_tests.complex_abs import make_complex_abs_tests
from tensorflow.lite.testing.op_tests.concat import make_concat_tests
from tensorflow.lite.testing.op_tests.cond import make_cond_tests
from tensorflow.lite.testing.op_tests.constant import make_constant_tests
from tensorflow.lite.testing.op_tests.control_dep import make_control_dep_tests
from tensorflow.lite.testing.op_tests.conv import make_conv_tests
from tensorflow.lite.testing.op_tests.conv2d_transpose import make_conv2d_transpose_tests
from tensorflow.lite.testing.op_tests.conv3d import make_conv3d_tests
from tensorflow.lite.testing.op_tests.conv3d_transpose import make_conv3d_transpose_tests
from tensorflow.lite.testing.op_tests.conv_activation import make_conv_relu_tests, make_conv_relu1_tests, make_conv_relu6_tests
from tensorflow.lite.testing.op_tests.conv_bias_activation import make_conv_bias_relu6_tests
from tensorflow.lite.testing.op_tests.conv_to_depthwiseconv_with_shared_weights import make_conv_to_depthwiseconv_with_shared_weights_tests
from tensorflow.lite.testing.op_tests.conv_with_shared_weights import make_conv_with_shared_weights_tests
from tensorflow.lite.testing.op_tests.cos import make_cos_tests
from tensorflow.lite.testing.op_tests.cumsum import make_cumsum_tests
# Placeholder for make_dense_image_warp_tests import
from tensorflow.lite.testing.op_tests.depth_to_space import make_depth_to_space_tests
from tensorflow.lite.testing.op_tests.depthwiseconv import make_depthwiseconv_tests
from tensorflow.lite.testing.op_tests.dynamic_rnn import make_dynamic_rnn_tests
from tensorflow.lite.testing.op_tests.dynamic_update_slice import make_dynamic_update_slice_tests
from tensorflow.lite.testing.op_tests.einsum import make_einsum_tests
from tensorflow.lite.testing.op_tests.elementwise import make_sin_tests, make_log_tests, make_sqrt_tests, make_rsqrt_tests, make_square_tests
from tensorflow.lite.testing.op_tests.elu import make_elu_tests
from tensorflow.lite.testing.op_tests.embedding_lookup import make_embedding_lookup_tests
from tensorflow.lite.testing.op_tests.equal import make_equal_tests
from tensorflow.lite.testing.op_tests.exp import make_exp_tests
from tensorflow.lite.testing.op_tests.expand_dims import make_expand_dims_tests
from tensorflow.lite.testing.op_tests.expm1 import make_expm1_tests
from tensorflow.lite.testing.op_tests.eye import make_eye_tests
from tensorflow.lite.testing.op_tests.fill import make_fill_tests
from tensorflow.lite.testing.op_tests.floor import make_floor_tests
from tensorflow.lite.testing.op_tests.fully_connected import make_fully_connected_tests
from tensorflow.lite.testing.op_tests.fused_batch_norm import make_fused_batch_norm_tests
from tensorflow.lite.testing.op_tests.gather import make_gather_tests
from tensorflow.lite.testing.op_tests.gather_nd import make_gather_nd_tests
from tensorflow.lite.testing.op_tests.gather_with_constant import make_gather_with_constant_tests
from tensorflow.lite.testing.op_tests.gelu import make_gelu_tests
from tensorflow.lite.testing.op_tests.global_batch_norm import make_global_batch_norm_tests
from tensorflow.lite.testing.op_tests.greater import make_greater_tests
from tensorflow.lite.testing.op_tests.greater_equal import make_greater_equal_tests
from tensorflow.lite.testing.op_tests.hardswish import make_hardswish_tests
from tensorflow.lite.testing.op_tests.identify_dilated_conv import make_identify_dilated_conv_tests
from tensorflow.lite.testing.op_tests.identify_dilated_conv1d import make_identify_dilated_conv1d_tests
from tensorflow.lite.testing.op_tests.identity import make_identity_tests
from tensorflow.lite.testing.op_tests.imag import make_imag_tests
from tensorflow.lite.testing.op_tests.irfft2d import make_irfft2d_tests
from tensorflow.lite.testing.op_tests.is_finite import make_is_finite_tests
from tensorflow.lite.testing.op_tests.l2norm import make_l2norm_tests
# Note: This is a regression test for a bug (b/122651451) that Toco incorrectly
# erases the reduction indices array while it's shared with other ops.
from tensorflow.lite.testing.op_tests.l2norm_shared_epsilon import make_l2norm_shared_epsilon_tests
from tensorflow.lite.testing.op_tests.leaky_relu import make_leaky_relu_tests
from tensorflow.lite.testing.op_tests.less import make_less_tests
from tensorflow.lite.testing.op_tests.less_equal import make_less_equal_tests
from tensorflow.lite.testing.op_tests.local_response_norm import make_local_response_norm_tests
from tensorflow.lite.testing.op_tests.log_softmax import make_log_softmax_tests
from tensorflow.lite.testing.op_tests.logic import make_logical_or_tests, make_logical_and_tests, make_logical_xor_tests
from tensorflow.lite.testing.op_tests.lstm import make_lstm_tests
from tensorflow.lite.testing.op_tests.matrix_diag import make_matrix_diag_tests
from tensorflow.lite.testing.op_tests.matrix_set_diag import make_matrix_set_diag_tests
from tensorflow.lite.testing.op_tests.max_pool_with_argmax import make_max_pool_with_argmax_tests
from tensorflow.lite.testing.op_tests.maximum import make_maximum_tests
from tensorflow.lite.testing.op_tests.minimum import make_minimum_tests
from tensorflow.lite.testing.op_tests.mirror_pad import make_mirror_pad_tests
from tensorflow.lite.testing.op_tests.multinomial import make_multinomial_tests
from tensorflow.lite.testing.op_tests.nearest_upsample import make_nearest_upsample_tests
from tensorflow.lite.testing.op_tests.neg import make_neg_tests
from tensorflow.lite.testing.op_tests.not_equal import make_not_equal_tests
from tensorflow.lite.testing.op_tests.one_hot import make_one_hot_tests
from tensorflow.lite.testing.op_tests.pack import make_pack_tests
from tensorflow.lite.testing.op_tests.pad import make_pad_tests
from tensorflow.lite.testing.op_tests.padv2 import make_padv2_tests
from tensorflow.lite.testing.op_tests.parse_example import make_parse_example_tests
from tensorflow.lite.testing.op_tests.placeholder_with_default import make_placeholder_with_default_tests
from tensorflow.lite.testing.op_tests.pool import make_l2_pool_tests, make_avg_pool_tests, make_max_pool_tests
from tensorflow.lite.testing.op_tests.pool3d import make_avg_pool3d_tests
from tensorflow.lite.testing.op_tests.pool3d import make_max_pool3d_tests
from tensorflow.lite.testing.op_tests.prelu import make_prelu_tests
from tensorflow.lite.testing.op_tests.random_standard_normal import make_random_standard_normal_tests
from tensorflow.lite.testing.op_tests.random_uniform import make_random_uniform_tests
from tensorflow.lite.testing.op_tests.range import make_range_tests
from tensorflow.lite.testing.op_tests.rank import make_rank_tests
from tensorflow.lite.testing.op_tests.real import make_real_tests
from tensorflow.lite.testing.op_tests.reciprocal import make_reciprocal_tests
from tensorflow.lite.testing.op_tests.reduce import make_mean_tests, make_sum_tests, make_reduce_prod_tests, make_reduce_max_tests, make_reduce_min_tests, make_reduce_any_tests, make_reduce_all_tests
from tensorflow.lite.testing.op_tests.relu import make_relu_tests
from tensorflow.lite.testing.op_tests.relu1 import make_relu1_tests
from tensorflow.lite.testing.op_tests.relu6 import make_relu6_tests
from tensorflow.lite.testing.op_tests.reshape import make_reshape_tests
from tensorflow.lite.testing.op_tests.resize_bilinear import make_resize_bilinear_tests
from tensorflow.lite.testing.op_tests.resize_nearest_neighbor import make_resize_nearest_neighbor_tests
# For verifying https://github.com/tensorflow/tensorflow/issues/23599
from tensorflow.lite.testing.op_tests.resolve_constant_strided_slice import make_resolve_constant_strided_slice_tests
from tensorflow.lite.testing.op_tests.reverse_sequence import make_reverse_sequence_tests
from tensorflow.lite.testing.op_tests.reverse_v2 import make_reverse_v2_tests
from tensorflow.lite.testing.op_tests.rfft import make_rfft_tests
from tensorflow.lite.testing.op_tests.rfft2d import make_rfft2d_tests
from tensorflow.lite.testing.op_tests.roll import make_roll_tests
from tensorflow.lite.testing.op_tests.roll import make_roll_with_constant_tests
from tensorflow.lite.testing.op_tests.round import make_round_tests
from tensorflow.lite.testing.op_tests.scatter_nd import make_scatter_nd_tests
from tensorflow.lite.testing.op_tests.segment_sum import make_segment_sum_tests
from tensorflow.lite.testing.op_tests.shape import make_shape_tests
from tensorflow.lite.testing.op_tests.shape_to_strided_slice import make_shape_to_strided_slice_tests
from tensorflow.lite.testing.op_tests.sigmoid import make_sigmoid_tests
from tensorflow.lite.testing.op_tests.slice import make_slice_tests
from tensorflow.lite.testing.op_tests.softmax import make_softmax_tests
from tensorflow.lite.testing.op_tests.softplus import make_softplus_tests
from tensorflow.lite.testing.op_tests.space_to_batch_nd import make_space_to_batch_nd_tests
from tensorflow.lite.testing.op_tests.space_to_depth import make_space_to_depth_tests
from tensorflow.lite.testing.op_tests.sparse_to_dense import make_sparse_to_dense_tests
from tensorflow.lite.testing.op_tests.split import make_split_tests
from tensorflow.lite.testing.op_tests.splitv import make_splitv_tests
from tensorflow.lite.testing.op_tests.squeeze import make_squeeze_tests
from tensorflow.lite.testing.op_tests.squeeze_transpose import make_squeeze_transpose_tests
from tensorflow.lite.testing.op_tests.static_hashtable import make_static_hashtable_tests
from tensorflow.lite.testing.op_tests.static_rnn_with_control_flow_v2 import make_static_rnn_with_control_flow_v2_tests
from tensorflow.lite.testing.op_tests.stft import make_stft_tests
from tensorflow.lite.testing.op_tests.strided_slice import make_strided_slice_tests, make_strided_slice_1d_exhaustive_tests
from tensorflow.lite.testing.op_tests.strided_slice_np_style import make_strided_slice_np_style_tests
from tensorflow.lite.testing.op_tests.tanh import make_tanh_tests
from tensorflow.lite.testing.op_tests.tensor_list_concat import make_tensor_list_concat_tests
from tensorflow.lite.testing.op_tests.tensor_list_dynamic_shape import make_tensor_list_dynamic_shape_tests
from tensorflow.lite.testing.op_tests.tensor_list_get_item import make_tensor_list_get_item_tests
from tensorflow.lite.testing.op_tests.tensor_list_length import make_tensor_list_length_tests
from tensorflow.lite.testing.op_tests.tensor_list_resize import make_tensor_list_resize_tests
from tensorflow.lite.testing.op_tests.tensor_list_set_item import make_tensor_list_set_item_tests
from tensorflow.lite.testing.op_tests.tensor_scatter_add import make_tensor_scatter_add_tests
from tensorflow.lite.testing.op_tests.tensor_scatter_update import make_tensor_scatter_update_tests
from tensorflow.lite.testing.op_tests.tile import make_tile_tests
from tensorflow.lite.testing.op_tests.topk import make_topk_tests
from tensorflow.lite.testing.op_tests.transpose import make_transpose_tests
from tensorflow.lite.testing.op_tests.transpose_conv import make_transpose_conv_tests
from tensorflow.lite.testing.op_tests.unfused_gru import make_unfused_gru_tests
from tensorflow.lite.testing.op_tests.unique import make_unique_tests
from tensorflow.lite.testing.op_tests.unpack import make_unpack_tests
from tensorflow.lite.testing.op_tests.unroll_batch_matmul import make_unroll_batch_matmul_tests
from tensorflow.lite.testing.op_tests.where import make_where_tests
from tensorflow.lite.testing.op_tests.where_v2 import make_where_v2_tests
from tensorflow.lite.testing.op_tests.while_loop import make_while_tests
from tensorflow.lite.testing.op_tests.zeros_like import make_zeros_like_tests

from tensorflow.lite.testing.zip_test_utils import get_test_function


class MultiGenState(object):
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


class Options(object):
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
    # Whether to disable unrolling batch matmul.
    self.disable_batchmatmul_unfold = False


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
