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
"""Upgrader for Python scripts from 1.* TensorFlow to 2.0 TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast

import pasta
import six

from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import renames_v2
from tensorflow.tools.compatibility import reorders_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # Maps from a function name to a dictionary that describes how to
    # map from an old argument keyword to the new argument keyword.
    # If the new argument is None, it will be removed.
    # Only keyword args are handled, so make sure to also put any function in
    # function_reorders to ensure that all args are made into keywords first.
    self.function_keyword_renames = {
        "tf.gradients": {
            "colocate_gradients_with_ops": None,
        },
        "tf.hessians": {
            "colocate_gradients_with_ops": None,
        },
        "*.minimize": {
            "colocate_gradients_with_ops": None,
        },
        "*.compute_gradients": {
            "colocate_gradients_with_ops": None,
        },
        "tf.cond": {
            "strict": None,
            "fn1": "true_fn",
            "fn2": "false_fn"
        },
        "tf.argmin": {
            "dimension": "axis",
        },
        "tf.argmax": {
            "dimension": "axis",
        },
        "tf.arg_min": {
            "dimension": "axis",
        },
        "tf.arg_max": {
            "dimension": "axis",
        },
        "tf.math.argmin": {
            "dimension": "axis",
        },
        "tf.math.argmax": {
            "dimension": "axis",
        },
        "tf.image.crop_and_resize": {
            "box_ind": "box_indices",
        },
        "tf.image.extract_image_patches": {
            "ksizes": "sizes",
        },
        "tf.extract_image_patches": {
            "ksizes": "sizes",
        },
        "tf.expand_dims": {
            "dim": "axis",
        },
        "tf.batch_to_space": {
            "block_size": "block_shape",
        },
        "tf.space_to_batch": {
            "block_size": "block_shape",
        },
        "tf.nn.space_to_batch": {
            "block_size": "block_shape",
        },
        "tf.constant": {
            "verify_shape": "verify_shape_is_now_always_true",
        },
        "tf.convert_to_tensor": {
            "preferred_dtype": "dtype_hint"
        },
        "tf.nn.softmax_cross_entropy_with_logits": {
            "dim": "axis",
            "_sentinel": None,
        },
        "tf.nn.softmax_cross_entropy_with_logits_v2": {
            "dim": "axis"
        },
        "tf.linalg.l2_normalize": {
            "dim": "axis",
        },
        "tf.linalg.norm": {
            "keep_dims": "keepdims",
        },
        "tf.norm": {
            "keep_dims": "keepdims",
        },
        "tf.load_file_system_library": {
            "library_filename": "library_location",
        },
        "tf.count_nonzero": {
            "input_tensor": "input",
            "keep_dims": "keepdims",
            "reduction_indices": "axis",
        },
        "tf.math.count_nonzero": {
            "input_tensor": "input",
            "keep_dims": "keepdims",
            "reduction_indices": "axis",
        },
        "tf.nn.erosion2d": {
            "kernel": "filters",
            "rates": "dilations",
        },
        "tf.math.l2_normalize": {
            "dim": "axis",
        },
        "tf.math.log_softmax": {
            "dim": "axis",
        },
        "tf.math.softmax": {
            "dim": "axis"
        },
        "tf.nn.l2_normalize": {
            "dim": "axis",
        },
        "tf.nn.log_softmax": {
            "dim": "axis",
        },
        "tf.nn.moments": {
            "keep_dims": "keepdims",
        },
        "tf.nn.pool": {
            "dilation_rate": "dilations"
        },
        "tf.nn.separable_conv2d": {
            "rate": "dilations"
        },
        "tf.nn.depthwise_conv2d": {
            "rate": "dilations"
        },
        "tf.nn.softmax": {
            "dim": "axis"
        },
        "tf.nn.sufficient_statistics": {
            "keep_dims": "keepdims"
        },
        "tf.debugging.assert_all_finite": {
            "t": "x",
            "msg": "message",
        },
        "tf.sparse.add": {
            "thresh": "threshold",
        },
        "tf.sparse_add": {
            "thresh": "threshold",
        },
        "tf.sparse.concat": {
            "concat_dim": "axis",
            "expand_nonconcat_dim": "expand_nonconcat_dims",
        },
        "tf.sparse_concat": {
            "concat_dim": "axis",
            "expand_nonconcat_dim": "expand_nonconcat_dims",
        },
        "tf.sparse.split": {
            "split_dim": "axis",
        },
        "tf.sparse_split": {
            "split_dim": "axis",
        },
        "tf.sparse.reduce_max": {
            "reduction_axes": "axis",
            "keep_dims": "keepdims",
        },
        "tf.sparse_reduce_max": {
            "reduction_axes": "axis",
            "keep_dims": "keepdims",
        },
        "tf.sparse.reduce_sum": {
            "reduction_axes": "axis",
            "keep_dims": "keepdims",
        },
        "tf.sparse_reduce_sum": {
            "reduction_axes": "axis",
            "keep_dims": "keepdims",
        },
        "tf.nn.max_pool_with_argmax": {
            "Targmax": "output_dtype",
        },
        "tf.multinomial": {
            "output_dtype": "dtype",
        },
        "tf.random.multinomial": {
            "output_dtype": "dtype",
        },
        "tf.reverse_sequence": {
            "seq_dim": "seq_axis",
            "batch_dim": "batch_axis",
        },
        "tf.nn.batch_norm_with_global_normalization": {
            "t": "input",
            "m": "mean",
            "v": "variance",
        },
        "tf.nn.dilation2d": {
            "filter": "filters",
            "rates": "dilations",
        },
        "tf.nn.conv3d": {
            "filter": "filters"
        },
        "tf.zeros_like": {
            "tensor": "input",
        },
        "tf.ones_like": {
            "tensor": "input",
        },
        "tf.nn.conv2d_transpose": {
            "value": "input",
            "filter": "filters",
        },
        "tf.nn.conv3d_transpose": {
            "value": "input",
            "filter": "filters",
        },
        "tf.nn.convolution": {
            "filter": "filters",
            "dilation_rate": "dilations",
        },
        "tf.gfile.Exists": {
            "filename": "path",
        },
        "tf.gfile.Remove": {
            "filename": "path",
        },
        "tf.gfile.Stat": {
            "filename": "path",
        },
        "tf.gfile.Glob": {
            "filename": "pattern",
        },
        "tf.gfile.MkDir": {
            "dirname": "path",
        },
        "tf.gfile.MakeDirs": {
            "dirname": "path",
        },
        "tf.gfile.DeleteRecursively": {
            "dirname": "path",
        },
        "tf.gfile.IsDirectory": {
            "dirname": "path",
        },
        "tf.gfile.ListDirectory": {
            "dirname": "path",
        },
        "tf.gfile.Copy": {
            "oldpath": "src",
            "newpath": "dst",
        },
        "tf.gfile.Rename": {
            "oldname": "src",
            "newname": "dst",
        },
        "tf.gfile.Walk": {
            "in_order": "topdown",
        },
        "tf.random.stateless_multinomial": {
            "output_dtype": "dtype",
        },
        "tf.string_to_number": {
            "string_tensor": "input",
        },
        "tf.strings.to_number": {
            "string_tensor": "input",
        },
        "tf.string_to_hash_bucket": {
            "string_tensor": "input",
        },
        "tf.strings.to_hash_bucket": {
            "string_tensor": "input",
        },
        "tf.reduce_all": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_all": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_any": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_any": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_min": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_min": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_max": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_max": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_sum": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_sum": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_mean": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_mean": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_prod": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_prod": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_logsumexp": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.math.reduce_logsumexp": {
            "reduction_indices": "axis",
            "keep_dims": "keepdims",
        },
        "tf.reduce_join": {
            "keep_dims": "keepdims",
            "reduction_indices": "axis"
        },
        "tf.strings.reduce_join": {
            "keep_dims": "keepdims",
            "reduction_indices": "axis"
        },
        "tf.squeeze": {
            "squeeze_dims": "axis",
        },
        "tf.nn.weighted_moments": {
            "keep_dims": "keepdims"
        },
    }

    # pylint: disable=line-too-long
    # Add additional renames not in renames_v2.py here.
    # IMPORTANT: For the renames in here, if you also need to add to
    # function_reorders or function_keyword_renames, use the OLD function name.
    # These renames happen after the arguments have been processed.
    self.manual_symbol_renames = {
        "tf.batch_to_space_nd":
            "tf.batch_to_space",
        "tf.batch_gather":
            "tf.gather",
        "tf.space_to_batch_nd":
            "tf.space_to_batch",
        "tf.nn.space_to_batch":
            "tf.space_to_batch",
        "tf.estimator.inputs":
            "tf.compat.v1.estimator.inputs",
        "tf.extract_image_patches":
            "tf.image.extract_image_patches",
        "tf.gfile.Copy":
            "tf.io.gfile.copy",
        "tf.gfile.DeleteRecursively":
            "tf.io.gfile.rmtree",
        "tf.gfile.Exists":
            "tf.io.gfile.exists",
        "tf.gfile.Glob":
            "tf.io.gfile.glob",
        "tf.gfile.IsDirectory":
            "tf.io.gfile.isdir",
        "tf.gfile.ListDirectory":
            "tf.io.gfile.listdir",
        "tf.gfile.MakeDirs":
            "tf.io.gfile.makedirs",
        "tf.gfile.MkDir":
            "tf.io.gfile.mkdir",
        "tf.gfile.Remove":
            "tf.io.gfile.remove",
        "tf.gfile.Rename":
            "tf.io.gfile.rename",
        "tf.gfile.Stat":
            "tf.io.gfile.stat",
        "tf.gfile.Walk":
            "tf.io.gfile.walk",
        "tf.contrib.data.AUTOTUNE":
            "tf.data.experimental.AUTOTUNE",
        "tf.contrib.data.Counter":
            "tf.data.experimental.Counter",
        "tf.contrib.data.CheckpointInputPipelineHook":
            "tf.data.experimental.CheckpointInputPipelineHook",
        "tf.contrib.data.CsvDataset":
            "tf.data.experimental.CsvDataset",
        "tf.contrib.data.Optional":
            "tf.data.experimental.Optional",
        "tf.contrib.data.RandomDataset":
            "tf.data.experimental.RandomDataset",
        "tf.contrib.data.Reducer":
            "tf.data.experimental.Reducer",
        "tf.contrib.data.SqlDataset":
            "tf.data.experimental.SqlDataset",
        "tf.contrib.data.StatsAggregator":
            "tf.data.experimental.StatsAggregator",
        "tf.contrib.data.TFRecordWriter":
            "tf.data.experimental.TFRecordWriter",
        "tf.contrib.data.assert_element_shape":
            "tf.data.experimental.assert_element_shape",
        "tf.contrib.data.batch_and_drop_remainder":
            "tf.compat.v1.contrib.data.batch_and_drop_remainder",
        "tf.contrib.data.bucket_by_sequence_length":
            "tf.data.experimental.bucket_by_sequence_length",
        "tf.contrib.data.choose_from_datasets":
            "tf.data.experimental.choose_from_datasets",
        "tf.contrib.data.copy_to_device":
            "tf.data.experimental.copy_to_device",
        "tf.contrib.data.dense_to_sparse_batch":
            "tf.data.experimental.dense_to_sparse_batch",
        "tf.contrib.data.enumerate_dataset":
            "tf.data.experimental.enumerate_dataset",
        "tf.contrib.data.get_next_as_optional":
            "tf.data.experimental.get_next_as_optional",
        "tf.contrib.data.get_single_element":
            "tf.data.experimental.get_single_element",
        "tf.contrib.data.group_by_reducer":
            "tf.data.experimental.group_by_reducer",
        "tf.contrib.data.group_by_window":
            "tf.data.experimental.group_by_window",
        "tf.contrib.data.ignore_errors":
            "tf.data.experimental.ignore_errors",
        "tf.contrib.data.latency_stats":
            "tf.data.experimental.latency_stats",
        "tf.contrib.data.make_batched_features_dataset":
            "tf.data.experimental.make_batched_features_dataset",
        "tf.contrib.data.make_csv_dataset":
            "tf.data.experimental.make_csv_dataset",
        "tf.contrib.data.make_saveable_from_iterator":
            "tf.data.experimental.make_saveable_from_iterator",
        "tf.contrib.data.map_and_batch":
            "tf.data.experimental.map_and_batch",
        "tf.contrib.data.padded_batch_and_drop_remainder":
            "tf.compat.v1.contrib.data.padded_batch_and_drop_remainder",
        "tf.contrib.data.parallel_interleave":
            "tf.data.experimental.parallel_interleave",
        "tf.contrib.data.parse_example_dataset":
            "tf.data.experimental.parse_example_dataset",
        "tf.contrib.data.prefetch_to_device":
            "tf.data.experimental.prefetch_to_device",
        "tf.contrib.data.read_batch_features":
            "tf.compat.v1.contrib.data.read_batch_features",
        "tf.contrib.data.reduce_dataset":
            "tf.compat.v1.contrib.data.reduce_dataset",
        "tf.contrib.data.rejection_resample":
            "tf.data.experimental.rejection_resample",
        "tf.contrib.data.sample_from_datasets":
            "tf.data.experimental.sample_from_datasets",
        "tf.contrib.data.scan":
            "tf.data.experimental.scan",
        "tf.contrib.data.set_stats_aggregator":
            "tf.data.experimental.set_stats_aggregator",
        "tf.contrib.data.shuffle_and_repeat":
            "tf.data.experimental.shuffle_and_repeat",
        "tf.contrib.data.sliding_window_batch":
            "tf.compat.v1.contrib.data.sliding_window_batch",
        "tf.contrib.data.sloppy_interleave":
            "tf.compat.v1.contrib.data.sloppy_interleave",
        "tf.contrib.data.unbatch":
            "tf.data.experimental.unbatch",
        "tf.contrib.data.unique":
            "tf.data.experimental.unique",
        "tf.contrib.rnn.RNNCell":
            "tf.nn.rnn_cell.RNNCell",
        "tf.contrib.rnn.LSTMStateTuple":
            "tf.nn.rnn_cell.LSTMStateTuple",
        "tf.contrib.framework.sort":
            "tf.sort",
        "tf.contrib.framework.argsort":
            "tf.argsort",
        "tf.count_nonzero":
            "tf.math.count_nonzero",
        "tf.manip.batch_to_space_nd":
            "tf.batch_to_space",
        "tf.quantize_v2":
            "tf.quantization.quantize",
        "tf.sparse_add":
            "tf.sparse.add",
        "tf.sparse_concat":
            "tf.sparse.concat",
        "tf.sparse_split":
            "tf.sparse.split",
        "tf.sparse_matmul":
            "tf.linalg.matmul",
        "tf.sparse_reduce_sum":
            "tf.sparse.reduce_sum",
        "tf.sparse_reduce_max":
            "tf.sparse.reduce_max",
        "tf.random.stateless_multinomial":
            "tf.random.stateless_categorical",
        "tf.substr":
            "tf.strings.substr",
        "tf.string_to_hash_bucket":
            "tf.strings.to_hash_bucket",
        "tf.string_to_number":
            "tf.strings.to_number",
        "tf.multinomial":
            "tf.random.categorical",
        "tf.random.multinomial":
            "tf.random.categorical",
        "tf.reduce_join":
            "tf.strings.reduce_join",
        "tf.load_file_system_library":
            "tf.load_library",
        "tf.pywrap_tensorflow":
            "tf.compat.v1.pywrap_tensorflow",
        "tf.bincount":
            "tf.math.bincount",
        "tf.confusion_matrix":
            "tf.math.confusion_matrix",
        "tf.train.confusion_matrix":
            "tf.math.confusion_matrix",
        "tf.decode_csv":
            "tf.io.decode_csv",
        "tf.data.Iterator":
            "tf.compat.v1.data.Iterator",
        "tf.parse_example":
            "tf.io.parse_example",
        "tf.parse_single_example":
            "tf.io.parse_single_example",
        "tf.nn.fused_batch_norm":
            "tf.compat.v1.nn.fused_batch_norm",
        "tf.nn.softmax_cross_entropy_with_logits_v2":
            "tf.nn.softmax_cross_entropy_with_logits",
        "tf.losses.Reduction.MEAN":
            "tf.compat.v1.losses.Reduction.MEAN",
        "tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS":
            "tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS",
        "tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS":
            "tf.compat.v1.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS",
        "tf.lite.constants.FLOAT":
            "tf.float32",
        "tf.lite.constants.INT32":
            "tf.int32",
        "tf.lite.constants.INT64":
            "tf.int64",
        "tf.lite.constants.STRING":
            "tf.string",
        "tf.lite.constants.QUANTIZED_UINT8":
            "tf.uint8",
        "tf.arg_max":
            "tf.argmax",
        "tf.arg_min":
            "tf.argmin",
        # tf.nn.ctc_loss is still available in 2.0 but behavior
        # changed significantly.
        "tf.nn.ctc_loss":
            "tf.compat.v1.nn.ctc_loss",
        "tf.zeros_initializer":
            "tf.compat.v1.initializers.zeros",
        "tf.ones_initializer":
            "tf.compat.v1.initializers.ones",
        "tf.constant_initializer":
            "tf.compat.v1.initializers.constant",
        "tf.random_uniform_initializer":
            "tf.compat.v1.initializers.random_uniform",
        "tf.random_normal_initializer":
            "tf.compat.v1.initializers.random_normal",
        "tf.truncated_normal_initializer":
            "tf.compat.v1.initializers.truncated_normal",
        "tf.image.resize_images":
            "tf.image.resize",
        "tf.random_poisson":
            "tf.random.poisson",
        "tf.debugging.assert_greater":
            "tf.compat.v1.debugging.assert_greater",
        "tf.debugging.assert_greater_equal":
            "tf.compat.v1.debugging.assert_greater_equal",
        "tf.debugging.assert_integer":
            "tf.compat.v1.debugging.assert_integer",
        "tf.debugging.assert_less":
            "tf.compat.v1.debugging.assert_less",
        "tf.debugging.assert_less_equal":
            "tf.compat.v1.debugging.assert_less_equal",
        "tf.debugging.assert_near":
            "tf.compat.v1.debugging.assert_near",
        "tf.debugging.assert_negative":
            "tf.compat.v1.debugging.assert_negative",
        "tf.debugging.assert_non_negative":
            "tf.compat.v1.debugging.assert_non_negative",
        "tf.debugging.assert_non_positive":
            "tf.compat.v1.debugging.assert_non_positive",
        "tf.debugging.assert_none_equal":
            "tf.compat.v1.debugging.assert_none_equal",
        "tf.debugging.assert_type":
            "tf.compat.v1.debugging.assert_type",
        "tf.debugging.assert_positive":
            "tf.compat.v1.debugging.assert_positive",
        "tf.debugging.assert_equal":
            "tf.compat.v1.debugging.assert_equal",
        "tf.debugging.assert_scalar":
            "tf.compat.v1.debugging.assert_scalar",
        "tf.assert_equal":
            "tf.compat.v1.assert_equal",
        "tf.assert_less":
            "tf.compat.v1.assert_less",
        "tf.assert_greater":
            "tf.compat.v1.assert_greater",
        "tf.debugging.assert_rank":
            "tf.compat.v1.debugging.assert_rank",
        "tf.debugging.assert_rank_at_least":
            "tf.compat.v1.debugging.assert_rank_at_least",
        "tf.debugging.assert_rank_in":
            "tf.compat.v1.debugging.assert_rank_in",
        "tf.assert_rank":
            "tf.compat.v1.assert_rank",
    }
    # pylint: enable=line-too-long

    # Mapping from function to the new name of the function
    self.symbol_renames = renames_v2.renames
    self.symbol_renames.update(self.manual_symbol_renames)

    # Variables that should be changed to functions.
    self.change_to_function = {}

    # pylint: disable=line-too-long
    # This list should just contain names of functions that had
    # their arguments reordered. After adding a function name to the list
    # run the following to update reorders_v2.py:
    # bazel build tensorflow/tools/compatibility/update:generate_v2_reorders_map
    # bazel-bin/tensorflow/tools/compatibility/update/generate_v2_reorders_map
    # pylint: enable=line-too-long
    self.reordered_function_names = {
        "tf.io.serialize_sparse",
        "tf.io.serialize_many_sparse",
        "tf.argmax",
        "tf.argmin",
        "tf.batch_gather",
        "tf.batch_to_space",
        "tf.cond",
        "tf.nn.space_to_batch",
        "tf.boolean_mask",
        "tf.convert_to_tensor",
        "tf.nn.moments",
        "tf.nn.convolution",
        "tf.nn.crelu",
        "tf.nn.weighted_moments",
        "tf.nn.pool",
        "tf.nn.separable_conv2d",
        "tf.nn.depthwise_conv2d",
        "tf.multinomial",
        "tf.random.multinomial",
        "tf.pad",
        "tf.quantize_v2",
        "tf.feature_column.categorical_column_with_vocabulary_file",
        "tf.shape",
        "tf.size",
        "tf.random.poisson",
        "tf.sparse.add",
        "tf.sparse_add",
        "tf.sparse.concat",
        "tf.sparse_concat",
        "tf.sparse.segment_mean",
        "tf.sparse.segment_sqrt_n",
        "tf.sparse.segment_sum",
        "tf.sparse_matmul",
        "tf.sparse.reduce_max",
        "tf.sparse_reduce_max",
        "tf.io.decode_csv",
        "tf.strings.length",
        "tf.strings.reduce_join",
        "tf.strings.substr",
        "tf.substr",
        "tf.transpose",
        "tf.tuple",
        "tf.parse_example",
        "tf.parse_single_example",
        "tf.io.parse_example",
        "tf.io.parse_single_example",
        "tf.while_loop",
        "tf.reduce_all",
        "tf.math.reduce_all",
        "tf.reduce_any",
        "tf.math.reduce_any",
        "tf.reduce_min",
        "tf.math.reduce_min",
        "tf.reduce_max",
        "tf.math.reduce_max",
        "tf.reduce_sum",
        "tf.math.reduce_sum",
        "tf.reduce_mean",
        "tf.math.reduce_mean",
        "tf.reduce_prod",
        "tf.math.reduce_prod",
        "tf.reduce_logsumexp",
        "tf.math.reduce_logsumexp",
        "tf.reduce_join",
        "tf.confusion_matrix",
        "tf.math.confusion_matrix",
        "tf.math.in_top_k",
        "tf.nn.depth_to_space",
        "tf.nn.embedding_lookup",
        "tf.nn.embedding_lookup_sparse",
        "tf.nn.in_top_k",
        "tf.nn.space_to_depth",
        "tf.linalg.norm",
        "tf.norm",
        "tf.reverse_sequence",
        "tf.sparse_split",
        # tf.nn.softmax_cross_entropy_with_logits *must* be called with
        # keyword arguments. Add keyword arguments in rare case when they
        # are not specified.
        "tf.nn.softmax_cross_entropy_with_logits",
    }

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    self.function_reorders = reorders_v2.reorders

    # Specially handled functions (pasta version)
    # Each transformer is a callable which will be called with the arguments
    #   transformer(parent, node, full_name, name, logs, errors)
    # Where logs and errors are lists to which (line, col, msg) tuples can be
    # appended, full_name is the FQN of the function called (or None if that is
    # unknown), name is the name of the function called (or None is that is
    # unknown). node is an ast.Call node representing this function call, and
    # parent is its parent in the AST.
    # The function may modify node (but not parent), and must return
    # - none, if nothing was modified
    # - node, if node was modified in place (make sure to use
    #   pasta.ast_utils.replace_child to swap out children, otherwise formatting
    #   may get messy)
    # - a replacement for node, if the whole call node was replaced. The caller
    #   will take care of changing parent.
    self.function_transformers = {
        "tf.nn.dropout": self._dropout_transformer,
        "tf.batch_gather": self._batch_gather_transformer,
        "tf.to_bfloat16": self._cast_transformer,
        "tf.to_complex128": self._cast_transformer,
        "tf.to_complex64": self._cast_transformer,
        "tf.to_double": self._cast_transformer,
        "tf.to_float": self._cast_transformer,
        "tf.to_int32": self._cast_transformer,
        "tf.to_int64": self._cast_transformer,
        "tf.nn.softmax_cross_entropy_with_logits":
            self._softmax_cross_entropy_with_logits_transformer,
        "tf.image.resize_area": self._image_resize_transformer,
        "tf.image.resize_bicubic": self._image_resize_transformer,
        "tf.image.resize_bilinear": self._image_resize_transformer,
        "tf.image.resize_nearest_neighbor": self._image_resize_transformer,

    }

    decay_function_comment = (
        "WARNING: <function name> has been changed to return a callable instead"
        " of a tensor when graph building, but its functionality remains "
        "unchanged during eager execution (returns a callable like "
        "before). The converter cannot detect and fix this reliably, so "
        "this usage has been converted to compat.v1 (even though it may already"
        " be correct).\n"
    )

    # TODO(b/118888586): add default value change to update script.
    default_loss_reduction_changed = (
        "WARNING: default value of loss_reduction has been changed to "
        "SUM_OVER_BATCH_SIZE.\n"
    )

    assert_return_type_comment = (
        "WARNING: assert_* functions have been changed to return None, the "
        "data argument has been removed, and arguments have been reordered."
        "\nThe calls have been converted to compat.v1 for safety (even though "
        " they may already have been correct)."
    )

    assert_rank_comment = (
        "WARNING: assert_rank_* functions have been changed to return None, and"
        " the data and summarize arguments have been removed."
        "\nThe calls have been converted to compat.v1 for safety (even though "
        " they may already have been correct)."
    )

    tf_01s_like_no_optimize_comment = (
        "WARNING: tf.zeros_like and tf.ones_like no longer have the optimize "
        "argument in TF 2.0 or after (also, `tensor' argument is renamed to "
        "`input')."
        "\nThe calls have been converted to compat.v1 for safety (even though "
        " they may already have been correct)."
    )

    deprecate_partition_strategy_comment = (
        "WARNING: `partition_strategy` has been removed from `%s` "
        " The 'div' strategy is used by default.")

    initializers_no_dtype_comment = (
        "WARNING: tf.initializers and tf.keras.initializers no longer have the "
        "dtype argument in the constructor or partition_info argument in the "
        "call method in TF 2.0 and after. The only API symbols are now "
        "tf.keras.initializers.* or tf.initializers.*."
        "\nThe calls have been converted to compat.v1 for safety (even though "
        "they may already have been correct).")

    uniform_unit_scaling_initializer_comment = (
        "WARNING: uniform_unit_scaling_initializer has been removed. Please use"
        " tf.initializers.variance_scaling instead with distribution=uniform "
        "to get equivalent behaviour.")

    metrics_comment = (
        "WARNING: tf.metrics have been converted to object oriented versions in"
        " TF 2.0 and after. The metric function calls have been converted to "
        "compat.v1 for backward compatibility. Please update these calls to "
        "the TF 2.0 versions.")

    losses_comment = (
        "WARNING: tf.losses have been converted to object oriented versions in"
        " TF 2.0 and after. The loss function calls have been converted to "
        "compat.v1 for backward compatibility. Please update these calls to "
        "the TF 2.0 versions.")

    export_saved_model_renamed = (
        "(Manual edit required) Please rename the method export_savedmodel() "
        "to export_saved_model(). Two things to note:\n\t(1) The argument "
        "strip_default_attributes has been removed. The function will always "
        "strip the default attributes from ops. If this breaks your code, "
        "please switch to tf.compat.v1.estimator.Estimator.\n\t(2) This change "
        "only effects core estimator. If you are using "
        "tf.contrib.learn.Estimator, please switch to using core estimator.")

    make_initializable_iterator_deprecation = (
        "(Manual edit required) The "
        "`tf.data.Dataset.make_initializable_iterator()` method has been "
        "removed. If you are using the Estimator API, you can return a dataset "
        "directly from your input functions without creating an iterator. "
        "As a last resort, please replace calls to that method on `dataset` "
        "with a call to "
        "`tf.compat.v1.data.make_initializable_iterator(dataset)`.")

    make_one_shot_iterator_deprecation = (
        "(Manual edit required) The "
        "`tf.data.Dataset.make_one_shot_iterator()` method has been "
        "removed. If you are using eager execution, you can iterate over "
        "`dataset` using a Python `for` loop. If you are using the Estimator "
        "API, you can return a dataset directly from your input functions "
        "without creating an iterator. As a last resort, please replace calls "
        "to that method on `dataset` with a call to "
        "`tf.compat.v1.data.make_one_shot_iterator(dataset)`.")

    # Function warnings. <function name> placeholder inside warnings will be
    # replaced by function name.
    # You can use *. to add items which do not check the FQN, and apply to e.g.,
    # methods.
    self.function_warnings = {
        "*.export_savedmodel":
            export_saved_model_renamed,
        "*.make_initializable_iterator":
            make_initializable_iterator_deprecation,
        "*.make_one_shot_iterator":
            make_one_shot_iterator_deprecation,
        "tf.assert_equal":
            assert_return_type_comment,
        "tf.assert_none_equal":
            assert_return_type_comment,
        "tf.assert_negative":
            assert_return_type_comment,
        "tf.assert_positive":
            assert_return_type_comment,
        "tf.assert_non_negative":
            assert_return_type_comment,
        "tf.assert_non_positive":
            assert_return_type_comment,
        "tf.assert_near":
            assert_return_type_comment,
        "tf.assert_less":
            assert_return_type_comment,
        "tf.assert_less_equal":
            assert_return_type_comment,
        "tf.assert_greater":
            assert_return_type_comment,
        "tf.assert_greater_equal":
            assert_return_type_comment,
        "tf.assert_integer":
            assert_return_type_comment,
        "tf.assert_type":
            assert_return_type_comment,
        "tf.assert_scalar":
            assert_return_type_comment,
        "tf.assert_rank":
            assert_rank_comment,
        "tf.assert_rank_at_least":
            assert_rank_comment,
        "tf.assert_rank_in":
            assert_rank_comment,
        "tf.debugging.assert_equal":
            assert_return_type_comment,
        "tf.debugging.assert_greater":
            assert_return_type_comment,
        "tf.debugging.assert_greater_equal":
            assert_return_type_comment,
        "tf.debugging.assert_integer":
            assert_return_type_comment,
        "tf.debugging.assert_less":
            assert_return_type_comment,
        "tf.debugging.assert_less_equal":
            assert_return_type_comment,
        "tf.debugging.assert_near":
            assert_return_type_comment,
        "tf.debugging.assert_negative":
            assert_return_type_comment,
        "tf.debugging.assert_non_negative":
            assert_return_type_comment,
        "tf.debugging.assert_non_positive":
            assert_return_type_comment,
        "tf.debugging.assert_none_equal":
            assert_return_type_comment,
        "tf.debugging.assert_positive":
            assert_return_type_comment,
        "tf.debugging.assert_type":
            assert_return_type_comment,
        "tf.debugging.assert_scalar":
            assert_return_type_comment,
        "tf.debugging.assert_rank":
            assert_rank_comment,
        "tf.debugging.assert_rank_at_least":
            assert_rank_comment,
        "tf.debugging.assert_rank_in":
            assert_rank_comment,
        "tf.device":
            "tf.device no longer takes function as an argument. "
            "'devide_name_or_function' argument has been renamed to "
            "'device_name'.",
        "tf.flags":
            "tf.flags has been removed, please use the argparse or absl"
            " module if you need command line parsing.",
        "tf.train.exponential_decay":
            decay_function_comment,
        "tf.train.piecewise_constant_decay":
            decay_function_comment,
        "tf.train.polynomial_decay":
            decay_function_comment,
        "tf.train.natural_exp_decay":
            decay_function_comment,
        "tf.train.inverse_time_decay":
            decay_function_comment,
        "tf.train.cosine_decay":
            decay_function_comment,
        "tf.train.cosine_decay_restarts":
            decay_function_comment,
        "tf.train.linear_cosine_decay":
            decay_function_comment,
        "tf.train.noisy_linear_cosine_decay":
            decay_function_comment,
        "tf.estimator.LinearClassifier":
            default_loss_reduction_changed,
        "tf.estimator.LinearRegressor":
            default_loss_reduction_changed,
        "tf.estimator.DNNLinearCombinedClassifier":
            default_loss_reduction_changed,
        "tf.estimator.DNNLinearCombinedRegressor":
            default_loss_reduction_changed,
        "tf.estimator.DNNRegressor":
            default_loss_reduction_changed,
        "tf.estimator.DNNClassifier":
            default_loss_reduction_changed,
        "tf.estimator.BaselineClassifier":
            default_loss_reduction_changed,
        "tf.estimator.BaselineRegressor":
            default_loss_reduction_changed,
        "tf.nn.conv1d":
            "WARNING: use_cudnn_on_gpu argument has been removed and \"value\""
            " was renamed to \"input\"",
        "tf.nn.conv2d":
            "WARNING: use_cudnn_on_gpu argument has been removed and "
            "\"filter\" was renamed to \"filters\"",
        "tf.nn.conv2d_backprop_filter":
            "WARNING: use_cudnn_on_gpu argument has been removed",
        "tf.nn.conv2d_backprop_input":
            "WARNING: use_cudnn_on_gpu argument has been removed and "
            "\"filter\" was renamed to \"filters\"",
        "tf.nn.erosion2d":
            "WARNING: <function name> now requires a data_format argument",
        "tf.nn.nce_loss":
            deprecate_partition_strategy_comment % "tf.nn.nce_loss",
        "tf.nn.safe_embedding_lookup_sparse":
            deprecate_partition_strategy_comment %
            "tf.nn.safe_embedding_lookup_sparse",
        "tf.nn.sampled_softmax_loss":
            deprecate_partition_strategy_comment % "tf.nn.sampled_softmax_loss",
        "tf.zeros_like":
            tf_01s_like_no_optimize_comment,
        "tf.ones_like":
            tf_01s_like_no_optimize_comment,
        "tf.nn.embedding_lookup":
            "WARNING: validate_indices argument has been removed.",
        "tf.while_loop":
            "tf.while_loop no longer takes 'return_same_structure' argument. "
            "'return_same_structure' now defaults to True. Also, 'name'"
            "argument is now the last argument.",
        "tf.image.sample_distorted_bounding_box":
            "tf.image.sample_distorted_bounding_box no longer takes 'seed2' "
            "argument.",
        "tf.nn.ctc_beam_search_decoder":
            "tf.nn.ctc_beam_search_decoder no longer takes 'merge_repeated' "
            "argument. 'merge_repeated' now defaults to False.",
        "tf.nn.fractional_avg_pool":
            "tf.nn.fractional_avg_pool no longer takes 'seed2' and "
            "'deterministic' arguments. Now it takes a single 'seed' arg. If "
            "'seed' is zero, the execution is random and deterministic "
            "otherwise",
        "tf.nn.fractional_max_pool":
            "tf.nn.fractional_max_pool no longer takes 'seed2' and "
            "'deterministic' arguments. Now it takes a single 'seed' arg. If "
            "'seed' is zero, the execution is random and deterministic "
            "otherwise",
        "tf.test.assert_equal_graph_def":
            "tf.assert_equal_graph_def no longer takes 'checkpoint_v2' "
            "argument. 'checkpoint_v2' now defaults to True.",
        "tf.keras.initializers.Zeros":
            initializers_no_dtype_comment,
        "tf.keras.initializers.zeros":
            initializers_no_dtype_comment,
        "tf.keras.initializers.Ones":
            initializers_no_dtype_comment,
        "tf.keras.initializers.ones":
            initializers_no_dtype_comment,
        "tf.keras.initializers.Constant":
            initializers_no_dtype_comment,
        "tf.keras.initializers.constant":
            initializers_no_dtype_comment,
        "tf.keras.initializers.VarianceScaling":
            initializers_no_dtype_comment,
        "tf.keras.initializers.Orthogonal":
            initializers_no_dtype_comment,
        "tf.keras.initializers.orthogonal":
            initializers_no_dtype_comment,
        "tf.keras.initializers.Identity":
            initializers_no_dtype_comment,
        "tf.keras.initializers.identity":
            initializers_no_dtype_comment,
        "tf.keras.initializers.glorot_uniform":
            initializers_no_dtype_comment,
        "tf.keras.initializers.glorot_normal":
            initializers_no_dtype_comment,
        "tf.initializers.zeros":
            initializers_no_dtype_comment,
        "tf.zeros_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.ones":
            initializers_no_dtype_comment,
        "tf.ones_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.constant":
            initializers_no_dtype_comment,
        "tf.constant_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.random_uniform":
            initializers_no_dtype_comment,
        "tf.random_uniform_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.random_normal":
            initializers_no_dtype_comment,
        "tf.random_normal_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.truncated_normal":
            initializers_no_dtype_comment,
        "tf.truncated_normal_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.variance_scaling":
            initializers_no_dtype_comment,
        "tf.variance_scaling_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.orthogonal":
            initializers_no_dtype_comment,
        "tf.orthogonal_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.identity":
            initializers_no_dtype_comment,
        "tf.glorot_uniform_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.glorot_uniform":
            initializers_no_dtype_comment,
        "tf.glorot_normal_initializer":
            initializers_no_dtype_comment,
        "tf.initializers.glorot_normal":
            initializers_no_dtype_comment,
        "tf.initializers.uniform_unit_scaling":
            uniform_unit_scaling_initializer_comment,
        "tf.uniform_unit_scaling_initializer":
            uniform_unit_scaling_initializer_comment,
        "tf.losses.absolute_difference":
            losses_comment,
        "tf.losses.add_loss":
            losses_comment,
        "tf.losses.compute_weighted_loss":
            losses_comment,
        "tf.losses.cosine_distance":
            losses_comment,
        "tf.losses.get_losses":
            losses_comment,
        "tf.losses.get_regularization_loss":
            losses_comment,
        "tf.losses.get_regularization_losses":
            losses_comment,
        "tf.losses.get_total_loss":
            losses_comment,
        "tf.losses.hinge_loss":
            losses_comment,
        "tf.losses.huber_loss":
            losses_comment,
        "tf.losses.log_loss":
            losses_comment,
        "tf.losses.mean_pairwise_squared_error":
            losses_comment,
        "tf.losses.mean_squared_error":
            losses_comment,
        "tf.losses.sigmoid_cross_entropy":
            losses_comment,
        "tf.losses.softmax_cross_entropy":
            losses_comment,
        "tf.losses.sparse_softmax_cross_entropy":
            losses_comment,
        "tf.metrics.accuracy":
            metrics_comment,
        "tf.metrics.auc":
            metrics_comment,
        "tf.metrics.average_precision_at_k":
            metrics_comment,
        "tf.metrics.false_negatives":
            metrics_comment,
        "tf.metrics.false_negatives_at_thresholds":
            metrics_comment,
        "tf.metrics.false_positives":
            metrics_comment,
        "tf.metrics.false_positives_at_thresholds":
            metrics_comment,
        "tf.metrics.mean":
            metrics_comment,
        "tf.metrics.mean_absolute_error":
            metrics_comment,
        "tf.metrics.mean_cosine_distance":
            metrics_comment,
        "tf.metrics.mean_iou":
            metrics_comment,
        "tf.metrics.mean_per_class_accuracy":
            metrics_comment,
        "tf.metrics.mean_relative_error":
            metrics_comment,
        "tf.metrics.mean_squared_error":
            metrics_comment,
        "tf.metrics.mean_tensor":
            metrics_comment,
        "tf.metrics.percentage_below":
            metrics_comment,
        "tf.metrics.precision":
            metrics_comment,
        "tf.metrics.precision_at_k":
            metrics_comment,
        "tf.metrics.precision_at_thresholds":
            metrics_comment,
        "tf.metrics.precision_at_top_k":
            metrics_comment,
        "tf.metrics.recall":
            metrics_comment,
        "tf.metrics.recall_at_k":
            metrics_comment,
        "tf.metrics.recall_at_thresholds":
            metrics_comment,
        "tf.metrics.recall_at_top_k":
            metrics_comment,
        "tf.metrics.root_mean_squared_error":
            metrics_comment,
        "tf.metrics.sensitivity_at_specificity":
            metrics_comment,
        "tf.metrics.sparse_average_precision_at_k":
            metrics_comment,
        "tf.metrics.sparse_precision_at_k":
            metrics_comment,
        "tf.metrics.specificity_at_sensitivity":
            metrics_comment,
        "tf.metrics.true_negatives":
            metrics_comment,
        "tf.metrics.true_negatives_at_thresholds":
            metrics_comment,
        "tf.metrics.true_positives":
            metrics_comment,
        "tf.metrics.true_positives_at_thresholds":
            metrics_comment,
    }

    # Warnings that are emitted only if a specific arg is found.
    self.function_arg_warnings = {
        "tf.gradients": {
            ("colocate_gradients_with_ops", 4):
                "tf.gradients no longer takes "
                "'colocate_gradients_with_ops' argument, it behaves as if it "
                "was set to True.",
        },
        "*.minimize": {
            ("colocate_gradients_with_ops", 5):
                "Optimizer.minimize no longer takes "
                "'colocate_gradients_with_ops' argument, it behaves as if it "
                "was set to True.",
        },
        "*.compute_gradients": {
            ("colocate_gradients_with_ops", 4):
                "Optimizer.compute_gradients no "
                "longer takes 'colocate_gradients_with_ops' argument, it "
                "behaves as if it was set to True.",
        },
        "tf.cond": {
            ("strict", 3):
                "tf.cond no longer takes 'strict' argument, it behaves as "
                "if was set to True."
        },
    }

    self.symbol_renames = {
        name: new_name
        for name, new_name in self.symbol_renames.items()
    }

  @staticmethod
  def _dropout_transformer(parent, node, full_name, name, logs, errors):
    def _replace_keep_prob_node(parent, old_value):
      """Replaces old_value with 1-(old_value)."""
      one = ast.Num(n=1)
      one.lineno = 0
      one.col_offset = 0
      new_value = ast.BinOp(left=one, op=ast.Sub(),
                            right=old_value)
      # This copies the prefix and suffix on old_value to new_value.
      pasta.ast_utils.replace_child(parent, old_value, new_value)
      ast.copy_location(new_value, old_value)
      # Put parentheses around keep_prob.value (and remove the old prefix/
      # suffix, they should only be around new_value).
      pasta.base.formatting.set(old_value, "prefix", "(")
      pasta.base.formatting.set(old_value, "suffix", ")")

    # Check if we have a keep_prob keyword arg
    for keep_prob in node.keywords:
      if keep_prob.arg == "keep_prob":
        logs.append((node.lineno, node.col_offset,
                     "Changing keep_prob arg of tf.nn.dropout to rate, and "
                     "recomputing value. Please check this transformation.\n"))
        keep_prob.arg = "rate"
        _replace_keep_prob_node(keep_prob, keep_prob.value)
        return node

    # Maybe it was a positional arg
    if len(node.args) < 2:
      errors.append((node.lineno, node.col_offset,
                     "ERROR: tf.nn.dropout called without arguments, so "
                     "automatic fix was disabled. tf.nn.dropout has changed "
                     "the semantics of the second argument."))
    else:
      _replace_keep_prob_node(node, node.args[1])
      logs.append((node.lineno, node.col_offset,
                   "Changing keep_prob arg of tf.nn.dropout to rate, and "
                   "recomputing value.\n"))
      errors.append((node.lineno, node.col_offset,
                     "WARNING: tf.nn.dropout has changed the semantics of the "
                     "second argument. Please check the applied transformation."
                    ))
      return node

  @staticmethod
  def _cast_transformer(parent, node, full_name, name, logs, errors):
    """Transforms to_int and to_float to cast(..., dtype=...)."""

    # Find out the dtype to cast to from the function name
    dtype_str = name[3:]
    # Special cases where the full dtype is not given
    if dtype_str == "float":
      dtype_str = "float32"
    elif dtype_str == "double":
      dtype_str = "float64"
    new_arg = ast.keyword(arg="dtype",
                          value=ast.Attribute(value=ast.Name(id="tf",
                                                             ctx=ast.Load()),
                                              attr=dtype_str, ctx=ast.Load()))
    # Ensures a valid transformation when a positional name arg is given
    if len(node.args) == 2:
      name_arg = ast.keyword(arg="name",
                             value=node.args[-1])
      node.args = node.args[:-1]
      node.keywords.append(name_arg)

    # Python3 ast requires the args for the Attribute, but codegen will mess up
    # the arg order if we just set them to 0.
    new_arg.value.lineno = node.lineno
    new_arg.value.col_offset = node.col_offset+100

    node.keywords.append(new_arg)
    if isinstance(node.func, ast.Attribute):
      node.func.attr = "cast"
    else:
      assert isinstance(node.func, ast.Name)
      node.func.id = "cast"

    logs.append((node.lineno, node.col_offset,
                 "Changed %s call to tf.cast(..., dtype=tf.%s)." % (full_name,
                                                                    dtype_str)))
    return node

  @staticmethod
  def _softmax_cross_entropy_with_logits_transformer(
      parent, node, full_name, name, logs, errors):
    def _wrap_label(parent, old_value):
      """Wrap labels with tf.stop_gradient."""
      if six.PY3:
        new_value = ast.Call(
            ast.Name(id="tf.stop_gradient", ctx=ast.Load()),
            [old_value], [])
      else:
        new_value = ast.Call(
            ast.Name(id="tf.stop_gradient", ctx=ast.Load()),
            [old_value], [], None, None)

      # This copies the prefix and suffix on old_value to new_value.
      pasta.ast_utils.replace_child(parent, old_value, new_value)
      ast.copy_location(new_value, old_value)

    # Check if we have a labels keyword arg
    for karg in node.keywords:
      if karg.arg == "labels":
        logs.append((node.lineno, node.col_offset,
                     "Changing labels arg of "
                     "tf.nn.softmax_cross_entropy_with_logits to "
                     "tf.stop_gradient(labels). Please check this "
                     "transformation.\n"))
        _wrap_label(karg, karg.value)
        return node
    return node

  @staticmethod
  def _batch_gather_transformer(parent, node, full_name, name, logs, errors):
    # Check if the call already has a batch_dims argument
    if any([kw.arg == "batch_dims" for kw in node.keywords]):
      logs.append((node.lineno, node.col_offset, "tf.batch_gather already has "
                   "batch_dims argument. Neat."))
      return None

    minus_one = ast.Num(n=-1)
    minus_one.lineno = 0
    minus_one.col_offset = 0
    new_arg = ast.keyword("batch_dims", minus_one)
    node.keywords.append(new_arg)
    logs.append((node.lineno, node.col_offset,
                 "Added keyword argument batch_dims=-1 to tf.batch_gather."))
    return node

  @staticmethod
  def _image_resize_transformer(parent, node, full_name, name, logs, errors):
    """Transforms image.resize_* to image.resize(..., method=*, ...)."""

    resize_method = name[7:].upper()
    new_arg = ast.keyword(arg="method",
                          value=ast.Attribute(
                              value=ast.Attribute(
                                  value=ast.Attribute(
                                      value=ast.Name(id="tf", ctx=ast.Load()),
                                      attr="image", ctx=ast.Load()),
                                  attr="ResizeMethod", ctx=ast.Load()),
                              attr=resize_method, ctx=ast.Load()))

    # Ensures a valid transformation when a positional name arg is given
    if len(node.args) == 4:
      pos_arg = ast.keyword(arg="preserve_aspect_ratio",
                            value=node.args[-1])
      node.args = node.args[:-1]
      node.keywords.append(pos_arg)
    if len(node.args) == 3:
      pos_arg = ast.keyword(arg="align_corners",
                            value=node.args[-1])
      node.args = node.args[:-1]
      node.keywords.append(pos_arg)

    # Python3 ast requires the args for the Attribute, but codegen will mess up
    # the arg order if we just set them to 0.
    new_arg.value.lineno = node.lineno
    new_arg.value.col_offset = node.col_offset+100

    node.keywords.append(new_arg)
    if isinstance(node.func, ast.Attribute):
      node.func.attr = "resize"
    else:
      assert isinstance(node.func, ast.Name)
      node.func.id = "resize"

    logs.append((node.lineno, node.col_offset,
                 "Changed %s call to tf.image.resize(..., "
                 "method=tf.image.ResizeMethod.%s)." % (full_name,
                                                        resize_method)))
    return node
