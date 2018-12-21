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

import re

from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import renames_v2
from tensorflow.tools.compatibility import reorders_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # Maps from a function name to a dictionary that describes how to
    # map from an old argument keyword to the new argument keyword.
    self.function_keyword_renames = {
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
    }

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    self.function_reorders = reorders_v2.reorders

    # Specially handled functions.
    self.function_handle = {
        "tf.batch_gather": self._batch_gather_handler,
        "tf.nn.dropout": self._dropout_handler,
        "tf.gradients": self._colocate_handler("tf.gradients"),
        "*.minimize": self._colocate_handler("Optimizer.minimize"),
        "*.compute_gradients":
            self._colocate_handler("Optimizer.compute_gradients"),
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

    # Function warnings. <function name> placeholder inside warnings will be
    # replaced by function name.
    self.function_warnings = {
        "tf.assert_greater":
            assert_return_type_comment,
        "tf.assert_equal":
            assert_return_type_comment,
        "tf.assert_less":
            assert_return_type_comment,
        "tf.assert_rank":
            assert_rank_comment,
        "tf.cond": "tf.cond no longer takes 'strict'. "
                   "Now 'strict' defaults to True."
                   "fn1/fn2 arguments are replaced by true_fn/false_fn.",
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
        "tf.debugging.assert_rank":
            assert_rank_comment,
        "tf.debugging.assert_rank_at_least":
            assert_rank_comment,
        "tf.debugging.assert_rank_in":
            assert_rank_comment,
        "tf.device": "tf.device no longer takes function as an argument. "
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
        "tf.hessians": "tf.hessians no longer takes "
                       "'colocate_gradients_with_ops' argument. Also, "
                       "arguments have been reordered so that 'name' is the "
                       "last argument.",
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
        "tf.nn.softmax_cross_entropy_with_logits":
            "tf.nn.softmax_cross_entropy_with_logits behavior has changed. "
            "'labels' needs to be wrapped with tf.stop_gradient to keep the "
            "old behavior. Also, 'dim' argument has been renamed to 'axis'.",
        "tf.test.assert_equal_graph_def":
            "tf.assert_equal_graph_def no longer takes 'checkpoint_v2' "
            "argument. 'checkpoint_v2' now defaults to True.",
    }

    self.symbol_renames = {
        name: new_name
        for name, new_name in self.symbol_renames.items()
    }

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

    # Specify warnings for functions that aren't restricted to the tf.x.y.z
    # format. This should only be used for methods with unique names, e.g.
    # export_savedmodel, which is only defined in Estimator objects.
    self.unrestricted_function_warnings = {
        "export_savedmodel": export_saved_model_renamed,
        "make_initializable_iterator": make_initializable_iterator_deprecation,
        "make_one_shot_iterator": make_one_shot_iterator_deprecation,
    }

  @staticmethod
  def _dropout_handler(file_edit_recorder, node, lines):
    del lines
    if len(node.args) < 2:
      comment = ("ERROR: tf.nn.dropout did not take arguments, so automatic "
                 "transformation was disabled. tf.nn.dropout has changed "
                 "the semantics of the second argument.")
      file_edit_recorder.add(
          comment,
          node.lineno,
          node.col_offset,
          "tf.nn.dropout",
          "tf.nn.dropout",
          error="tf.nn.dropout requires manual check.")
    else:
      comment = ("WARNING: tf.nn.dropout has changed the semantics of the "
                 "second argument. Please check the transformation.\n")
      file_edit_recorder.add(
          comment,
          node.args[1].lineno,
          node.args[1].col_offset,
          "",
          "1 - ")

  @staticmethod
  def _colocate_handler(name):
    def _helper(file_edit_recorder, node, lines):
      """Handler for updating colocate arguments."""
      del lines
      for keyword in node.keywords:
        if keyword.arg == "colocate_gradients_with_ops":
          # TODO(jhseu): Since ast_edit.py does string replacement, there's no
          # straightforward way to remove the argument. Try to fix before 2.0 is
          # final.
          comment = ("For tf.gradients and tf.Optimizer.minimize, "
                     "colocate_gradients_with_op has been removed and now "
                     "defaults to True.")
          file_edit_recorder.add(
              comment,
              node.lineno,
              node.col_offset,
              "",
              "",
              error="{} requires manual check.".format(name))
    return _helper

  @staticmethod
  def _batch_gather_handler(file_edit_recorder, node, lines):
    lineno = node.lineno
    column = node.col_offset

    # Find the position to add the batch_dims argument.  We add it as the
    # first argument, since that's easiest.  This is safe because we included
    # batch_gather in self.reordered_function_names, so it will have all
    # of its arguments changed to keyword arguments.
    m = re.match(r"tf\s*\.\s*batch_gather\s*\(", lines[lineno - 1][column:])
    if m is not None:
      file_edit_recorder.add(
          "Added keyword argument 'batch_dims=-1' to 'tf.batch_gather'",
          lineno, column + m.end(), "", "batch_dims=-1, ")
    else:
      file_edit_recorder.add(
          "Unable to add keyword argument 'batch_dims=-1' to 'tf.batch_gather'",
          lineno, column, "", "",
          error="Unable to add keyword argument batch_dims=-1 to "
          "tf.batch_gather; please add it manually.")
