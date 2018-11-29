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

from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import renames_v2


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
        "tf.batch_to_space_nd": {
            "block_size": "block_shape",
        },
        "tf.constant": {
            "verify_shapes": "verify_shapes_is_now_always_true",
        },
        "tf.convert_to_tensor": {
            "preferred_dtype": "dtype_hint"
        },
        "tf.linalg.l2_normalize": {
            "dim": "axis",
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
        },
        "tf.sparse_concat": {
            "concat_dim": "axis",
        },
        "tf.sparse.split": {
            "split_dim": "axis",
        },
        "tf.max_pool_with_argmax": {
            "Targmax": "output_dtype",
        },
        "tf.multinomial": {
            "output_dtype": "dtype",
        },
        "tf.random.multinomial": {
            "output_dtype": "dtype",
        },
        "tf.nn.batch_norm_with_global_normalization": {
            "t": "input",
            "m": "mean",
            "v": "variance",
        },
        "tf.manip.batch_to_space_nd": {
            "block_size": "block_shape",
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
            "oldpath": "src",
            "newpath": "dst",
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
    }

    # pylint: disable=line-too-long
    # Add additional renames not in renames_v2.py here.
    # IMPORTANT: For the renames in here, if you also need to add to
    # function_reorders or function_keyword_renames, use the OLD function name.
    # These renames happen after the arguments have been processed.
    self.manual_symbol_renames = {
        "tf.batch_to_space_nd":
            "tf.batch_to_space",
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
        "tf.contrib.framework.sort":
            "tf.sort",
        "tf.contrib.framework.argsort":
            "tf.argsort",
        "tf.manip.batch_to_space_nd":
            "tf.batch_to_space",
        "tf.quantize_v2":
            "tf.quantization.quantize",
        "tf.sparse_concat":
            "tf.sparse.concat",
        "tf.sparse_split":
            "tf.sparse.split",
        "tf.string_to_hash_bucket":
            "tf.strings.to_hash_bucket",
        "tf.string_to_number":
            "tf.strings.to_number",
        "tf.multinomial":
            "tf.random.categorical",
        "tf.random.multinomial":
            "tf.random.categorical",
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
        "tf.nn.fused_batch_norm":
            "tf.compat.v1.nn.fused_batch_norm",
    }
    # pylint: enable=line-too-long

    # Mapping from function to the new name of the function
    self.symbol_renames = renames_v2.renames
    self.symbol_renames.update(self.manual_symbol_renames)

    # Variables that should be changed to functions.
    self.change_to_function = {}

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    # IMPORTANT: order here should correspond to OLD argument order.
    # We just prepend "arg_name=" to all arguments in function calls.
    self.function_reorders = {
        "tf.io.serialize_sparse": ["sp_input", "name", "out_type"],
        "tf.io.serialize_many_sparse": ["sp_input", "name", "out_type"],
        "tf.argmax": ["input", "axis", "name", "axis", "output_type"],
        "tf.argmin": ["input", "axis", "name", "axis", "output_type"],
        "tf.batch_to_space": ["input", "crops", "block_size", "name"],
        "tf.boolean_mask": ["tensor", "mask", "name", "axis"],
        "tf.convert_to_tensor": ["value", "dtype", "name", "preferred_dtype"],
        "tf.nn.moments": ["x", "axes", "shift", "keepdims", "name"],
        "tf.nn.convolution": [
            "input", "filter", "padding", "strides", "dilation_rate", "name",
            "data_format"
        ],
        "tf.nn.crelu": ["features", "name", "axis"],
        "tf.nn.pool": [
            "input", "window_shape", "pooling_type", "padding", "dilation_rate",
            "strides", "name", "data_format"
        ],
        "tf.nn.depthwise_conv2d": [
            "input", "filter", "strides", "padding", "rate", "name",
            "data_format"
        ],
        "tf.manip.batch_to_space_nd": ["input", "crops", "block_size", "name"],
        "tf.multinomial": [
            "logits", "num_samples", "seed", "name", "output_dtype"
        ],
        "tf.random.multinomial": [
            "logits", "num_samples", "seed", "name", "output_dtype"
        ],
        "tf.pad": ["tensor", "paddings", "mode", "name", "constant_values"],
        "tf.quantize_v2": [
            "input", "min_range", "max_range", "T", "mode", "name", "round_mode"
        ],
        "tf.feature_column.categorical_column_with_vocabulary_file": [
            "key", "vocabulary_file", "vocabulary_size", "num_oov_buckets",
            "default_value", "dtype"
        ],
        "tf.shape": ["input", "name", "out_type"],
        "tf.size": ["input", "name", "out_type"],
        "tf.random.poisson": ["lam", "shape", "dtype", "seed", "name"],
        "tf.sparse.add": ["a", "b", "thresh"],
        "tf.sparse_add": ["a", "b", "thresh"],
        "tf.sparse.concat": [
            "axis", "sp_inputs", "name", "expand_nonconcat_dim", "concat_dim"
        ],
        "tf.sparse_concat": [
            "axis", "sp_inputs", "name", "expand_nonconcat_dim", "concat_dim"
        ],
        "tf.sparse.segment_mean": [
            "data", "indices", "segment_ids", "name", "num_segments"
        ],
        "tf.sparse.segment_sqrt_n": [
            "data", "indices", "segment_ids", "name", "num_segments"
        ],
        "tf.sparse.segment_sum": [
            "data", "indices", "segment_ids", "name", "num_segments"
        ],
        "tf.io.decode_csv": [
            "records",
            "record_defaults",
            "field_delim",
            "use_quote_delim",
            "name",
            "na_value",
            "select_cols",
        ],
        "tf.strings.substr": ["input", "pos", "len", "name", "unit"],
        "tf.strings.reduce_join": [
            "input", "axis", "keep_dims", "separator", "name",
            "reduction_indices"
        ],
        "tf.strings.length": ["input", "name", "unit"],
        "tf.transpose": ["a", "perm", "name", "conjugate"],
        "tf.tuple": ["tensors", "name", "control_inputs"],
        "tf.io.parse_example": [
            "serialized", "features", "name", "example_names"
        ],
        "tf.io.parse_single_example": [
            "serialized", "features", "name", "example_names"
        ],
        "tf.while_loop": [
            "cond", "body", "loop_vars", "shape_invariants",
            "parallel_iterations", "back_prop", "swap_memory", "name",
            "maximum_iterations", "return_same_structure"
        ],
        "tf.reduce_all": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_all": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_any": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_any": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_min": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_min": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_max": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_max": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_sum": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_sum": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_mean": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_mean": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_prod": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_prod": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_logsumexp": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.math.reduce_logsumexp": [
            "input_tensor", "axis", "keepdims", "name", "reduction_indices",
            "keep_dims"
        ],
        "tf.reduce_join": [
            "input", "axis", "keep_dims", "separator", "name",
            "reduction_indices"
        ],
        "tf.confusion_matrix": [
            "labels", "predictions", "num_classes", "dtype", "name", "weights"
        ],
        "tf.math.confusion_matrix": [
            "labels", "predictions", "num_classes", "dtype", "name", "weights"
        ]
    }

    # Specially handled functions.
    self.function_handle = {
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

    # Function warnings. <function name> placeholder inside warnings will be
    # replaced by function name.
    self.function_warnings = {
        "tf.assert_greater": assert_return_type_comment,
        "tf.assert_equal": assert_return_type_comment,
        "tf.assert_less": assert_return_type_comment,
        "tf.assert_rank": assert_rank_comment,
        "tf.debugging.assert_equal": assert_return_type_comment,
        "tf.debugging.assert_greater": assert_return_type_comment,
        "tf.debugging.assert_greater_equal": assert_return_type_comment,
        "tf.debugging.assert_integer": assert_return_type_comment,
        "tf.debugging.assert_less": assert_return_type_comment,
        "tf.debugging.assert_less_equal": assert_return_type_comment,
        "tf.debugging.assert_near": assert_return_type_comment,
        "tf.debugging.assert_negative": assert_return_type_comment,
        "tf.debugging.assert_non_negative": assert_return_type_comment,
        "tf.debugging.assert_non_positive": assert_return_type_comment,
        "tf.debugging.assert_none_equal": assert_return_type_comment,
        "tf.debugging.assert_positive": assert_return_type_comment,
        "tf.debugging.assert_rank": assert_rank_comment,
        "tf.debugging.assert_rank_at_least": assert_rank_comment,
        "tf.debugging.assert_rank_in": assert_rank_comment,
        "tf.flags": "tf.flags has been removed, please use the argparse or absl"
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
        "WARNING: use_cudnn_on_gpu argument has been removed and \"value\" was "
        "renamed to \"input\"",
        "tf.nn.conv2d":
        "WARNING: use_cudnn_on_gpu argument has been removed and \"filter\" "
        "was renamed to \"filters\"",
        "tf.nn.conv2d_backprop_filter":
        "WARNING: use_cudnn_on_gpu argument has been removed",
        "tf.nn.conv2d_backprop_input":
        "WARNING: use_cudnn_on_gpu argument has been removed and \"filter\" "
        "was renamed to \"filters\"",
        "tf.nn.erosion2d":
        "WARNING: <function name> now requires a data_format argument",
        "tf.zeros_like": tf_01s_like_no_optimize_comment,
        "tf.ones_like": tf_01s_like_no_optimize_comment,
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

    # Specify warnings for functions that aren't restricted to the tf.x.y.z
    # format. This should only be used for methods with unique names, e.g.
    # export_savedmodel, which is only defined in Estimator objects.
    self.unrestricted_function_warnings = {
        "export_savedmodel": export_saved_model_renamed,
    }

  @staticmethod
  def _dropout_handler(file_edit_recorder, node):
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
    def _helper(file_edit_recorder, node):
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
