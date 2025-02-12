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
"""Provides a list of renames between TensorFlow 1.* and 2.0."""
from tensorflow.tools.compatibility import renames_v2

# pylint: disable=line-too-long

# Add additional renames not in renames_v2.py here.
# IMPORTANT: For the renames in here, if you also need to add to
# function_reorders or function_keyword_renames in tf_upgrade_v2.py,
# use the OLD function name.
# These renames happen after the arguments have been processed.
# After modifying this dict, run the following to update reorders_v2.py:
# bazel run tensorflow/tools/compatibility/update:generate_v2_reorders_map
manual_symbol_renames = {
    "tf.batch_to_space_nd": "tf.batch_to_space",
    "tf.batch_gather": "tf.compat.v1.batch_gather",
    "tf.space_to_batch_nd": "tf.space_to_batch",
    "tf.nn.space_to_batch": "tf.space_to_batch",
    "tf.extract_image_patches": "tf.image.extract_patches",
    "tf.image.extract_image_patches": "tf.image.extract_patches",
    "tf.gfile.Copy": "tf.io.gfile.copy",
    "tf.gfile.DeleteRecursively": "tf.io.gfile.rmtree",
    "tf.gfile.Exists": "tf.io.gfile.exists",
    "tf.gfile.Glob": "tf.io.gfile.glob",
    "tf.gfile.GFile": "tf.io.gfile.GFile",
    "tf.gfile.IsDirectory": "tf.io.gfile.isdir",
    "tf.gfile.ListDirectory": "tf.io.gfile.listdir",
    "tf.gfile.MakeDirs": "tf.io.gfile.makedirs",
    "tf.gfile.MkDir": "tf.io.gfile.mkdir",
    "tf.gfile.Open": "tf.io.gfile.GFile",
    "tf.gfile.Remove": "tf.io.gfile.remove",
    "tf.gfile.Rename": "tf.io.gfile.rename",
    "tf.gfile.Stat": "tf.io.gfile.stat",
    "tf.gfile.Walk": "tf.io.gfile.walk",
    "tf.contrib.cluster_resolver.ClusterResolver": (
        "tf.distribute.cluster_resolver.ClusterResolver"
    ),
    "tf.contrib.cluster_resolver.GceClusterResolver": (
        "tf.distribute.cluster_resolver.GCEClusterResolver"
    ),
    "tf.contrib.cluster_resolver.KubernetesClusterResolver": (
        "tf.distribute.cluster_resolver.KubernetesClusterResolver"
    ),
    "tf.contrib.cluster_resolver.SimpleClusterResolver": (
        "tf.distribute.cluster_resolver.SimpleClusterResolver"
    ),
    "tf.contrib.cluster_resolver.SlurmClusterResolver": (
        "tf.distribute.cluster_resolver.SlurmClusterResolver"
    ),
    "tf.contrib.cluster_resolver.TFConfigClusterResolver": (
        "tf.distribute.cluster_resolver.TFConfigClusterResolver"
    ),
    "tf.contrib.cluster_resolver.TPUClusterResolver": (
        "tf.distribute.cluster_resolver.TPUClusterResolver"
    ),
    "tf.contrib.cluster_resolver.UnionClusterResolver": (
        "tf.distribute.cluster_resolver.UnionClusterResolver"
    ),
    "tf.contrib.data.AUTOTUNE": "tf.data.experimental.AUTOTUNE",
    "tf.contrib.data.Counter": "tf.data.experimental.Counter",
    "tf.contrib.data.CheckpointInputPipelineHook": (
        "tf.data.experimental.CheckpointInputPipelineHook"
    ),
    "tf.contrib.data.CsvDataset": "tf.data.experimental.CsvDataset",
    "tf.contrib.data.Optional": "tf.data.experimental.Optional",
    "tf.contrib.data.RandomDataset": "tf.data.experimental.RandomDataset",
    "tf.contrib.data.Reducer": "tf.data.experimental.Reducer",
    "tf.contrib.data.SqlDataset": "tf.data.experimental.SqlDataset",
    "tf.contrib.data.StatsAggregator": "tf.data.experimental.StatsAggregator",
    "tf.contrib.data.TFRecordWriter": "tf.data.experimental.TFRecordWriter",
    "tf.contrib.data.assert_element_shape": (
        "tf.data.experimental.assert_element_shape"
    ),
    "tf.contrib.data.bucket_by_sequence_length": (
        "tf.data.experimental.bucket_by_sequence_length"
    ),
    "tf.contrib.data.choose_from_datasets": (
        "tf.data.experimental.choose_from_datasets"
    ),
    "tf.contrib.data.copy_to_device": "tf.data.experimental.copy_to_device",
    "tf.contrib.data.dense_to_sparse_batch": (
        "tf.data.experimental.dense_to_sparse_batch"
    ),
    "tf.contrib.data.enumerate_dataset": (
        "tf.data.experimental.enumerate_dataset"
    ),
    "tf.contrib.data.get_next_as_optional": (
        "tf.data.experimental.get_next_as_optional"
    ),
    "tf.contrib.data.get_single_element": (
        "tf.data.experimental.get_single_element"
    ),
    "tf.contrib.data.group_by_reducer": "tf.data.experimental.group_by_reducer",
    "tf.contrib.data.group_by_window": "tf.data.experimental.group_by_window",
    "tf.contrib.data.ignore_errors": "tf.data.experimental.ignore_errors",
    "tf.contrib.data.latency_stats": "tf.data.experimental.latency_stats",
    "tf.contrib.data.make_batched_features_dataset": (
        "tf.data.experimental.make_batched_features_dataset"
    ),
    "tf.contrib.data.make_csv_dataset": "tf.data.experimental.make_csv_dataset",
    "tf.contrib.data.make_saveable_from_iterator": (
        "tf.data.experimental.make_saveable_from_iterator"
    ),
    "tf.contrib.data.map_and_batch": "tf.data.experimental.map_and_batch",
    "tf.contrib.data.parallel_interleave": (
        "tf.data.experimental.parallel_interleave"
    ),
    "tf.contrib.data.parse_example_dataset": (
        "tf.data.experimental.parse_example_dataset"
    ),
    "tf.contrib.data.prefetch_to_device": (
        "tf.data.experimental.prefetch_to_device"
    ),
    "tf.contrib.data.rejection_resample": (
        "tf.data.experimental.rejection_resample"
    ),
    "tf.contrib.data.sample_from_datasets": (
        "tf.data.experimental.sample_from_datasets"
    ),
    "tf.contrib.data.scan": "tf.data.experimental.scan",
    "tf.contrib.data.set_stats_aggregator": (
        "tf.data.experimental.set_stats_aggregator"
    ),
    "tf.contrib.data.shuffle_and_repeat": (
        "tf.data.experimental.shuffle_and_repeat"
    ),
    "tf.contrib.data.unbatch": "tf.data.experimental.unbatch",
    "tf.contrib.data.unique": "tf.data.experimental.unique",
    "tf.contrib.distribute.CrossDeviceOps": "tf.distribute.CrossDeviceOps",
    "tf.contrib.distribute.ReductionToOneDeviceCrossDeviceOps": (
        "tf.distribute.ReductionToOneDevice"
    ),
    "tf.contrib.framework.CriticalSection": "tf.CriticalSection",
    "tf.contrib.framework.is_tensor": "tf.is_tensor",
    "tf.contrib.framework.load_variable": "tf.train.load_variable",
    "tf.contrib.framework.nest.assert_same_structure": (
        "tf.nest.assert_same_structure"
    ),
    "tf.contrib.framework.nest.flatten": "tf.nest.flatten",
    "tf.contrib.framework.nest.is_nested": "tf.nest.is_nested",
    "tf.contrib.framework.nest.map_structure": "tf.nest.map_structure",
    "tf.contrib.framework.nest.pack_sequence_as": "tf.nest.pack_sequence_as",
    "tf.contrib.batching.batch_function": "tf.nondifferentiable_batch_function",
    "tf.contrib.util.constant_value": "tf.get_static_value",
    "tf.contrib.saved_model.load_keras_model": (
        "tf.compat.v1.keras.experimental.load_from_saved_model"
    ),
    "tf.contrib.saved_model.save_keras_model": (
        "tf.compat.v1.keras.experimental.export_saved_model"
    ),
    "tf.contrib.rnn.RNNCell": "tf.compat.v1.nn.rnn_cell.RNNCell",
    "tf.contrib.rnn.LSTMStateTuple": "tf.nn.rnn_cell.LSTMStateTuple",
    "tf.contrib.rnn.BasicLSTMCell": "tf.compat.v1.nn.rnn_cell.BasicLSTMCell",
    "tf.contrib.rnn.BasicRNNCell": "tf.compat.v1.nn.rnn_cell.BasicRNNCell",
    "tf.contrib.rnn.GRUCell": "tf.compat.v1.nn.rnn_cell.GRUCell",
    "tf.contrib.rnn.LSTMCell": "tf.compat.v1.nn.rnn_cell.LSTMCell",
    "tf.contrib.rnn.MultiRNNCell": "tf.compat.v1.nn.rnn_cell.MultiRNNCell",
    "tf.contrib.rnn.static_rnn": "tf.compat.v1.nn.static_rnn",
    "tf.contrib.rnn.static_state_saving_rnn": (
        "tf.compat.v1.nn.static_state_saving_rnn"
    ),
    "tf.contrib.rnn.static_bidirectional_rnn": (
        "tf.compat.v1.nn.static_bidirectional_rnn"
    ),
    "tf.contrib.framework.sort": "tf.sort",
    "tf.contrib.framework.argsort": "tf.argsort",
    "tf.contrib.summary.all_summary_ops": (
        "tf.compat.v1.summary.all_v2_summary_ops"
    ),
    "tf.contrib.summary.always_record_summaries": (
        "tf.compat.v2.summary.record_if"
    ),
    "tf.contrib.summary.audio": "tf.compat.v2.summary.audio",
    "tf.contrib.summary.create_file_writer": (
        "tf.compat.v2.summary.create_file_writer"
    ),
    "tf.contrib.summary.flush": "tf.compat.v2.summary.flush",
    "tf.contrib.summary.generic": "tf.compat.v2.summary.write",
    "tf.contrib.summary.histogram": "tf.compat.v2.summary.histogram",
    "tf.contrib.summary.image": "tf.compat.v2.summary.image",
    "tf.contrib.summary.initialize": "tf.compat.v1.summary.initialize",
    "tf.contrib.summary.never_record_summaries": (
        "tf.compat.v2.summary.record_if"
    ),
    "tf.contrib.summary.scalar": "tf.compat.v2.summary.scalar",
    "tf.contrib.tpu.CrossShardOptimizer": (
        "tf.compat.v1.tpu.CrossShardOptimizer"
    ),
    "tf.contrib.tpu.batch_parallel": "tf.compat.v1.tpu.batch_parallel",
    "tf.contrib.tpu.bfloat16_scope": "tf.compat.v1.tpu.bfloat16_scope",
    "tf.contrib.tpu.core": "tf.compat.v1.tpu.core",
    "tf.contrib.tpu.cross_replica_sum": "tf.compat.v1.tpu.cross_replica_sum",
    "tf.contrib.tpu.initialize_system": "tf.compat.v1.tpu.initialize_system",
    "tf.contrib.tpu.outside_compilation": (
        "tf.compat.v1.tpu.outside_compilation"
    ),
    "tf.contrib.tpu.replicate": "tf.compat.v1.tpu.replicate",
    "tf.contrib.tpu.rewrite": "tf.compat.v1.tpu.rewrite",
    "tf.contrib.tpu.shard": "tf.compat.v1.tpu.shard",
    "tf.contrib.tpu.shutdown_system": "tf.compat.v1.tpu.shutdown_system",
    "tf.contrib.training.checkpoints_iterator": "tf.train.checkpoints_iterator",
    "tf.contrib.layers.recompute_grad": "tf.recompute_grad",
    "tf.count_nonzero": "tf.math.count_nonzero",
    "tf.decode_raw": "tf.io.decode_raw",
    "tf.manip.batch_to_space_nd": "tf.batch_to_space",
    "tf.quantize_v2": "tf.quantization.quantize",
    "tf.sparse_matmul": "tf.linalg.matmul",
    "tf.random.stateless_multinomial": "tf.random.stateless_categorical",
    "tf.substr": "tf.strings.substr",
    # TODO(b/129398290)
    "tf.string_split": "tf.compat.v1.string_split",
    "tf.string_to_hash_bucket": "tf.strings.to_hash_bucket",
    "tf.string_to_number": "tf.strings.to_number",
    "tf.multinomial": "tf.random.categorical",
    "tf.random.multinomial": "tf.random.categorical",
    "tf.reduce_join": "tf.strings.reduce_join",
    "tf.load_file_system_library": "tf.load_library",
    "tf.bincount": "tf.math.bincount",
    "tf.confusion_matrix": "tf.math.confusion_matrix",
    "tf.train.confusion_matrix": "tf.math.confusion_matrix",
    "tf.train.sdca_fprint": "tf.raw_ops.SdcaFprint",
    "tf.train.sdca_optimizer": "tf.raw_ops.SdcaOptimizer",
    "tf.train.sdca_shrink_l1": "tf.raw_ops.SdcaShrinkL1",
    "tf.decode_csv": "tf.io.decode_csv",
    "tf.data.Iterator": "tf.compat.v1.data.Iterator",
    "tf.data.experimental.DatasetStructure": "tf.data.DatasetSpec",
    "tf.data.experimental.OptionalStructure": "tf.OptionalSpec",
    "tf.data.experimental.RaggedTensorStructure": "tf.RaggedTensorSpec",
    "tf.data.experimental.SparseTensorStructure": "tf.SparseTensorSpec",
    "tf.data.experimental.Structure": "tf.TypeSpec",
    "tf.data.experimental.TensorArrayStructure": "tf.TensorArraySpec",
    "tf.data.experimental.TensorStructure": "tf.TensorSpec",
    "tf.parse_example": "tf.io.parse_example",
    "tf.parse_single_example": "tf.io.parse_single_example",
    "tf.nn.fused_batch_norm": "tf.compat.v1.nn.fused_batch_norm",
    "tf.nn.softmax_cross_entropy_with_logits_v2": (
        "tf.nn.softmax_cross_entropy_with_logits"
    ),
    "tf.losses.Reduction.MEAN": "tf.compat.v1.losses.Reduction.MEAN",
    "tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS": (
        "tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS"
    ),
    "tf.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS": (
        "tf.compat.v1.losses.Reduction.SUM_OVER_NONZERO_WEIGHTS"
    ),
    "tf.lite.constants.FLOAT": "tf.float32",
    "tf.lite.constants.FLOAT16": "tf.float16",
    "tf.lite.constants.INT16": "tf.int16",
    "tf.lite.constants.INT32": "tf.int32",
    "tf.lite.constants.INT64": "tf.int64",
    "tf.lite.constants.INT8": "tf.int8",
    "tf.lite.constants.STRING": "tf.string",
    "tf.lite.constants.QUANTIZED_UINT8": "tf.uint8",
    "tf.arg_max": "tf.argmax",
    "tf.arg_min": "tf.argmin",
    # tf.nn.ctc_loss is still available in 2.0 but behavior
    # changed significantly.
    "tf.nn.ctc_loss": "tf.compat.v1.nn.ctc_loss",
    # tf.saved_model.load in 1.x has no equivalent in 2.x, but there is a
    # symbol with the same name.
    "tf.saved_model.load": "tf.compat.v1.saved_model.load",
    "tf.saved_model.loader.load": "tf.compat.v1.saved_model.load",
    "tf.saved_model.load_v2": "tf.compat.v2.saved_model.load",
    "tf.image.resize_images": "tf.image.resize",
    "tf.assert_equal": "tf.compat.v1.assert_equal",
    "tf.assert_greater": "tf.compat.v1.assert_greater",
    "tf.assert_greater_equal": "tf.compat.v1.assert_greater_equal",
    "tf.assert_integer": "tf.compat.v1.assert_integer",
    "tf.assert_less": "tf.compat.v1.assert_less",
    "tf.assert_less_equal": "tf.compat.v1.assert_less_equal",
    "tf.assert_near": "tf.compat.v1.assert_near",
    "tf.assert_negative": "tf.compat.v1.assert_negative",
    "tf.assert_non_negative": "tf.compat.v1.assert_non_negative",
    "tf.assert_non_positive": "tf.compat.v1.assert_non_positive",
    "tf.assert_none_equal": "tf.compat.v1.assert_none_equal",
    "tf.assert_positive": "tf.compat.v1.assert_positive",
    "tf.assert_rank": "tf.compat.v1.assert_rank",
    "tf.assert_rank_at_least": "tf.compat.v1.assert_rank_at_least",
    "tf.assert_rank_in": "tf.compat.v1.assert_rank_in",
    "tf.assert_scalar": "tf.compat.v1.assert_scalar",
    "tf.assert_type": "tf.compat.v1.assert_type",
    "tf.assert_variables_initialized": (
        "tf.compat.v1.assert_variables_initialized"
    ),
    "tf.debugging.assert_equal": "tf.compat.v1.debugging.assert_equal",
    "tf.debugging.assert_greater": "tf.compat.v1.debugging.assert_greater",
    "tf.debugging.assert_greater_equal": (
        "tf.compat.v1.debugging.assert_greater_equal"
    ),
    "tf.debugging.assert_integer": "tf.compat.v1.debugging.assert_integer",
    "tf.debugging.assert_less": "tf.compat.v1.debugging.assert_less",
    "tf.debugging.assert_less_equal": (
        "tf.compat.v1.debugging.assert_less_equal"
    ),
    "tf.debugging.assert_near": "tf.compat.v1.debugging.assert_near",
    "tf.debugging.assert_negative": "tf.compat.v1.debugging.assert_negative",
    "tf.debugging.assert_non_negative": (
        "tf.compat.v1.debugging.assert_non_negative"
    ),
    "tf.debugging.assert_non_positive": (
        "tf.compat.v1.debugging.assert_non_positive"
    ),
    "tf.debugging.assert_none_equal": (
        "tf.compat.v1.debugging.assert_none_equal"
    ),
    "tf.debugging.assert_positive": "tf.compat.v1.debugging.assert_positive",
    "tf.debugging.assert_rank": "tf.compat.v1.debugging.assert_rank",
    "tf.debugging.assert_rank_at_least": (
        "tf.compat.v1.debugging.assert_rank_at_least"
    ),
    "tf.debugging.assert_rank_in": "tf.compat.v1.debugging.assert_rank_in",
    "tf.debugging.assert_scalar": "tf.compat.v1.debugging.assert_scalar",
    "tf.debugging.assert_type": "tf.compat.v1.debugging.assert_type",
    "tf.errors.exception_type_from_error_code": (
        "tf.compat.v1.errors.exception_type_from_error_code"
    ),
    "tf.errors.error_code_from_exception_type": (
        "tf.compat.v1.errors.error_code_from_exception_type"
    ),
    "tf.errors.raise_exception_on_not_ok_status": (
        "tf.compat.v1.errors.raise_exception_on_not_ok_status"
    ),
    "tf.nn.max_pool": "tf.nn.max_pool2d",
    "tf.nn.avg_pool": "tf.nn.avg_pool2d",
    "tf.keras.initializers.zeros": "tf.compat.v1.keras.initializers.zeros",
    "tf.keras.initializers.Zeros": "tf.compat.v1.keras.initializers.Zeros",
    "tf.keras.initializers.ones": "tf.compat.v1.keras.initializers.ones",
    "tf.keras.initializers.Ones": "tf.compat.v1.keras.initializers.Ones",
    "tf.keras.initializers.constant": (
        "tf.compat.v1.keras.initializers.constant"
    ),
    "tf.keras.initializers.Constant": (
        "tf.compat.v1.keras.initializers.Constant"
    ),
    "tf.keras.initializers.VarianceScaling": (
        "tf.compat.v1.keras.initializers.VarianceScaling"
    ),
    "tf.keras.initializers.Orthogonal": (
        "tf.compat.v1.keras.initializers.Orthogonal"
    ),
    "tf.keras.initializers.orthogonal": (
        "tf.compat.v1.keras.initializers.orthogonal"
    ),
    "tf.keras.initializers.Identity": (
        "tf.compat.v1.keras.initializers.Identity"
    ),
    "tf.keras.initializers.identity": (
        "tf.compat.v1.keras.initializers.identity"
    ),
    "tf.keras.initializers.glorot_uniform": (
        "tf.compat.v1.keras.initializers.glorot_uniform"
    ),
    "tf.keras.initializers.glorot_normal": (
        "tf.compat.v1.keras.initializers.glorot_normal"
    ),
    "tf.keras.initializers.lecun_normal": (
        "tf.compat.v1.keras.initializers.lecun_normal"
    ),
    "tf.keras.initializers.lecun_uniform": (
        "tf.compat.v1.keras.initializers.lecun_uniform"
    ),
    "tf.keras.initializers.he_normal": (
        "tf.compat.v1.keras.initializers.he_normal"
    ),
    "tf.keras.initializers.he_uniform": (
        "tf.compat.v1.keras.initializers.he_uniform"
    ),
    "tf.keras.initializers.TruncatedNormal": (
        "tf.compat.v1.keras.initializers.TruncatedNormal"
    ),
    "tf.keras.initializers.truncated_normal": (
        "tf.compat.v1.keras.initializers.truncated_normal"
    ),
    "tf.keras.initializers.RandomUniform": (
        "tf.compat.v1.keras.initializers.RandomUniform"
    ),
    "tf.keras.initializers.uniform": "tf.compat.v1.keras.initializers.uniform",
    "tf.keras.initializers.random_uniform": (
        "tf.compat.v1.keras.initializers.random_uniform"
    ),
    "tf.keras.initializers.RandomNormal": (
        "tf.compat.v1.keras.initializers.RandomNormal"
    ),
    "tf.keras.initializers.normal": "tf.compat.v1.keras.initializers.normal",
    "tf.keras.initializers.random_normal": (
        "tf.compat.v1.keras.initializers.random_normal"
    ),
    "tf.zeros_initializer": "tf.compat.v1.zeros_initializer",
    "tf.initializers.zeros": "tf.compat.v1.initializers.zeros",
    "tf.ones_initializer": "tf.compat.v1.ones_initializer",
    "tf.initializers.ones": "tf.compat.v1.initializers.ones",
    "tf.constant_initializer": "tf.compat.v1.constant_initializer",
    "tf.initializers.constant": "tf.compat.v1.initializers.constant",
    "tf.random_uniform_initializer": "tf.compat.v1.random_uniform_initializer",
    "tf.initializers.random_uniform": (
        "tf.compat.v1.initializers.random_uniform"
    ),
    "tf.random_normal_initializer": "tf.compat.v1.random_normal_initializer",
    "tf.initializers.random_normal": "tf.compat.v1.initializers.random_normal",
    "tf.truncated_normal_initializer": (
        "tf.compat.v1.truncated_normal_initializer"
    ),
    "tf.initializers.truncated_normal": (
        "tf.compat.v1.initializers.truncated_normal"
    ),
    "tf.variance_scaling_initializer": (
        "tf.compat.v1.variance_scaling_initializer"
    ),
    "tf.initializers.variance_scaling": (
        "tf.compat.v1.initializers.variance_scaling"
    ),
    "tf.orthogonal_initializer": "tf.compat.v1.orthogonal_initializer",
    "tf.initializers.orthogonal": "tf.compat.v1.initializers.orthogonal",
    "tf.glorot_uniform_initializer": "tf.compat.v1.glorot_uniform_initializer",
    "tf.initializers.glorot_uniform": (
        "tf.compat.v1.initializers.glorot_uniform"
    ),
    "tf.glorot_normal_initializer": "tf.compat.v1.glorot_normal_initializer",
    "tf.initializers.glorot_normal": "tf.compat.v1.initializers.glorot_normal",
    "tf.initializers.identity": "tf.compat.v1.initializers.identity",
    "tf.initializers.lecun_normal": "tf.compat.v1.initializers.lecun_normal",
    "tf.initializers.lecun_uniform": "tf.compat.v1.initializers.lecun_uniform",
    "tf.initializers.he_normal": "tf.compat.v1.initializers.he_normal",
    "tf.initializers.he_uniform": "tf.compat.v1.initializers.he_uniform",
    "tf.data.experimental.map_and_batch_with_legacy_function": (
        "tf.compat.v1.data.experimental.map_and_batch_with_legacy_function"
    ),
    "tf.nn.conv2d_backprop_input": "tf.nn.conv2d_transpose",
    "tf.test.compute_gradient": "tf.compat.v1.test.compute_gradient",
    "tf.floor_div": "tf.math.floordiv",
    "tf.where": "tf.compat.v1.where",
    "tf.where_v2": "tf.compat.v2.where",
    "tf.app.flags": "tf.compat.v1.app.flags",
}
# pylint: enable=line-too-long


def add_contrib_direct_import_support(symbol_dict):
  """Add support for `tf.contrib.*` alias `contrib_*.` Updates dict in place."""
  for symbol_name in list(symbol_dict.keys()):
    symbol_alias = symbol_name.replace("tf.contrib.", "contrib_")
    symbol_dict[symbol_alias] = symbol_dict[symbol_name]

add_contrib_direct_import_support(manual_symbol_renames)

symbol_renames = renames_v2.renames
symbol_renames.update(manual_symbol_renames)

addons_symbol_mappings = {
    "tf.contrib.layers.poincare_normalize":
        "tfa.layers.PoincareNormalize",
    "tf.contrib.layers.maxout":
        "tfa.layers.Maxout",
    "tf.contrib.layers.group_norm":
        "tfa.layers.GroupNormalization",
    "tf.contrib.layers.instance_norm":
        "tfa.layers.InstanceNormalization",
    "tf.contrib.sparsemax.sparsemax":
        "tfa.activations.sparsemax",
    "tf.contrib.losses.metric_learning.contrastive_loss":
        "tfa.losses.ContrastiveLoss",
    "tf.contrib.losses.metric_learning.lifted_struct_loss":
        "tfa.losses.LiftedStructLoss",
    "tf.contrib.sparsemax.sparsemax_loss":
        "tfa.losses.SparsemaxLoss",
    "tf.contrib.losses.metric_learning.triplet_semihard_loss":
        "tfa.losses.TripletSemiHardLoss",
    "tf.contrib.opt.LazyAdamOptimizer":
        "tfa.optimizers.LazyAdam",
    "tf.contrib.opt.MovingAverageOptimizer":
        "tfa.optimizers.MovingAverage",
    "tf.contrib.opt.MomentumWOptimizer":
        "tfa.optimizers.SGDW",
    "tf.contrib.opt.AdamWOptimizer":
        "tfa.optimizers.AdamW",
    "tf.contrib.opt.extend_with_decoupled_weight_decay":
        "tfa.optimizers.extend_with_decoupled_weight_decay",
    "tf.contrib.text.skip_gram_sample":
        "tfa.text.skip_gram_sample",
    "tf.contrib.text.skip_gram_sample_with_text_vocab":
        "tfa.text.skip_gram_sample_with_text_vocab",
    "tf.contrib.image.dense_image_warp":
        "tfa.image.dense_image_warp",
    "tf.contrib.image.adjust_hsv_in_yiq":
        "tfa.image.adjust_hsv_in_yiq",
    "tf.contrib.image.compose_transforms":
        "tfa.image.compose_transforms",
    "tf.contrib.image.random_hsv_in_yiq":
        "tfa.image.random_hsv_in_yiq",
    "tf.contrib.image.angles_to_projective_transforms":
        "tfa.image.angles_to_projective_transforms",
    "tf.contrib.image.matrices_to_flat_transforms":
        "tfa.image.matrices_to_flat_transforms",
    "tf.contrib.image.rotate":
        "tfa.image.rotate",
    "tf.contrib.image.transform":
        "tfa.image.transform",
    "tf.contrib.rnn.NASCell":
        "tfa.rnn.NASCell",
    "tf.contrib.rnn.LayerNormBasicLSTMCell":
        "tfa.rnn.LayerNormLSTMCell"
}

add_contrib_direct_import_support(addons_symbol_mappings)
