// RUN: tf-quant-opt %s -tf-quant-convert-tpu-model-to-cpu -inline -tf-quant-cast-bf16-ops-to-f32 -split-input-file | \
// RUN: FileCheck %s

// Remove TPU related ops.
func.func @tpu_conv(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
  %0 = "tf.TPUOrdinalSelector"() {device = ""} : () -> tensor<?xi32>
  %1 = "tf.TPUPartitionedCall"(%arg0, %0) {autotuner_thresh = 0 : i64, device = "", f = @tpu_func_0_optim0} : (tensor<1x3x4x3xf32>, tensor<?xi32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.IdentityN"(%1) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}

func.func private @tpu_func_0_optim0(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> attributes {tf._original_func_name = "tpu_func_0_optim"} {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %cst_0 = "tf.Const"() {device = "", value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {_tpu_replicate = "cluster", device = "", value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "cluster", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
  %1 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  %2 = "tf.Transpose"(%0, %cst_0) {device = ""} : (tensor<1x3x4x3xbf16>, tensor<4xi32>) -> tensor<1x3x3x4xbf16>
  %3 = "tf.TPUReplicatedInput"(%2) {device = "", index = -1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<1x3x3x4xbf16>) -> tensor<1x3x3x4xbf16>
  %4 = "tf.Transpose"(%3, %cst_1) {_tpu_replicate = "cluster", device = ""} : (tensor<1x3x3x4xbf16>, tensor<4xi32>) -> tensor<1x3x4x3xbf16>
  %5 = "tf.Conv2D"(%4, %cst) {_tpu_replicate = "cluster", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16>
  %6 = "tf.TPUReplicatedOutput"(%5) {device = ""} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xbf16>
  %7 = "tf.Cast"(%6) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  func.return %7 : tensor<1x3x2x2xf32>
}

// CHECK: func @tpu_conv(%[[ARG0:.*]]: tensor<1x3x4x3xf32>)
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>}> {device = ""} : () -> tensor<2x3x3x2xbf16>
// CHECK: %[[cast:.*]] = "tf.Cast"(%[[cst]]) <{Truncate = false}> : (tensor<2x3x3x2xbf16>) -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%[[ARG0]], %[[cast]])
// CHECK: %[[identity:.*]] = "tf.IdentityN"(%[[conv]]) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[identity]] : tensor<1x3x2x2xf32>

// -----

// Tests that `tf.BatchFunction` is inlined.

func.func @serving_default(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.BatchFunction"(%arg0, %arg1) {f = @batched_func, num_batch_threads = 1 : i64, max_batch_size = 2 : i64, batch_timeout_micros = 10000 : i64, operandSegmentSizes = array<i32: 1, 1>} : (tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>)
  return %0 : tensor<1xf32>
}
// The contents of `@serving_default` should have been inlined to `@batch_func`.
// CHECK: func.func @serving_default(%[[ARG0:.*]]: tensor<1xf32>, %[[ARG1:.*]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK-NOT: tf.BatchFunction
// CHECK: %[[ADD0:.*]] = "tf.AddV2"(%[[ARG0]], %[[ARG1]])
// CHECK: return %[[ADD0]] : tensor<1xf32>

func.func private @batched_func(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.Identity"(%arg1) : (tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.AddV2"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>
}
// The called function should be removed.
// CHECK-NOT: batched_func
