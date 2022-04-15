// RUN: tf-tfrt-opt -pass-pipeline='func.func(tf-tensor-device-copy),tf-to-tfrt' %s | FileCheck %s --dump-input=fail

func.func private @batched_function(%arg0: tensor<1x3xf32> {tf._user_specified_name = "0"}, %arg1: tensor<*x!tf_type.resource>) -> tensor<1x3xf32> attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<1x3xf32>
  %1 = "tf.AddV2"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "batch/BatchFunction:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  // CHECK: [[var:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle
  // CHECK: tfrt_fallback_async.batch_function(%arg0) @batched_function ({{%.*}}, [[var]])
  // CHECK-NOT: device
  // CHECK-SAME: Tcaptured = [!corert.resource]
  // CHECK-SAME: Tin = [f32]
  // CHECK-SAME: Tout = [f32]
  // CHECK-SAME: allowed_batch_sizes = [6]
  // CHECK-SAME: batch_timeout_micros = 100000 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: max_batch_size = 6 : i64
  // CHECK-SAME: max_enqueued_batches = 10 : i64
  // CHECK-SAME: num_batch_threads = 1 : i64
  // CHECK-SAME: operand_segment_sizes = dense<1> : vector<2xi32>
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0, %0) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = dense<1> : vector<2xi32>, shared_name = "batch/"} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
