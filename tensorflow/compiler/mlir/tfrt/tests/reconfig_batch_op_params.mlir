// RUN: tf-tfrt-opt -split-input-file -tfrt-reconfig-batch-op="tfrt-min-num-batch-threads=4 tfrt-min-max-enqueued-batches=4 tfrt-num-batch-threads=3 tfrt-max-batch-size=3 tfrt-batch-timeout-micros=3 tfrt-allowed-batch-sizes=3,4 tfrt-max-enqueued-batches=3 tfrt-enable-large-batch-splitting=true tfrt-mixed-priority-batching-policy=priority_merge" %s | FileCheck %s --dump-input=always

// -----

// The num_batch_threads is updated to 3 from the original attribute of 2,
// overriding the min_num_batch_threads of 4.
// The max_batch_size is updated to 3 from the original attribute of 2.
// The batch_timeout_micros is updated to 3 from the original attribute of 2.
// The allowed_batch_sizes is updated to [3, 4] from the original attribute of
// [1, 2].
// The max_enqueued_batches is updated to 3 from the original attribute of 2,
// overriding the min_max_enqueued_batches of 4.
// The enable_large_batch_splitting is updated to true from the original
// attribute of false.
// The mixed_priority_policy is updated to "priority_merge" from the original
// attribute of "".

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [3, 4]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 3 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = true
  // CHECK-SAME: max_batch_size = 3 : i64
  // CHECK-SAME: max_enqueued_batches = 3 : i64
  // CHECK-SAME: mixed_priority_policy = "priority_merge"
  // CHECK-SAME: num_batch_threads = 3 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [1, 2], batch_padding_policy = "PAD_UP", batch_timeout_micros = 2 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 2 : i64, max_enqueued_batches = 2 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
