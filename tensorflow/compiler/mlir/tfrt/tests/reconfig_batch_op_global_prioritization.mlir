// RUN: tf-tfrt-opt -split-input-file -tfrt-reconfig-batch-op="tfrt-batch-queue-global-prioritization-num-threads=4" %s | FileCheck %s --dump-input=always

// -----

// The num_batch_threads is updated to 4 from the original attribute of 2,
// a mixed_priority_policy is set along with all low priority batching params
// being copied from the high priority batching params since no low priority
// settings are provided.

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [1, 2]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 2 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: low_priority_allowed_batch_sizes = [1, 2]
  // CHECK-SAME: low_priority_batch_timeout_micros = 2 : i64
  // CHECK-SAME: low_priority_max_batch_size = 2 : i64
  // CHECK-SAME: low_priority_max_enqueued_batches = 2 : i64
  // CHECK-SAME: max_batch_size = 2 : i64
  // CHECK-SAME: max_enqueued_batches = 2 : i64
  // CHECK-SAME: mixed_priority_policy = "priority_merge"
  // CHECK-SAME: num_batch_threads = 4 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [1, 2], batch_padding_policy = "PAD_UP", batch_timeout_micros = 2 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 2 : i64, max_enqueued_batches = 2 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// Same as first test, but low_priority_* parameters are already provided
// so not overriden.

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [1, 2]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 2 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: low_priority_allowed_batch_sizes = [1, 10]
  // CHECK-SAME: low_priority_batch_timeout_micros = 7 : i64
  // CHECK-SAME: low_priority_max_batch_size = 8 : i64
  // CHECK-SAME: low_priority_max_enqueued_batches = 9 : i64
  // CHECK-SAME: max_batch_size = 2 : i64
  // CHECK-SAME: max_enqueued_batches = 2 : i64
  // CHECK-SAME: mixed_priority_policy = "priority_merge"
  // CHECK-SAME: num_batch_threads = 4 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [1, 2], batch_padding_policy = "PAD_UP", batch_timeout_micros = 2 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 2 : i64, max_enqueued_batches = 2 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/", low_priority_batch_timeout_micros = 7 : i64, low_priority_max_batch_size = 8 : i64, low_priority_max_enqueued_batches = 9 : i64, low_priority_allowed_batch_sizes = [1, 10]} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
