// RUN: tf-tfrt-opt -split-input-file -tfrt-reconfig-batch-op="tfrt-min-num-batch-threads=2 tfrt-min-max-enqueued-batches=3 tfrt-batch-padding-policy=PAD_UP" %s | FileCheck %s --dump-input=always

// -----

// The num_batch_threads is lowered bound to 2 from the original attribute of 1

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [6]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 100000 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: max_batch_size = 6 : i64
  // CHECK-SAME: max_enqueued_batches = 10 : i64
  // CHECK-SAME: num_batch_threads = 2 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [6], batch_padding_policy = "PAD_UP", batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// The num_batch_threads remains 3 (the same as the original attribute)

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [6]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 100000 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: max_batch_size = 6 : i64
  // CHECK-SAME: max_enqueued_batches = 10 : i64
  // CHECK-SAME: num_batch_threads = 3 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [6], batch_padding_policy = "PAD_UP", batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 3 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// The max_enqueued_batches is changed to 3 from the original attribute of 2

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [6]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 100000 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: max_batch_size = 6 : i64
  // CHECK-SAME: max_enqueued_batches = 3 : i64
  // CHECK-SAME: num_batch_threads = 2 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [6], batch_padding_policy = "PAD_UP", batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 2 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// The max_enqueued_batches remains 10 (the same as the original attribute)

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %2 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.BatchFunction"
  // CHECK-SAME: allowed_batch_sizes = [6]
  // CHECK-SAME: batch_padding_policy = "PAD_UP"
  // CHECK-SAME: batch_timeout_micros = 100000 : i64
  // CHECK-SAME: batching_queue = ""
  // CHECK-SAME: container = ""
  // CHECK-SAME: enable_large_batch_splitting = false
  // CHECK-SAME: max_batch_size = 6 : i64
  // CHECK-SAME: max_enqueued_batches = 10 : i64
  // CHECK-SAME: num_batch_threads = 3 : i64
  // CHECK-SAME: shared_name = "batch/"
  %1 = "tf.BatchFunction"(%arg0) {allowed_batch_sizes = [6], batch_padding_policy = "PAD_UP", batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 3 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = "batch/"} : (tensor<1x3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
