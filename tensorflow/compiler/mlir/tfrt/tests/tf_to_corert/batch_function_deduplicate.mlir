// RUN: tf-tfrt-opt -tfrt-deduplicate-functions-invoked-by-batch-function %s | FileCheck %s

// This test verifies the function `compute_1` will be removed to deduplicate
// the functions invoked by BatchFunction with the same shared_name and the
// function `compute_2` will not be removed as the shared_name is different.

// CHECK-LABEL: func private @batch_0
// CHECK: f = @compute_0
func.func private @batch_0(%arg0: tensor<?x?xi32>) -> tensor<*xi32> {
  %0:2= "tf.BatchFunction"(%arg0, %arg0) {_xla_inferred_shapes = [#tf_type.shape<*>, #tf_type.shape<*>], allowed_batch_sizes = [64, 128, 256], batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", enable_large_batch_splitting = true, f = @compute_0, max_batch_size = 256 : i64, max_enqueued_batches = 10000 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 1>, shared_name = "computation"} : (tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0 : tensor<*xi32>
}

// CHECK: func private @compute_0
func.func private @compute_0(%arg0: tensor<?x?xi32> {tf._user_specified_name = "0"}, %arg1: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  func.return %arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>
}

// Batch function in batch_1 uses the same shared_name as the one in batch_0,
// so compute_1 is deduped, and compute_0 will be used here.
// CHECK-LABEL: func private @batch_1
// CHECK: f = @compute_0
// CHECK-NOT: f = @compute_1
func.func private @batch_1(%arg0: tensor<?x?xi32>) -> tensor<*xi32> {
  %0:2 = "tf.BatchFunction"(%arg0, %arg0) {_xla_inferred_shapes = [#tf_type.shape<*>, #tf_type.shape<*>], allowed_batch_sizes = [64, 128, 256], batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", enable_large_batch_splitting = true, f = @compute_1, max_batch_size = 256 : i64, max_enqueued_batches = 10000 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 1>, shared_name = "computation"} : (tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0 : tensor<*xi32>
}

// CHECK-NOT: func private @compute_1
func.func private @compute_1(%arg0: tensor<?x?xi32> {tf._user_specified_name = "0"}, %arg1: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  func.return %arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>
}

// Batch function in batch_2 uses a different shared_name from the one in
// batch_0, so it should be kept.
// CHECK-LABEL: func private @batch_2
// CHECK: f = @compute_2
func.func private @batch_2(%arg0: tensor<?x?xi32>) -> tensor<*xi32> {
  %0:2 = "tf.BatchFunction"(%arg0, %arg0) {_xla_inferred_shapes = [#tf_type.shape<*>, #tf_type.shape<*>], allowed_batch_sizes = [64, 128, 256], batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", enable_large_batch_splitting = true, f = @compute_2, max_batch_size = 256 : i64, max_enqueued_batches = 10000 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 1>, shared_name = "computation_unique_name"} : (tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0 : tensor<*xi32>
}

// CHECK: func private @compute_2
func.func private @compute_2(%arg0: tensor<?x?xi32> {tf._user_specified_name = "0"}, %arg1: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  func.return %arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>
}
