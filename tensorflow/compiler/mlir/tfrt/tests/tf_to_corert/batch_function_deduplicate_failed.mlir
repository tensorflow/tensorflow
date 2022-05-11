// RUN: not tf-tfrt-opt -tfrt-deduplicate-functions-invoked-by-batch-function %s 2>&1 | FileCheck %s

// This test verifies the error when two functions are different but invoked by
// the batch functions with same shared_name.

func.func private @batch_0(%arg0: tensor<?x?xi32>) -> tensor<*xi32> {
  %0:2 = "tf.BatchFunction"(%arg0, %arg0) {_xla_inferred_shapes = [#tf_type.shape<*>, #tf_type.shape<*>], allowed_batch_sizes = [64, 128, 256], batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", enable_large_batch_splitting = true, f = @compute_0, max_batch_size = 256 : i64, max_enqueued_batches = 10000 : i64, num_batch_threads = 2 : i64, operand_segment_sizes = dense<1> : vector<2xi32>, shared_name = "computation"} : (tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0 : tensor<*xi32>
}

func.func private @compute_0(%arg0: tensor<?x?xi32> {tf._user_specified_name = "0"}, %arg1: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  func.return %arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>
}

func.func private @batch_3(%arg0: tensor<?x1xi32>) -> tensor<*xi32> {
  %0:2 = "tf.BatchFunction"(%arg0, %arg0) {_xla_inferred_shapes = [#tf_type.shape<*>, #tf_type.shape<*>], allowed_batch_sizes = [64, 128, 256], batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", enable_large_batch_splitting = true, f = @compute_3, max_batch_size = 256 : i64, max_enqueued_batches = 10000 : i64, num_batch_threads = 2 : i64, operand_segment_sizes = dense<1> : vector<2xi32>, shared_name = "computation"} : (tensor<?x1xi32>, tensor<?x1xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0 : tensor<*xi32>
}

// compute_3 has different argument types from compute_1, thus error is reported.
// CHECK: error: func_ops for BatchFunctionOp with the same shared name are different
func.func private @compute_3(%arg0: tensor<?x1xi32> {tf._user_specified_name = "0"}, %arg1: tensor<?x1xi32>) -> (tensor<?x1xi32>, tensor<?x1xi32>) {
  func.return %arg0, %arg1 : tensor<?x1xi32>, tensor<?x1xi32>
}
