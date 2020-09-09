// RUN: tf-opt %s -tf-parallize-embedding-params-ops -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @two_shards
func @two_shards(%arg0: tensor<*x!tf.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf.resource<tensor<8xf32>>>, %arg2: tensor<*x!tf.resource<tensor<8xf32>>>, %arg3: tensor<*x!tf.resource<tensor<8xf32>>>) {
  tf_executor.graph {
    %control = tf_executor.island {
      // CHECK: "tf_device.parallel_execute"
      // CHECK:   "tf.ReadVariableOp"
      // CHECK:   "tf.ReadVariableOp"
      // CHECK:   "tf.LoadTPUEmbeddingAdagradParameters"
      // CHECK:   tf_device.return
      // CHECK:   "tf.ReadVariableOp"
      // CHECK:   "tf.ReadVariableOp"
      // CHECK:   "tf.LoadTPUEmbeddingAdagradParameters"
      // CHECK:   tf_device.return
      %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %2 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %3 = "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:0/device:CPU:0", num_shards = 2 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      "tf.LoadTPUEmbeddingAdagradParameters"(%2, %3) {config = "", device = "/job:worker/replica:0/task:1/device:CPU:0", num_shards = 2 : i64, shard_id = 1 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  return
}

// Verifies that resource reads shared across two shards are kept outside the
// parallel_execute op.

// CHECK-LABEL: func @shared_reads
func @shared_reads(%arg0: tensor<*x!tf.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf.resource<tensor<8xf32>>>) {
  tf_executor.graph {
    %control = tf_executor.island {
      // CHECK: "tf.ReadVariableOp"
      %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      // CHECK: "tf.ReadVariableOp"
      %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>

      // CHECK: "tf_device.parallel_execute"
      // CHECK:   "tf.LoadTPUEmbeddingAdagradParameters"
      // CHECK:   tf_device.return
      // CHECK:   "tf.LoadTPUEmbeddingAdagradParameters"
      // CHECK:   tf_device.return
      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:0/device:CPU:0", num_shards = 2 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:1/device:CPU:0", num_shards = 2 : i64, shard_id = 1 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  return
}

// Verifies that if the resource variables are used in ops other than read
// variable op whose semantics are not known then the function is kept
// unchanged.

// CHECK-LABEL: func @update_var
func @update_var(%arg0: tensor<*x!tf.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf.resource<tensor<8xf32>>>, %arg2: tensor<*x!tf.resource<tensor<8xf32>>>) {
  tf_executor.graph {
    // CHECK-NOT: tf_device.parallel_execute
    %control = tf_executor.island {
      %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:0/device:CPU:0", num_shards = 2 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()

      %2 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %zeros = "tf.Const"() {value = dense<1.0> : tensor<8xf32>} : () -> tensor<8xf32>
      "tf.AssignVariableOp"(%arg2, %zeros) : (tensor<*x!tf.resource<tensor<8xf32>>>, tensor<8xf32>) -> ()
      %3 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      "tf.LoadTPUEmbeddingAdagradParameters"(%2, %3) {config = "", device = "/job:worker/replica:0/task:1/device:CPU:0", num_shards = 2 : i64, shard_id = 1 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  return
}

// -----

func @invalid_shard_range(%arg0: tensor<*x!tf.resource<tensor<8xf32>>>, %arg1: tensor<*x!tf.resource<tensor<8xf32>>>) {
  tf_executor.graph {
    %control = tf_executor.island {
      // expected-error @-1 {{require continuous range of shards}}
      %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>

      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:0/device:CPU:0", num_shards = 3 : i64, shard_id = 0 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      "tf.LoadTPUEmbeddingAdagradParameters"(%0, %1) {config = "", device = "/job:worker/replica:0/task:1/device:CPU:0", num_shards = 3 : i64, shard_id = 3 : i64, table_id = -1 : i64, table_name = "param_table"} : (tensor<8xf32>, tensor<8xf32>) -> ()
      tf_executor.yield
    }
    tf_executor.fetch %control : !tf_executor.control
  }
  return
}
