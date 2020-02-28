// RUN: tf-opt -split-input-file -tf-tpu-merge-variables-with-execute %s | FileCheck %s --dump-input=fail

// Tests that the pass merges only variable reads/writes on the same device.

// CHECK-LABEL: func @merge_same_device_variables
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<*x!tf.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:.*]]: tensor<*x!tf.resource<tensor<16xf32>>>
// CHECK-SAME: %[[ARG_3:.*]]: tensor<!tf.string>
func @merge_same_device_variables(
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg2: tensor<*x!tf.resource<tensor<16xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
  %arg3: tensor<!tf.string>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: %[[ID_0:.*]] = "tf.IdentityN"(%[[ARG_0]])
      %id0 = "tf.IdentityN"(%arg0) {device = "/job:localhost/replica:0/task:0/device:TPU:0"}
        : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<*x!tf.resource<tensor<32xf32>>>
      // CHECK-NEXT: %[[READ_2:.*]] = "tf.ReadVariableOp"(%[[ARG_2]])
      %read0 = "tf.ReadVariableOp"(%id0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<64xf32>>>) -> tensor<64xf32>
      %read2 = "tf.ReadVariableOp"(%arg2) : (tensor<*x!tf.resource<tensor<16xf32>>>) -> tensor<16xf32>
      // CHECK-NEXT: %[[EXE:.*]] = "tf.TPUExecuteAndUpdateVariables"(%[[ID_0]], %[[ARG_1]], %[[READ_2]], %[[ARG_3]])
      // CHECK-SAME: device_var_reads_indices = [0, 1],
      // CHECK-SAME: device_var_updates_indices = [0, -1]
      %execute:2 = "tf.TPUExecute"(%read0, %read1, %read2, %arg3) {
        Targs = [tensor<32xf32>, tensor<64xf32>, tensor<16xf32>],
        Tresults = [tensor<32xf32>, tensor<16xf32>],
        device = "/job:localhost/replica:0/task:0/device:TPU:0"}
        : (tensor<32xf32>, tensor<64xf32>, tensor<16xf32>, tensor<!tf.string>) -> (tensor<32xf32>, tensor<16xf32>)
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_2]], %[[EXE]])
      "tf.AssignVariableOp"(%id0, %execute#0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      "tf.AssignVariableOp"(%arg2, %execute#1) : (tensor<*x!tf.resource<tensor<16xf32>>>, tensor<16xf32>) -> ()
      // CHECK-NEXT: tf_executor.yield
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// -----

// Tests that the pass do not check devices for replicated region.

// CHECK-LABEL: func @merge_replicated_variables
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>, %[[ARG_1:.*]]: tensor<!tf.string>,
// CHECK-SAME: %[[ARG_2:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>,
// CHECK-SAME: %[[ARG_3:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>
func @merge_replicated_variables(
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg1: tensor<!tf.string>,
  %arg2: tensor<*x!tf.resource<tensor<32xf32>>>,
  %arg3: tensor<*x!tf.resource<tensor<32xf32>>>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // CHECK-NEXT: tf_device.replicate([%[[ARG_2]], %[[ARG_3]]] as %[[R_ARG:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>)
      tf_device.replicate([%arg2, %arg3] as %r: tensor<*x!tf.resource<tensor<32xf32>>>) {n = 2 : i32} {
        // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[READ_0]], %[[R_ARG]], %[[ARG_1]])
        // CHECK-SAME: device_var_reads_indices = [1],
        // CHECK-SAME: device_var_updates_indices = [0]
        %read1 = "tf.ReadVariableOp"(%r) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
        %execute = "tf.TPUExecute"(%read0, %read1, %arg1)
          : (tensor<32xf32>, tensor<32xf32>, tensor<!tf.string>) -> tensor<32xf32>
        "tf.AssignVariableOp"(%r, %execute) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
        // CHECK-NEXT: tf_device.return
        tf_device.return
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: tf_executor.yield
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// -----

// Tests that the pass do not merge reads/assigns if there are interferencing
// accesses in between.

// CHECK-LABEL: func @interferencing_accesses
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<*x!tf.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:.*]]: tensor<32xf32>
// CHECK-SAME: %[[ARG_3:.*]]: tensor<!tf.string>
// CHECK-SAME: %[[ARG_4:.*]]: tensor<*x!tf.resource<tensor<8xf32>>>
// CHECK-SAME: %[[ARG_5:.*]]: tensor<*x!tf.resource<tensor<2xf32>>>
// CHECK-SAME: %[[ARG_6:.*]]: tensor<2xf32>
func @interferencing_accesses(
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg2: tensor<32xf32>,
  %arg3: tensor<!tf.string>,
  %arg4: tensor<*x!tf.resource<tensor<8xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg5: tensor<*x!tf.resource<tensor<2xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg6: tensor<2xf32>) -> (tensor<8xf32>) {
  %graph = tf_executor.graph {
    // CHECK: tf_executor.island
    %island:2 = tf_executor.island {
      // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // CHECK-NEXT: %[[READ_5:.*]] = "tf.ReadVariableOp"(%[[ARG_5]])
      %read5 = "tf.ReadVariableOp"(%arg5) : (tensor<*x!tf.resource<tensor<2xf32>>>) -> tensor<2xf32>
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[ARG_2]])
      "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_5]], %[[ARG_6]])
      "tf.AssignVariableOp"(%arg5, %arg6) : (tensor<*x!tf.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
      %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<64xf32>>>) -> tensor<64xf32>
      %read2 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      // CHECK-NEXT: %[[EXE:.*]]:2 = "tf.TPUExecuteAndUpdateVariables"(%[[READ_0]], %[[ARG_1]], %[[ARG_4]], %[[READ_5]], %[[ARG_3]])
      // CHECK-SAME: device_var_reads_indices = [1, 2],
      // CHECK-SAME: device_var_updates_indices = [1, -1]
      %execute:3 = "tf.TPUExecute"(%read0, %read1, %read2, %read5, %arg3) {
        Targs = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>],
        Tresults = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>],
        device = "/job:localhost/replica:0/task:0/device:TPU:0"}
        : (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>, tensor<!tf.string>)
          -> (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>)
      "tf.AssignVariableOp"(%arg1, %execute#1) : (tensor<*x!tf.resource<tensor<64xf32>>>, tensor<64xf32>) -> ()
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#0)
      "tf.AssignVariableOp"(%arg0, %execute#0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // CHECK-NEXT: %[[READ_3:.*]] = "tf.ReadVariableOp"(%[[ARG_4]])
      %read3 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf.resource<tensor<8xf32>>>) -> tensor<8xf32>
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_4]], %[[EXE]]#1)
      "tf.AssignVariableOp"(%arg4, %execute#2) : (tensor<*x!tf.resource<tensor<8xf32>>>, tensor<8xf32>) -> ()
      // CHECK-NEXT: tf_executor.yield %[[READ_3]]
      tf_executor.yield %read3 : tensor<8xf32>
    }
    tf_executor.fetch %island#0 : tensor<8xf32>
  }
  return %graph : tensor<8xf32>
}

// -----

// Tests that the pass do not merge for an execute node that has multiple inputs
// read from the same variable.

// CHECK-LABEL: func @do_not_merge_multi_read
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<!tf.string>
func @do_not_merge_multi_read(
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<!tf.string>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // CHECK-NEXT: %[[READ_1:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
      %read1 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // CHECK-NEXT: %[[EXE:.*]] = "tf.TPUExecute"(%[[READ_0]], %[[READ_1]], %[[ARG_1]])
      %execute = "tf.TPUExecute"(%read0, %read1, %arg1) {
        Targs = [tensor<32xf32>, tensor<32xf32>], Tresults = [tensor<32xf32>],
        device = "/job:localhost/replica:0/task:0/device:TPU:0"}
        : (tensor<32xf32>, tensor<32xf32>, tensor<!tf.string>) -> (tensor<32xf32>)
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]])
      "tf.AssignVariableOp"(%arg0, %execute) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // CHECK-NEXT: tf_executor.yield
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}

// -----

// Tests that the pass do not merge for an execute node that has multiple
// outputs used to update a variable.

// CHECK-LABEL: func @do_not_merge_multi_assign
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<!tf.string>
func @do_not_merge_multi_assign(
  %arg0: tensor<*x!tf.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<!tf.string>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    %island = tf_executor.island {
      // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
      %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
      // CHECK-NEXT: %[[EXE:.*]]:2 = "tf.TPUExecute"(%[[READ_0]], %[[ARG_1]])
      %execute:2 = "tf.TPUExecute"(%read0, %arg1) {
        Targs = [tensor<32xf32>], Tresults = [tensor<32xf32>, tensor<32xf32>],
        device = "/job:localhost/replica:0/task:0/device:TPU:0"}
        : (tensor<32xf32>, tensor<!tf.string>) -> (tensor<32xf32>, tensor<32xf32>)
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#0)
      "tf.AssignVariableOp"(%arg0, %execute#0) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#1)
      "tf.AssignVariableOp"(%arg0, %execute#1) : (tensor<*x!tf.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
      // CHECK-NEXT: tf_executor.yield
      tf_executor.yield
    }
    tf_executor.fetch %island : !tf_executor.control
  }
  return
}
