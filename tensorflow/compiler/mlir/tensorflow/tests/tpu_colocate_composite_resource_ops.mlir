// RUN: tf-opt %s -tf-tpu-colocate-composite-resource-ops | FileCheck %s

// Tests ReadVariable op using composite device resource is wrapped inside
// tf_device.Cluster.

// CHECK-LABEL: func @testReadVariableOpColocated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
func.func @testReadVariableOpColocated(%arg0: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
  // CHECK:      tf_device.replicate
  // CHECK-SAME: (%[[ARG0]] as %[[RI_0:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
  tf_device.replicate(%arg0 as %arg1: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
    _mirrored_variable_indices = [0], _replicated_input_indices = [-1],
    devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]},
    n = 2 : i32} {
     // CHECK:      %[[RESOURCE_OUT:.*]] = "tf_device.launch"()
     // CHECK-NEXT:   %[[READ_OUT:.*]] = "tf.ReadVariableOp"(%[[RI_0]])
     // CHECK-NEXT:   tf_device.return %[[READ_OUT]]
     // CHECK-NEXT: TPU_REPLICATED_CORE_0
     %0 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
     %1 = "tf.A"() : () -> (tensor<2x!tf_type.string>)
     "tf_device.launch"() ({
       "tf.TPUExecuteAndUpdateVariables"(%arg1, %1) {device_var_reads_indices = [0], device_var_updates_indices = [-1]} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<2x!tf_type.string>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
    "tf_device.launch"() ({
      // CHECK:  "tf.B"(%[[RESOURCE_OUT]])
      "tf.B"(%0) : (tensor<4xf32>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
    tf_device.return
  }
  func.return
}

// CHECK-LABEL: func @testReadVariableOpAfterIdentityColocated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
func.func @testReadVariableOpAfterIdentityColocated(%arg0: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
  // CHECK:      tf_device.replicate
  // CHECK-SAME: (%[[ARG0]] as %[[RI_0:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
  tf_device.replicate(%arg0 as %arg1: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
    _mirrored_variable_indices = [0], _replicated_input_indices = [-1],
    devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]},
    n = 2 : i32} {
     // CHECK:      %[[IDENTITY_OUT:.*]] = "tf.Identity"(%[[RI_0]])
     // CHECK:      %[[RESOURCE_OUT:.*]] = "tf_device.launch"()
     // CHECK-NEXT:   %[[READ_OUT:.*]] = "tf.ReadVariableOp"(%[[IDENTITY_OUT]])
     // CHECK-NEXT:   tf_device.return %[[READ_OUT]]
     // CHECK-NEXT: TPU_REPLICATED_CORE_0
     %0 = "tf.Identity"(%arg1) : (tensor<*x!tf_type.resource<tensor<4xf32>>>) -> tensor<*x!tf_type.resource<tensor<4xf32>>>
     %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
     %2 = "tf.A"() : () -> (tensor<2x!tf_type.string>)
     "tf_device.launch"() ({
       "tf.TPUExecuteAndUpdateVariables"(%arg1, %2) {device_var_reads_indices = [0], device_var_updates_indices = [-1]} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<2x!tf_type.string>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
    "tf_device.launch"() ({
      // CHECK:  "tf.B"(%[[RESOURCE_OUT]])
      "tf.B"(%1) : (tensor<4xf32>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
    tf_device.return
  }
  func.return
}

// Tests AssignVariable op using composite device resource is wrapped inside
// tf_device.Cluster.

// CHECK-LABEL: func @testAssignVariableOpColocated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
func.func @testAssignVariableOpColocated(%arg0: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
  // CHECK:      tf_device.replicate
  // CHECK-SAME: (%[[ARG0]] as %[[RI_0:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
  tf_device.replicate(%arg0 as %arg1: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
    _mirrored_variable_indices = [0], _replicated_input_indices = [-1],
    devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]},
    n = 2 : i32} {
     // CHECK:      %[[VAL_OUT:.*]] = "tf.A"() : () -> tensor<4xf32>
     // CHECK:      "tf_device.launch"()
     // CHECK-NEXT:   "tf.AssignVariableOp"(%[[RI_0]], %[[VAL_OUT]])
     // CHECK-NEXT:   tf_device.return
     // CHECK-NEXT: TPU_REPLICATED_CORE_0
     %1 = "tf.A"() : () -> (tensor<4xf32>)
     "tf.AssignVariableOp"(%arg1, %1) : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<4xf32>) -> ()
     %2 = "tf.B"() : () -> (tensor<2x!tf_type.string>)
     "tf_device.launch"() ({
       "tf.TPUExecuteAndUpdateVariables"(%arg1, %2) {device_var_reads_indices = [0], device_var_updates_indices = [-1]} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<2x!tf_type.string>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
    tf_device.return
  }
  func.return
}

// Tests tf_device.replicate op not running on TPU devices ignored.

// CHECK-LABEL: func @testNonTPUDeviceReplicationIgnored
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
func.func @testNonTPUDeviceReplicationIgnored(%arg0: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
  // CHECK:      tf_device.replicate
  // CHECK-SAME: (%[[ARG0]] as %[[RI_0:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>)
  tf_device.replicate(%arg0 as %arg1: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
    _mirrored_variable_indices = [0], _replicated_input_indices = [-1],
    devices = {TPU_REPLICATED_HOST = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:CPU:1"]},
    n = 2 : i32} {
     // CHECK:      %[[VAL_OUT:.*]] = "tf.A"() : () -> tensor<4xf32>
     // CHECK-NEXT: "tf.AssignVariableOp"(%[[RI_0]], %[[VAL_OUT]])
     %1 = "tf.A"() : () -> (tensor<4xf32>)
     "tf.AssignVariableOp"(%arg1, %1) : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<4xf32>) -> ()
     %2 = "tf.B"() : () -> (tensor<2x!tf_type.string>)
     "tf_device.launch"() ({
       "tf.TPUExecuteAndUpdateVariables"(%arg1, %2) {device_var_reads_indices = [0], device_var_updates_indices = [-1]} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<2x!tf_type.string>) -> ()
       tf_device.return
    }) {device = "TPU_REPLICATED_HOST"} : () -> ()
    tf_device.return
  }
  func.return
}
