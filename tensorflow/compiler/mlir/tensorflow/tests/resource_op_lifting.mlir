// RUN: tf-opt %s -split-input-file -tf-resource-op-lifting | FileCheck %s -dump-input-on-failure

// Tests that resource load operations are hoisted.

// CHECK-LABEL: func @only_resource_load
func @only_resource_load() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = "tfdtype$DT_INT32"}
  // CHECK: "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: "tf_device.return"(%[[COMPUTE_RES]])
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> tensor<*xi32>

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    "tf_device.return"(%3) : (tensor<*xi32>) -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  return %1 : tensor<*xi32>
}

// -----

// Tests that resource store operations are hoisted.

// CHECK-LABEL: func @only_resource_store
func @only_resource_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() : () -> tensor<*x!tf.resource>

  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"()
  // CHECK: "tf_device.return"(%[[COMPUTE_RES]], %[[COMPUTE_RES]])
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = "tfdtype$DT_INT32"}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.SomeComputation"() : () -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %2) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    "tf_device.return"(%2) : (tensor<*xi32>) -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that a resource ops with both load and store are hoisted.

// CHECK-LABEL: func @same_resource_load_and_store
func @same_resource_load_and_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = "tfdtype$DT_INT32"}
  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: "tf_device.return"(%[[COMPUTE_RES]], %[[COMPUTE_RES]])
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = "tfdtype$DT_INT32"}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %3) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    "tf_device.return"(%3) : (tensor<*xi32>) -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that composite resource store operation is decomposed and hoisted.

// CHECK-LABEL: func @composite_resource_store
func @composite_resource_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = "tfdtype$DT_INT32"}
  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.AddV2"(%[[RES_READ_VAL]], %[[RES_READ_VAL]])
  // CHECK: "tf_device.return"(%[[COMPUTE_RES]], %[[COMPUTE_RES]])
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = "tfdtype$DT_INT32"}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    "tf.AssignAddVariableOp"(%0, %2) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    %3 = "tf.ReadVariableOp"(%0) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    "tf_device.return"(%3) : (tensor<*xi32>) -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that internal resource operations are not hoisted.

// CHECK-LABEL: func @internal_resource
func @internal_resource() -> tensor<*xi32> {

  // CHECK: %[[LAUNCH_RES:[0-9]*]] = "tf_device.launch"
  %0 = "tf_device.launch"() ( {

    // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    %1 = "tf.VarHandleOp"() : () -> tensor<*x!tf.resource>

    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]])
    %2 = "tf.ReadVariableOp"(%1) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>) -> tensor<*xi32>

    // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)

    // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[COMPUTE_RES]])
    "tf.AssignVariableOp"(%1, %3) {dtype = "tfdtype$DT_INT32"} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()

    // CHECK: "tf_device.return"(%[[COMPUTE_RES]])
    "tf_device.return"(%3) : (tensor<*xi32>) -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]
  return %0 : tensor<*xi32>
}
