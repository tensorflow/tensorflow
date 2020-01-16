// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-resource-op-lifting | FileCheck %s -dump-input-on-failure

// Tests that resource load operations are hoisted.

// CHECK-LABEL: func @only_resource_load
func @only_resource_load() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = i32}
  // CHECK: "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: tf_device.return %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> tensor<*xi32>

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  return %1 : tensor<*xi32>
}

// -----

// Tests that resource store operations are hoisted.

// CHECK-LABEL: func @only_resource_store
func @only_resource_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"()
  // CHECK: tf_device.return %[[COMPUTE_RES]], %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = i32}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.SomeComputation"() : () -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %2) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %2 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that a resource ops with both load and store are hoisted.

// CHECK-LABEL: func @same_resource_load_and_store
func @same_resource_load_and_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = i32}
  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: tf_device.return %[[COMPUTE_RES]], %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = i32}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %3 : tensor<*xi32>
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
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]])
    %2 = "tf.ReadVariableOp"(%1) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>

    // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)

    // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[COMPUTE_RES]])
    "tf.AssignVariableOp"(%1, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()

    // CHECK: tf_device.return %[[COMPUTE_RES]]
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]
  return %0 : tensor<*xi32>
}

// -----

// Tests that pass fails when there are remaining resource operationss that can
// not be lifted.

func @lifting_failure() -> tensor<*xi32> {

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // expected-error @+1 {{has remaining resource inputs that can not be lifted}}
  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
		%3 = "tf.SomeResourceOp"(%0, %2) : (tensor<*x!tf.resource>, tensor<*xi32>) -> tensor<*xi32>
    "tf.AssignVariableOp"(%0, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  return %1 : tensor<*xi32>
}
