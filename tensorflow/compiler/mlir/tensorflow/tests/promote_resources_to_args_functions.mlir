// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-promote-resources-to-args=functions="add_and_pack,read" | FILECHECK_OPTS="" FileCheck %s

module {

  // One resource, one read. The initial value of the resource is read.
  // CHECK-LABEL: func @add_and_pack(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
  func.func @add_and_pack(%arg0: tensor<i1>) -> tensor<2xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    // CHECK-NOT: "tf.ReadVariableOp"
    // CHECK: %[[CONST:.*]] = "tf.Const"()
    // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%arg1, %[[CONST]])
    // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD]])
    // CHECK: return %[[PACK]]
    %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
    %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = "tf.Pack"(%0, %3) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
    func.return %4 : tensor<2xf32>
  }

  // One resource, one read. _is_initialized is true, should be promoted.
  // CHECK-LABEL: func @read(%arg0: tensor<f32> {tf.resource_name = "x"}) -> tensor<f32>
  func.func @read() -> tensor<f32> {
    // CHECK-NOT: "tf.VarHandleOp"
    // CHECK-NOT: "tf.ReadVariableOp"
    // CHECK: return %arg0
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "x", _is_initialized = true} : () -> tensor<!tf_type.resource<tensor<f32>>>
    %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return %2 : tensor<f32>
  }
}
