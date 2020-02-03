// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-promote-resources-to-args | FileCheck %s -dump-input-on-failure

// One resource, one read.
// CHECK-LABEL: func @main(%arg0: tensor<f32>) -> tensor<2xf32>
func @main() -> tensor<2xf32> {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%arg0, %[[CONST:[0-9]*]])
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD]])
  // CHECK: return %[[PACK]]
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tf.Pack"(%0, %3) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  return %4 : tensor<2xf32>
}

// -----

// One resource, two reads using different resource handles.
// CHECK-LABEL: func @main(%arg0: tensor<f32>) -> tensor<2xf32>
func @main() -> tensor<2xf32> {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg0, %[[CONST:[0-9]*]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %arg0)
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD2]])
  // CHECK: return %[[PACK]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %5 = "tf.ReadVariableOp"(%4) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%3, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%0, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  return %7 : tensor<2xf32>
}

// -----

// Two resources, two reads using different resources.
// CHECK-LABEL: func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<2xf32>
func @main() -> tensor<2xf32> {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg0, %[[CONST:[0-9]*]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %arg1)
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD2]])
  // CHECK: return %[[PACK]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tf.VarHandleOp"() {container = "", shared_name = "y"} : () -> tensor<!tf.resource<tensor<f32>>>
  %5 = "tf.ReadVariableOp"(%4) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%3, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%0, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  return %7 : tensor<2xf32>
}

// -----

// One resource with read and write.
// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.aliasing_output = 1 : i64}) -> (tensor<2xf32>, tensor<f32>)
func @main() -> tensor<2xf32> {
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg0, %{{[0-9]*}})
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %[[ADD1]])
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%arg0, %[[ADD2]])
  // CHECK: return %[[PACK]], %[[ADD1]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %4 = "tf.AddV2"(%3, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%1, %4) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  %5 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%4, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%2, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  return %7 : tensor<2xf32>
}

// -----

// A resource is passed into tf.If
// expected-error @+1 {{potential nested resource accesses in function}}
func @cond_false(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  return %arg1 : tensor<f32>
}

// expected-error @+1 {{potential nested resource accesses in function}}
func @cond_true(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %2 = "tf.AddV2"(%1, %0) {T = f32, device = ""} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

func @main() -> tensor<2xf32> attributes {tf.entry_function = {inputs = "", outputs = "result"}} {
  %0 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.Less"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = "tf.If"(%3, %1, %2) {Tcond = i1, Tin = ["tfdtype$DT_RESOURCE", "tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"],
       else_branch = @cond_false, is_stateless = false, output_shapes = ["tfshape$"],
       then_branch = @cond_true} : (tensor<i1>, tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
  %5 = "tf.Identity"(%4) : (tensor<f32>) -> tensor<f32>
  %6 = "tf.Pack"(%2, %5) {N = 2 : i64, T = f32, axis = 0 : i64, device = ""} : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  return %6 : tensor<2xf32>
}
