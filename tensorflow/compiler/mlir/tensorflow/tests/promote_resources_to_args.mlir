// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-promote-resources-to-args | FileCheck %s -dump-input-on-failure

// One resource, one read.
// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
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
// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
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
// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.resource_name = "x"}, %arg1: tensor<f32> {tf.resource_name = "y"}) -> tensor<2xf32>
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
// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.aliasing_output = 1 : i64, tf.resource_name = "x"}) -> (tensor<2xf32>, tensor<f32>)
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
func @cond_false(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  return %arg1 : tensor<f32>
}

func @cond_true(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  %2 = "tf.AddV2"(%1, %0) {T = f32, device = ""} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func @main(%arg0: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
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

// -----

// Tests resource passed in as an argument is not modified and not returned.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32>
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: "tf.AddV2"(%[[ARG_0]], %[[ARG_0]])
  %1 = "tf.AddV2"(%0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return
  return
}

// -----

// Tests resource passed in as an argument is modified but not returned.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: -> tensor<f32>
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) {
  // CHECK-NEXT: %[[CONST:[a-z0-9]+]] = "tf.Const"
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]] : tensor<f32>
  return
}

// -----

// Tests last resource assign is returned as a result.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: -> tensor<f32>
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) {
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK: %[[CONST:[a-z0-9]+]] = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>}
  %1 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]] : tensor<f32>
  return
}

// -----

// Tests last resource assign is returned even when the original function
// returns the same value prior.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 1 : i64}
// CHECK-SAME: -> (tensor<f32>, tensor<f32>)
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK: %[[CONST:[a-z0-9]+]] = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>}
  %1 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]], %[[CONST]] : tensor<f32>, tensor<f32>
  return %1 : tensor<f32>
}

// -----

// Tests read interleaved between writes.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 1 : i64}
// CHECK-SAME: -> (tensor<f32>, tensor<f32>)
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) -> tensor<f32> {
  // CHECK-NEXT: %[[CONST_0:[a-z0-9]+]] = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>}
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: %[[ADD:[a-z0-9]+]] = "tf.AddV2"(%[[CONST_0]], %[[CONST_0]])
  %2 = "tf.AddV2"(%1, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: %[[CONST_1:[a-z0-9]+]] = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>}
  %3 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %3) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ADD]], %[[CONST_1]] : tensor<f32>, tensor<f32>
  return %2 : tensor<f32>
}

// -----

// Tests resource write takes on value that is from an argument.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<f32>
// CHECK-SAME: -> tensor<f32>
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<f32>)  {
  "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ARG_1]] : tensor<f32>
  return
}

// -----

// Tests first read of one resource is used as a value to write to another
// resource.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<f32>
// CHECK-SAME: -> tensor<f32>
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<!tf.resource<tensor<f32>>>)  {
  %1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ARG_1]] : tensor<f32>
  return
}

// -----

// Tests main function with multiple blocks.

// expected-error@+1 {{expects 'main' function to have 1 block, got 2}}
func @main() {
  br ^bb1
^bb1:
  return
}

// -----

// Tests main function is terminated with a non MLIR ReturnOp.

// expected-error@+1 {{expects 'main' function to have a MLIR ReturnOp}}
func @main() {
^bb0:
  tf_device.return
}

// -----

// Tests non main function with resource arguments.

func @main() {
  return
}

// expected-error@+1 {{potential nested resource accesses in function}}
func @other(%arg0: tensor<!tf.resource<tensor<f32>>>) {
  return
}

// -----

// Tests main function with invalid resource argument subtype.

// expected-error@+1 {{expects resource type of argument 0 to have one subtype, got '!tf.resource'}}
func @main(%arg0: tensor<!tf.resource>) {
  return
}

// -----

// Tests main function with invalid VarHandleOp resource subtype.

func @main() {
  // expected-error@+1 {{expects resource type to have one subtype, got '!tf.resource'}}
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource>
  return
}

// -----

// Tests resource argument has users that are not ReadVariableOp or
// AssignVariableOp.

// expected-error@+1 {{expects users of resource argument 0 to be 'tf.ReadVariableOp' or 'tf.AssignVariableOp'}}
func @main(%arg0: tensor<!tf.resource<tensor<f32>>>) -> tensor<i1> {
  %0 = "tf.VarIsInitializedOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<i1>
  return %0 : tensor<i1>
}

// -----

// Tests VarHandleOp has users that are not removed.

func @main() -> tensor<i1> {
  // expected-error@+1 {{expects no uses but used by operations: tf.UnknownOp, tf.VarIsInitializedOp}}
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf.resource<tensor<f32>>>
  %1 = "tf.VarIsInitializedOp"(%0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<i1>
  %2 = "tf.UnknownOp"(%0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<i1>
  return %1 : tensor<i1>
}
