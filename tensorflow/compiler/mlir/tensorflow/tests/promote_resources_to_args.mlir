// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-promote-resources-to-args | FILECHECK_OPTS="" FileCheck %s

// One resource, one read. The initial value of the resource is read.
// CHECK-LABEL: func @main(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> {
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

// -----

// One resource, one read. _is_initialized is false, shouldn't be promoted.
// CHECK-LABEL: func @main()
func.func @main() -> tensor<f32> {
  // CHECK: "tf.VarHandleOp"
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x", _is_initialized = false} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// One resource, one read. _is_initialized is true, should be promoted.
// CHECK-LABEL: func @main
// CHECK-SAME: ({{%.+}}: tensor<f32> {tf.resource_name = "x"})
func.func @main() -> tensor<f32> {
  // CHECK-NOT: "tf.VarHandleOp"
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x", _is_initialized = true} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// One resource, one write. The initial value of the resource is not read.
// CHECK-LABEL: func @main(%arg0: tensor<i1>) -> (tensor<f32> {tf.resource_name = "x"})
func.func @main(%arg0: tensor<i1>) {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: %[[RES:.*]] = "tf.Const"()
  // CHECK: return %[[RES]]
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  func.return
}

// -----

// One resource, two reads using different resource handles.
// CHECK-LABEL: func @main(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: %[[CONST:.*]] = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}>
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg1, %[[CONST]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %arg1)
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD2]])
  // CHECK: return %[[PACK]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %5 = "tf.ReadVariableOp"(%4) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%3, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%0, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  func.return %7 : tensor<2xf32>
}

// -----

// Two resources, two reads using different resources.
// CHECK-LABEL: func @main(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.resource_name = "x"}, %arg2: tensor<f32> {tf.resource_name = "y"}) -> tensor<2xf32>
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK-NOT: "tf.VarHandleOp"
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg1, %[[CONST]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %arg2)
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD2]])
  // CHECK: return %[[PACK]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.AddV2"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = "tf.VarHandleOp"() {container = "", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %5 = "tf.ReadVariableOp"(%4) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%3, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%0, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  func.return %7 : tensor<2xf32>
}

// -----

// One resource with read and write. The initial value of the resource is read.
// CHECK-LABEL: func @main(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.aliasing_output = 1 : i64, tf.resource_name = "x"}) -> (tensor<2xf32>, tensor<f32>)
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%arg1, %[[CONST]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %[[ADD1]])
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%arg1, %[[ADD2]])
  // CHECK: return %[[PACK]], %[[ADD1]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %4 = "tf.AddV2"(%3, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%1, %4) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %5 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%4, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%2, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  func.return %7 : tensor<2xf32>
}

// -----

// One resource with read and write. The initial value of the resource is not read.
// CHECK-LABEL: func @main(%arg0: tensor<i1>) -> (tensor<2xf32>, tensor<f32> {tf.resource_name = "x"})
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: %[[CONST:.*]] = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}>
  // CHECK: %[[ADD1:[0-9]*]] = "tf.AddV2"(%[[CONST]], %[[CONST]])
  // CHECK: %[[ADD2:[0-9]*]] = "tf.AddV2"(%[[ADD1]], %[[ADD1]])
  // CHECK: %[[PACK:[0-9]*]] = "tf.Pack"(%[[CONST]], %[[ADD2]])
  // CHECK: return %[[PACK]], %[[ADD1]]

  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %4 = "tf.AddV2"(%3, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%1, %4) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %5 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %6 = "tf.AddV2"(%4, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %7 = "tf.Pack"(%2, %6) : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  func.return %7 : tensor<2xf32>
}

// -----

// A resource is passed into tf.If
func.func @cond_false(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  func.return %arg1 : tensor<f32>
}

func.func @cond_true(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %2 = "tf.AddV2"(%1, %0) {T = f32, device = ""} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// CHECK-LABEL: func @main(%arg0: tensor<i1>, %arg1: tensor<f32> {tf.resource_name = "x"}) -> tensor<2xf32>
func.func @main(%arg0: tensor<i1>) -> tensor<2xf32> attributes {tf.entry_function = {inputs = "", outputs = "result"}} {
  %0 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.Less"(%2, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = "tf.If"(%3, %1, %2) {Tcond = i1, Tin = ["tfdtype$DT_RESOURCE", "tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"],
       else_branch = @cond_false, is_stateless = false,then_branch = @cond_true} :
       (tensor<i1>, tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> tensor<f32>
  %5 = "tf.Identity"(%4) : (tensor<f32>) -> tensor<f32>
  %6 = "tf.Pack"(%2, %5) {N = 2 : i64, T = f32, axis = 0 : i64, device = ""} : (tensor<f32>, tensor<f32>) -> tensor<2xf32>
  func.return %6 : tensor<2xf32>
}

// -----

// Tests resource passed in as an argument is not modified and not returned.

// CHECK-LABEL: func @main
// CHECK-SAME: %arg0: tensor<i1>
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<f32>
func.func @main(%arg0: tensor<i1>, %arg1: tensor<!tf_type.resource<tensor<f32>>>) {
  %0 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: "tf.AddV2"(%[[ARG_1]], %[[ARG_1]])
  %1 = "tf.AddV2"(%0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return
  func.return
}

// -----

// Tests resource passed in as an argument is modified but not returned.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %arg1: tensor<i1>
// CHECK-SAME: -> tensor<f32>
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<i1>) {
  // CHECK-NEXT: %[[CONST:.*]] = "tf.Const"
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]] : tensor<f32>
  func.return
}

// -----

// Tests last resource assign is returned as a result.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %arg1: tensor<i1>
// CHECK-SAME: -> tensor<f32>
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<i1>) {
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK: %[[CONST:.*]] = "tf.Const"() <{value = dense<1.050000e+03> : tensor<f32>}>
  %1 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]] : tensor<f32>
  func.return
}

// -----

// Tests last resource assign is returned even when the original function
// returns the same value prior.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 1 : i64}
// CHECK-SAME: %arg1: tensor<i1>
// CHECK-SAME: -> (tensor<f32>, tensor<f32>)
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<i1>) -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK: %[[CONST:.*]] = "tf.Const"() <{value = dense<1.050000e+03> : tensor<f32>}>
  %1 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]], %[[CONST]] : tensor<f32>, tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// Tests read interleaved between writes.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 1 : i64}
// CHECK-SAME: %arg1: tensor<i1>
// CHECK-SAME: -> (tensor<f32>, tensor<f32>)
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<i1>) -> tensor<f32> {
  // CHECK-NEXT: %[[CONST_0:.*]] = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}>
  %0 = "tf.Const"() {value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: %[[ADD:[a-z0-9]+]] = "tf.AddV2"(%[[CONST_0]], %[[CONST_0]])
  %2 = "tf.AddV2"(%1, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: %[[CONST_1:.*]] = "tf.Const"() <{value = dense<1.050000e+03> : tensor<f32>}>
  %3 = "tf.Const"() {value = dense<1.050000e+03> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ADD]], %[[CONST_1]] : tensor<f32>, tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// Tests resource write takes on value that is from an argument.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<f32>
// CHECK-SAME: -> tensor<f32>
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<f32>)  {
  "tf.AssignVariableOp"(%arg0, %arg1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ARG_1]] : tensor<f32>
  func.return
}

// -----

// Tests removal of dead local variables.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<2xf32>) {
  // CHECK-NOT: tf.MlirLocalVarOp
  // CHECK-NOT: tf.AssignVariableOp
  %0 = "tf.MlirLocalVarOp"() : () -> tensor<!tf_type.resource<tensor<2xf32>>>
  "tf.AssignVariableOp"(%0, %arg0) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  func.return
}

// -----

// Tests first read of one resource is used as a value to write to another
// resource.

// CHECK-LABEL: func @main
// CHECK-SAME: %{{[a-z0-9]+}}: tensor<f32> {tf.aliasing_output = 0 : i64}
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<f32>
// CHECK-SAME: -> tensor<f32>
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<!tf_type.resource<tensor<f32>>>)  {
  %1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[ARG_1]] : tensor<f32>
  func.return
}

// -----

// Tests if local variables that are dead after resource op lifting are removed.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> tensor<2xf32> {
  // CHECK-NOT: tf.MlirLocalVarOp
  // CHECK-NOT: tf.AssignVariableOp
  %0 = "tf.MlirLocalVarOp"() : () -> tensor<!tf_type.resource<tensor<2xf32>>>
  %1 = "tf._SomeOp"() : () -> tensor<2xf32>
  "tf.AssignVariableOp"(%0, %1) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  %2 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
  func.return %2 : tensor<2xf32>
}
func.func private @callee(%arg0: tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32> {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}


// -----

// Tests main function with multiple blocks.

// expected-error@+1 {{expects function 'main' to have 1 block, got 2}}
func.func @main() {
  cf.br ^bb1
^bb1:
  func.return
}

// -----

// Tests main function is terminated with a non MLIR ReturnOp.

// expected-error@+1 {{expects function 'main' to have a MLIR ReturnOp}}
func.func @main() {
^bb0:
  tf_device.return
}

// -----

// Tests main function with invalid resource argument subtype.

// expected-error@+1 {{expects resource type of argument 0 to have one subtype, got '!tf_type.resource'}}
func.func @main(%arg0: tensor<!tf_type.resource>) {
  func.return
}

// -----

// Tests main function with invalid VarHandleOp resource subtype.

func.func @main() {
  // expected-error @+1 {{must have exactly one subtype in the result resource type}}
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource>
  func.return
}

// -----

// Tests resource argument has users that are not ReadVariableOp or
// AssignVariableOp.

// expected-error@+1 {{expects users of resource argument 0 to be 'tf.ReadVariableOp' or 'tf.AssignVariableOp', got [tf.UnknownOp, tf.VarIsInitializedOp]}}
func.func @main(%arg0: tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1> {
  %0 = "tf.VarIsInitializedOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  %1 = "tf.UnknownOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// Tests VarHandleOp has users that are not removed.

func.func @main() -> tensor<i1> {
  // expected-error@+1 {{expects users to be 'tf.ReadVariableOp' or 'tf.AssignVariableOp', got [tf.UnknownOp, tf.VarIsInitializedOp]}}
  %0 = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %1 = "tf.VarIsInitializedOp"(%0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  %2 = "tf.UnknownOp"(%0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
