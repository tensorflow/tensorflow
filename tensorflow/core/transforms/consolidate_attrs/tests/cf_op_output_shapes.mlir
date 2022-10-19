// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %cond, %arg, %ctl = Args : () -> (tensor<i1>, tensor<*xi32>)
  // CHECK: If
  // CHECK-NOT: output_shapes
  // CHECK-SAME: -> (tensor<4xi32>)
  %If, %ctl_0 = If(%cond, %arg) {
    Tcond = i1, Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<4>],
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<*xi32>) -> (tensor<*xi32>)
}

// -----

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %index, %arg, %ctl = Args : () -> (tensor<i32>, tensor<*xi32>)
  // CHECK: Case
  // CHECK-NOT: output_shapes
  // CHECK-SAME: -> (tensor<4xi32>)
  %Case, %ctl_0 = Case(%index, %arg) {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<4>],
    branches = []
  } : (tensor<i32>, tensor<*xi32>) -> (tensor<*xi32>)
}

// -----

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %arg, %ctl = Args : () -> (tensor<*xi32>)
  // CHECK: While
  // CHECK-NOT: output_shapes
  // CHECK-SAME: -> (tensor<4xi32>)
  %While, %ctl_0 = While(%arg) {
    T = [i32], output_shapes = [#tf_type.shape<4>],
    cond = #tf_type.func<@cond, {}>, body = #tf_type.func<@body, {}>,
    parallel_iterations = 10 : i64
  } : (tensor<*xi32>) -> (tensor<*xi32>)
}

// -----

// `output_shapes` is an optional attribute in TF. When imported to TFG, if the
// attribute is not present on the NodeDef, it will be imported as `[]`. Check
// that it is ignored.

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %arg, %ctl = Args : () -> (tensor<*xi32>)
  // CHECK: While
  // CHECK-SAME: output_shapes = []
  // CHECK-SAME: -> (tensor<*xi32>)
  %While, %ctl_0 = While(%arg) {
    T = [i32], output_shapes = [],
    cond = #tf_type.func<@cond, {}>, body = #tf_type.func<@body, {}>,
    parallel_iterations = 10 : i64
  } : (tensor<*xi32>) -> (tensor<*xi32>)
}
