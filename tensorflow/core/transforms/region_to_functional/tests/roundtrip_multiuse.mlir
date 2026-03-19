// RUN: tfg-transforms-opt --tfg-functional-to-region --tfg-region-to-functional %s | FileCheck %s

// Check that conversion back to functional form can re-use the same function
// more than once.

// CHECK: tfg.func @B
tfg.func @B(%arg0: tensor<i32>) -> (tensor<i32>) {
  %C, %ctl = C(%arg0) : (tensor<i32>) -> (tensor<i32>)
  return(%C) : tensor<i32>
}

// CHECK: tfg.func @A
tfg.func @A(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Case
  // CHECK-SAME: branches = [#tf_type.func<@B, {}>, #tf_type.func<@B, {}>]
  %Case, %ctl = Case(%arg0, %arg0) {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [
      #tf_type.func<@B, {}>,
      #tf_type.func<@B, {}>
    ]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// CHECK: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %arg0, %arg1, %ctlArg = Arg : () -> (tensor<i32>, tensor<i32>)
  // CHECK: Case
  // CHECK-SAME: branches = [#tf_type.func<@A, {}>, #tf_type.func<@A, {}>]
  %Case, %ctl = Case(%arg0, %arg1) {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [
      #tf_type.func<@A, {}>,
      #tf_type.func<@A, {}>
    ]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
}

// CHECK-NOT: tfg.func
