// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK: tfg.func @case
// CHECK-SAME: tensor<*x!tf_type.resource<tensor<32xf32>>>
tfg.func @case(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<i32>) {
  %A, %ctl = A : () -> (tensor<i32>)
  return(%A) : tensor<i32>
}

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %index, %arg, %ctl = Args : () -> (tensor<i32>, tensor<*x!tf_type.resource>)
  // CHECK: Case
  // CHECK-SAME: @case
  // CHECK-SAME: tensor<*x!tf_type.resource>
  %Case, %ctl_0 = Case(%index, %arg) {
    Tin = [!tf_type.resource], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<*x!tf_type.resource>) -> (tensor<i32>)
}
