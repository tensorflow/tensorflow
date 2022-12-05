// RUN: tfg-transforms-opt --tfg-functional-to-region --tfg-region-to-functional %s | FileCheck %s

// Check that roundtripping through conversion where the function is re-used
// does not change the op names.

// CHECK-LABEL: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Case
  // CHECK-SAME: @case
  %Case, %ctl = Case(%arg0, %arg1) name("case") {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// CHECK: tfg.func @case(
tfg.func @case(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: A(
  // CHECK-SAME: name("foo")
  // CHECK-SAME: {_some_attr = 5 : i32}
  %A, %ctl = A(%arg0) name("foo") {_some_attr = 5 : i32} : (tensor<i32>) -> (tensor<i32>)
  return(%A) : tensor<i32>
}

// CHECK-NOT: tfg.func
