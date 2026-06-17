// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @branch
tfg.func @branch(%arg0: tensor<i32>) -> (tensor<i32>) {
  return(%arg0) : tensor<i32>
}

// CHECK-LABEL: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: CaseRegion
  %Case, %ctl = Case(%arg0, %arg1) {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@branch, {}>]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// CHECK-LABEL: tfg.func @use
tfg.func @use(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Foo
  // CHECK-SAME: @branch
  %ctl = Foo {_use = #tf_type.func<@branch, {}>}
  return(%arg0) [%ctl] : tensor<i32>
}
