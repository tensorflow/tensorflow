// RUN: tfg-transforms-opt %s \
// RUN: --tfg-functional-to-region --tfg-region-to-functional \
// RUN: --tfg-functional-to-region --tfg-region-to-functional \
// RUN: | FileCheck %s

// Check that function names remain consistent when passed through region
// conversion multiple times. In this case, the first pass will specialize the
// same branch function twice and create two new functions.

// The function is specialized twice, but it must retain unique names.
// CHECK: tfg.func @case
tfg.func @case(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  return(%arg1) : tensor<i32>
}

// CHECK: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: Case
  // CHECK-SAME: name("foo")
  // CHECK-SAME: @case_tfg_region_specialized_foo_0
  %Case0, %ctl0 = Case(%arg0, %arg1, %arg2) name("foo") {
    Tin = [i32, i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: Case
  // CHECK-SAME: name("bar")
  // CHECK-SAME: @case_tfg_region_specialized_bar_0
  %Case1, %ctl1 = Case(%arg0, %arg1, %arg2) name("bar") {
    Tin = [i32, i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case0, %Case1) : tensor<i32>, tensor<i32>
}

// CHECK: tfg.func @case_tfg_region_specialized_foo_0
// CHECK: tfg.func @case_tfg_region_specialized_bar_0
// CHECK-NOT: tfg.func
