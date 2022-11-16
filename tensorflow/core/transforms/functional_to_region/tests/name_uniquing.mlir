// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK-LABEL: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: CaseRegion
  // CHECK-NEXT: name("foo_tfg_inlined_case_0")
  %Case, %ctl = Case(%arg0, %arg1) name("case") {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

tfg.func @case(%arg0: tensor<i32>) -> (tensor<i32>) {
  %A, %ctl = A(%arg0) name("foo") : (tensor<i32>) -> (tensor<i32>)
  return(%A) : tensor<i32>
}
