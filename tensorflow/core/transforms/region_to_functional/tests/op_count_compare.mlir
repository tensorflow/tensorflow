// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that function re-use will check the number of operations in the region.

tfg.func @then(%arg0: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>) {
  return(%arg0) : tensor<f32>
}

tfg.func @else(%arg0: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>) {
  return(%arg2) : tensor<f32>
}

tfg.func @test(%cond: tensor<i1>, %lhs: tensor<f32>, %rhs: tensor<f32>) -> (tensor<f32>) {
  // CHECK: If
  // CHECK-SAME: else_branch = #tf_type.func<@else_0, {}>
  %If, %ctl = IfRegion %cond then {
    yield(%lhs) : tensor<f32>
  } else {
    %A, %ctlA = A(%rhs) : (tensor<f32>) -> (tensor<f32>)
    yield(%A) : tensor<f32>
  } {then_region_attrs = #tfg.region_attrs<{sym_name = "then"} [] [{}]>,
     else_region_attrs = #tfg.region_attrs<{sym_name = "else"} [] [{}]>} : (tensor<i1>) -> (tensor<f32>)
  return(%If) : tensor<f32>
}

// CHECK: tfg.func @else_0
