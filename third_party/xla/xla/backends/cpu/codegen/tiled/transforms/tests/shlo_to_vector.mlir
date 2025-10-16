// RUN: emitters_opt %s --xtile-cpu-shlo-to-vector -split-input-file | FileCheck %s

func.func @transpose(%input : tensor<1024x32xf32>) -> tensor<32x1024xf32> {
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<1024x32xf32> to vector<32x1024xf32>
  %transposed = stablehlo.transpose %input, dims = [1, 0] : (tensor<1024x32xf32>) -> tensor<32x1024xf32>
  return %transposed : tensor<32x1024xf32>
}
// -----

// CHECK-DAG: #[[LHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[RHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @dot_general(%lhs : tensor<1024x32xf32>, %rhs : tensor<32x1024xf32>) -> tensor<1024x1024xf32> {
  // CHECK: %[[ACCUMULATOR:.*]] = arith.constant dense<0.000000e+00> : vector<1024x1024xf32>
  // CHECK: vector.contract
  // CHECK-SAME: {indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]],
  // CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"],
  // CHECK-SAME: kind = #vector.kind<add>}
  // CHECK-SAME: %[[ACCUMULATOR]] : vector<1024x32xf32>, vector<32x1024xf32> into vector<1024x1024xf32>
  %result = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<1024x32xf32>, tensor<32x1024xf32>) -> tensor<1024x1024xf32>
  return %result : tensor<1024x1024xf32>
}

// -----

// CHECK-DAG: #[[INPUT_MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[OUTPUT_MAP:.*]] = affine_map<(d0) -> ()>
func.func @dot_scalar_output(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<f32> {
  // CHECK: %[[ACCUMULATOR:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[RESULT:.*]] = vector.contract
  // CHECK-SAME: {indexing_maps = [#[[INPUT_MAP]], #[[INPUT_MAP]], #[[OUTPUT_MAP]]],
  // CHECK-SAME: iterator_types = ["reduction"],
  // CHECK-SAME: kind = #vector.kind<add>}
  // CHECK-SAME: %[[ACCUMULATOR]] : vector<1024xf32>, vector<1024xf32> into f32
  // CHECK: %[[RESULT_TENSOR:.*]] = tensor.from_elements %[[RESULT]] : tensor<f32>
  %result = stablehlo.dot_general %lhs, %rhs, contracting_dims = [0] x [0] : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>
  // CHECK: return %[[RESULT_TENSOR]] : tensor<f32>
  return %result : tensor<f32>
}
