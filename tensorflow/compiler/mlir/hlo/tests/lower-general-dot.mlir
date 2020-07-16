// RUN: mlir-hlo-opt -mhlo-test-lower-general-dot -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @testDebatch1
func @testDebatch1(%arg0: tensor<1x1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x1x3xf32> {
  // CHECK-DAG: [[R0:%.+]] = "mhlo.reshape"(%arg0) : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
  // CHECK-DAG: [[R1:%.+]] = "mhlo.dot"([[R0]], %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  // CHECK: [[R2:%.+]] = "mhlo.reshape"([[R1]]) : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<2> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<0> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>

  return %0 : tensor<1x1x3xf32>
}

// -----

// CHECK-LABEL: @testDebatch2
func @testDebatch2(%arg0: tensor<2x3xf32>, %arg1: tensor<1x1x2xf32>) -> tensor<3x1x1xf32> {
  // CHECK-DAG: [[R0:%.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-DAG: [[R1:%.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[2, 0, 1]> : tensor<3xi64>} : (tensor<1x1x2xf32>) -> tensor<2x1x1xf32>
  // CHECK-DAG: [[R2:%.+]] = "mhlo.reshape"([[R1]]) : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
  // CHECK-DAG: [[R3:%.+]] = "mhlo.dot"([[R0]], [[R2]]) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<3x2xf32>, tensor<2x1xf32>) -> tensor<3x1xf32>
  // CHECK: [[R4:%.+]] = "mhlo.reshape"([[R3]]) : (tensor<3x1xf32>) -> tensor<3x1x1xf32>

  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<0> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<2> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  return %0 : tensor<3x1x1xf32>
}

// -----

// CHECK-LABEL: @testBatchPassthrough
func @testBatchPassthrough(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x1x2xf32>) -> tensor<3x2x1xf32> {
  // CHECK-NEXT: "mhlo.dot_general"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[0]> : tensor<1xi64>, lhs_contracting_dimensions = dense<1> : tensor<1xi64>, rhs_batching_dimensions = dense<[0]> : tensor<1xi64>, rhs_contracting_dimensions = dense<2> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<2x2x3xf32>, tensor<2x1x2xf32>) -> tensor<3x2x1xf32>
  return %0 : tensor<3x2x1xf32>
}

