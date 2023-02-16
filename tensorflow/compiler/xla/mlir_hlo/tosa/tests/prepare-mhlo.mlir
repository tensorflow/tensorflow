// RUN: mhlo-tosa-opt %s -split-input-file --tosa-prepare-mhlo | FileCheck %s

// CHECK-LABEL: func @dot_general_to_dot_vector_vector
func.func @dot_general_to_dot_vector_vector(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<f32> {
  // CHECK: "mhlo.dot"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dot_general_to_dot_vector_matrix
func.func @dot_general_to_dot_vector_matrix(%arg0: tensor<2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3xf32> {
  // CHECK: "mhlo.dot"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @dot_general_to_dot_matrix_vector
func.func @dot_general_to_dot_matrix_vector(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2xf32> {
  // CHECK: "mhlo.dot"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @dot_general_to_dot_matrix_matrix
func.func @dot_general_to_dot_matrix_matrix(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK: "mhlo.dot"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @dot_general_to_dot_batch_dimensions
func.func @dot_general_to_dot_batch_dimensions(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x1x2xf32>) -> tensor<2x3x1xf32> {
  // CHECK: mhlo.dot_general
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x2x3xf32>, tensor<2x1x2xf32>) -> tensor<2x3x1xf32>
  func.return %0 : tensor<2x3x1xf32>
}
