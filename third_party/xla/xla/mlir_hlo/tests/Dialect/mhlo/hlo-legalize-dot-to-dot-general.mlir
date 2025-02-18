// RUN: mlir-hlo-opt -mhlo-legalize-dot-to-dot-general -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @dot_to_dot_general_vector_dot_vector
func.func @dot_to_dot_general_vector_dot_vector(%arg0 : tensor<4xi64>, %arg1 : tensor<4xi64>) -> tensor<i64> {
  // CHECK: [[RES:%.+]] = "mhlo.dot_general"(%arg0, %arg1) <{
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_contracting_dimensions = [0],
  // CHECK-SAME:     rhs_contracting_dimensions = [0]
  // CHECK-SAME:   >
  // CHECK-SAME: }> : (tensor<4xi64>, tensor<4xi64>) -> tensor<i64>
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: @dot_to_dot_general_matrix_dot_vector
func.func @dot_to_dot_general_matrix_dot_vector(%arg0 : tensor<4x5xi64>, %arg1 : tensor<5xi64>) -> tensor<4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.dot_general"(%arg0, %arg1) <{
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_contracting_dimensions = [1],
  // CHECK-SAME:     rhs_contracting_dimensions = [0]
  // CHECK-SAME:   >
  // CHECK-SAME: }> : (tensor<4x5xi64>, tensor<5xi64>) -> tensor<4xi64>
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x5xi64>, tensor<5xi64>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: @dot_to_dot_general_vector_dot_matrix
func.func @dot_to_dot_general_vector_dot_matrix(%arg0 : tensor<5xi64>, %arg1 : tensor<5x4xi64>) -> tensor<4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.dot_general"(%arg0, %arg1) <{
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_contracting_dimensions = [0],
  // CHECK-SAME:     rhs_contracting_dimensions = [0]
  // CHECK-SAME:   >
  // CHECK-SAME: }> : (tensor<5xi64>, tensor<5x4xi64>) -> tensor<4xi64>
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<5xi64>, tensor<5x4xi64>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// -----

// CHECK-LABEL: @dot_to_dot_general_matrix_dot_matrix
func.func @dot_to_dot_general_matrix_dot_matrix(%arg0 : tensor<4x5xi64>, %arg1 : tensor<5x4xi64>) -> tensor<4x4xi64> {
  // CHECK: [[RES:%.+]] = "mhlo.dot_general"(%arg0, %arg1) <{
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_contracting_dimensions = [1],
  // CHECK-SAME:     rhs_contracting_dimensions = [0]
  // CHECK-SAME:   >
  // CHECK-SAME: }> : (tensor<4x5xi64>, tensor<5x4xi64>) -> tensor<4x4xi64>
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x5xi64>, tensor<5x4xi64>) -> tensor<4x4xi64>
  func.return %0 : tensor<4x4xi64>
}
