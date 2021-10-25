// RUN: mlir-hlo-opt -mhlo-legalize-einsum-to-dot-general %s -o - | FileCheck %s

func @einsum_diag(%arg0: tensor<6x6xf32>) -> tensor<6xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "mhlo.einsum"(%0, %arg0) {einsum_config = ",ii->i"} : (tensor<f32>, tensor<6x6xf32>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
}
// CHECK-LABEL: func @einsum_diag
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         %[[CST:.+]] = mhlo.constant dense<{{.*}} : tensor<f32>
// CHECK:         %{{.+}} = "mhlo.dot_general"(%[[CST]], %[[ARG0]])
// CHECK-SAME:          dot_dimension_numbers = #mhlo.dot<>
// CHECK-SAME:    : (tensor<f32>, tensor<6x6xf32>) -> tensor<6xf32>

func @einsum_batched_matrix_high_rank_vector_mul(%arg0: tensor<8x2x6xf32>, %arg1: tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "bxy,bijy->bijx"} : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32>
  return %0 : tensor<8x5x3x2xf32>
}
// CHECK-LABEL: func @einsum_batched_matrix_high_rank_vector_mul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:      dot_dimension_numbers =
// CHECK-SAME:        lhs_batching_dimensions = [0]
// CHECK-SAME:        rhs_batching_dimensions = [0]
// CHECK-SAME:        lhs_contracting_dimensions = [2]
// CHECK-SAME:        rhs_contracting_dimensions = [3]
// CHECK-SAME:    : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32>

func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ij,jk->ik"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:      dot_dimension_numbers =
// CHECK-SAME:        lhs_contracting_dimensions = [1]
// CHECK-SAME:        rhs_contracting_dimensions = [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

func @matvec(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ij,j->i"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @matvec
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:      dot_dimension_numbers =
// CHECK-SAME:        lhs_contracting_dimensions = [1]
// CHECK-SAME:        rhs_contracting_dimensions = [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>

func @dot(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "i,i->"} : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL: func @dot
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %{{.+}} = "mhlo.dot_general"(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:      dot_dimension_numbers =
// CHECK-SAME:        lhs_contracting_dimensions = [0]
// CHECK-SAME:        rhs_contracting_dimensions = [0]
// CHECK-SAME:    : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>

