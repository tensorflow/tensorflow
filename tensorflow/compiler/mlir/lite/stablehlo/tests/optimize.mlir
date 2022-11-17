// RUN: tf-mhlo-tfl-opt %s -split-input-file -mhlo-optimize | FileCheck %s

// CHECK-LABEL: testDotToDotGeneralVectorVector
func.func @testDotToDotGeneralVectorVector(%arg0: tensor<3072xf32>, %arg1: tensor<3072xf32>) -> tensor<f32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [0],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
// CHECK:      return %[[RES]] : tensor<f32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralVectorMatrix
func.func @testDotToDotGeneralVectorMatrix(%arg0: tensor<3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<512xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
  func.return %0 : tensor<512xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [0],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
// CHECK:      return %[[RES]] : tensor<512xf32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralMatrixVector
func.func @testDotToDotGeneralMatrixVector(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<2xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [1],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
// CHECK:      return %[[RES]] : tensor<2xf32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralMatrixMatrix
func.func @testDotToDotGeneralMatrixMatrix(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<2x512xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
  func.return %0 : tensor<2x512xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [1],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
// CHECK:      return %[[RES]] : tensor<2x512xf32>
}
