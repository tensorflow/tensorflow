// RUN: odml-to-stablehlo-opt %s -split-input-file -mhlo-optimize | FileCheck %s

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

// -----

// CHECK-LABEL: testRemoveReshapeAroundDotGeneral
func.func @testRemoveReshapeAroundDotGeneral(%arg0: tensor<3x72x1x2048xf32>, %arg1: tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<3x72x1x2048xf32>) -> tensor<3x72x2048xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
    >} : (tensor<3x72x2048xf32>, tensor<3x2048x512xf32>) -> tensor<3x72x512xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<3x72x512xf32>) -> tensor<3x72x1x512xf32>
  func.return %2 : tensor<3x72x1x512xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0],
// CHECK-SAME:     rhs_batching_dimensions = [0],
// CHECK-SAME:     lhs_contracting_dimensions = [3],
// CHECK-SAME:     rhs_contracting_dimensions = [1]
// CHECK-SAME: >} : (tensor<3x72x1x2048xf32>, tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32>
// CHECK:      return %[[RES]] : tensor<3x72x1x512xf32>
}

// -----

// CHECK-LABEL: testRemoveReshapeAroundDot
func.func @testRemoveReshapeAroundDot(%arg0: tensor<1x1x512xf32>, %arg1: tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x1x13xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x13xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<1x13xf32>) -> tensor<1x1x13xf32>
  func.return %2 : tensor<1x1x13xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<1x1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 2.850000e-03>>) -> tensor<1x1x13xf32>
// CHECK:      return %[[RES]] : tensor<1x1x13xf32>
}

// -----

// CHECK-LABEL: testLiftDotConcatLHSSimple
func.func @testLiftDotConcatLHSSimple(%arg0: tensor<1x1x512xf32>, %arg1: tensor<2x1x512xf32>, %arg2: tensor<3x1x512xf32>, %arg3: tensor<512x13xf32>) -> tensor<6x1x13xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg3) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<1x1x512xf32>, tensor<512x13xf32>) -> tensor<1x1x13xf32>
  %1 = "mhlo.dot_general"(%arg1, %arg3) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<2x1x512xf32>, tensor<512x13xf32>) -> tensor<2x1x13xf32>
  %2 = "mhlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<3x1x512xf32>, tensor<512x13xf32>) -> tensor<3x1x13xf32>
  %r = "mhlo.concatenate"(%0, %1, %2) {dimension = 0 : i64} : (tensor<1x1x13xf32>, tensor<2x1x13xf32>, tensor<3x1x13xf32>) -> tensor<6x1x13xf32>
  func.return %r : tensor<6x1x13xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 0 : i64} : (tensor<1x1x512xf32>, tensor<2x1x512xf32>, tensor<3x1x512xf32>) -> tensor<6x1x512xf32>
// CHECK:      %[[R1:.*]] = "mhlo.dot_general"(%[[R0]], %arg3) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >} : (tensor<6x1x512xf32>, tensor<512x13xf32>) -> tensor<6x1x13xf32>
// CHECK:      return %[[R1]] : tensor<6x1x13xf32>
}

// -----

// CHECK-LABEL: testLiftDotConcatLHSComplex
func.func @testLiftDotConcatLHSComplex(%arg0: tensor<1x9x2x3x8x4x10xf32>, %arg1: tensor<1x9x2x3x8x100x10xf32>, %arg2: tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg2) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [4, 1, 6],
      rhs_contracting_dimensions = [6, 0, 4]
  >} : (tensor<1x9x2x3x8x4x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x4x5x5x7xf32>
  %1 = "mhlo.dot_general"(%arg1, %arg2) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [4, 1, 6],
      rhs_contracting_dimensions = [6, 0, 4]
  >} : (tensor<1x9x2x3x8x100x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x100x5x5x7xf32>
  %r = "mhlo.concatenate"(%0, %1) {dimension = 3 : i64} : (tensor<1x2x3x4x5x5x7xf32>, tensor<1x2x3x100x5x5x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
  func.return %r : tensor<1x2x3x104x5x5x7xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg1) {dimension = 5 : i64} : (tensor<1x9x2x3x8x4x10xf32>, tensor<1x9x2x3x8x100x10xf32>) -> tensor<1x9x2x3x8x104x10xf32>
// CHECK:      %[[R1:.*]] = "mhlo.dot_general"(%[[R0]], %arg2) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0, 2],
// CHECK-SAME:     rhs_batching_dimensions = [2, 1],
// CHECK-SAME:     lhs_contracting_dimensions = [4, 1, 6],
// CHECK-SAME:     rhs_contracting_dimensions = [6, 0, 4]
// CHECK-SAME: >} : (tensor<1x9x2x3x8x104x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
// CHECK:      return %[[R1]] : tensor<1x2x3x104x5x5x7xf32>
}

// -----

// CHECK-LABEL: testLiftDotConcatLHSAndRHS
func.func @testLiftDotConcatLHSAndRHS(%arg0: tensor<1x72x128xf32>, %arg1: tensor<1x128x72xf32>, %arg2: tensor<1x72x128xf32>, %arg3: tensor<1x128x72xf32>, %arg4: tensor<1x72x128xf32>, %arg5: tensor<1x128x72xf32>, %arg6: tensor<1x72x128xf32>, %arg7: tensor<1x128x72xf32>) -> tensor<4x72x72xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %1 = "mhlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %2 = "mhlo.dot_general"(%arg4, %arg5) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %3 = "mhlo.dot_general"(%arg6, %arg7) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %4 = "mhlo.concatenate"(%0, %1, %2, %3) {dimension = 0 : i64} : (tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>) -> tensor<4x72x72xf32>
  func.return %4 : tensor<4x72x72xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg2, %arg4, %arg6) {dimension = 0 : i64} : (tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>) -> tensor<4x72x128xf32>
// CHECK:      %[[R1:.*]] = "mhlo.concatenate"(%arg1, %arg3, %arg5, %arg7) {dimension = 0 : i64} : (tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>) -> tensor<4x128x72xf32>
// CHECK:      %[[R2:.*]] = "mhlo.dot_general"(%[[R0]], %[[R1]]) {
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0],
// CHECK-SAME:     rhs_batching_dimensions = [0],
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [1]
// CHECK-SAME:   >} : (tensor<4x72x128xf32>, tensor<4x128x72xf32>) -> tensor<4x72x72xf32>
// CHECK:      return %[[R2]] : tensor<4x72x72xf32>
}

// -----

// CHECK-LABEL: testSliceConcat
func.func @testSliceConcat(%arg0: tensor<3x1x512xf32>) -> tensor<3x1x512xf32> {
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 1, 512]> : tensor<3xi64>, start_indices = dense<[0, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 1, 512]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %2 = "mhlo.slice"(%arg0) {limit_indices = dense<[3, 1, 512]> : tensor<3xi64>, start_indices = dense<[2, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %r = "mhlo.concatenate"(%0, %1, %2) {dimension = 0 : i64} : (tensor<1x1x512xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<3x1x512xf32>
  func.return %r : tensor<3x1x512xf32>

// CHECK: return %arg0 : tensor<3x1x512xf32>
}

// -----

// CHECK-LABEL: testConvertReshapeDotRhsToBatchedDot
func.func @testConvertReshapeDotRhsToBatchedDot(%arg0: tensor<1x72x72xf32>, %arg1: tensor<1x72x128xf32>) -> tensor<1x72x128xf32> {
  %0 = mhlo.reshape %arg1 : (tensor<1x72x128xf32>) -> tensor<72x128xf32>
  %1 = "mhlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (tensor<1x72x72xf32>, tensor<72x128xf32>) -> tensor<1x72x128xf32>
  func.return %1 : tensor<1x72x128xf32>

// CHECK:      %[[R:.*]] = "mhlo.dot_general"(%arg0, %arg1) {
// CHECK-SAME: dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:   lhs_batching_dimensions = [0],
// CHECK-SAME:   rhs_batching_dimensions = [0],
// CHECK-SAME:   lhs_contracting_dimensions = [2],
// CHECK-SAME:   rhs_contracting_dimensions = [1]
// CHECK-SAME: >} : (tensor<1x72x72xf32>, tensor<1x72x128xf32>) -> tensor<1x72x128xf32>
// CHECK:      return %[[R]] : tensor<1x72x128xf32>
}
