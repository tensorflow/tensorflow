// RUN: odml-to-stablehlo-opt %s -split-input-file -mhlo-optimize | FileCheck %s

// CHECK-LABEL: testDotToDotGeneralVectorVector
func.func @testDotToDotGeneralVectorVector(%arg0: tensor<3072xf32>, %arg1: tensor<3072xf32>) -> tensor<f32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [0],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
// CHECK:      return %[[RES]] : tensor<f32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralVectorMatrix
func.func @testDotToDotGeneralVectorMatrix(%arg0: tensor<3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<512xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
  func.return %0 : tensor<512xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [0],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
// CHECK:      return %[[RES]] : tensor<512xf32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralMatrixVector
func.func @testDotToDotGeneralMatrixVector(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<2xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [1],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
// CHECK:      return %[[RES]] : tensor<2xf32>
}

// -----

// CHECK-LABEL: testDotToDotGeneralMatrixMatrix
func.func @testDotToDotGeneralMatrixMatrix(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<2x512xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
  func.return %0 : tensor<2x512xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [1],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
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

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0],
// CHECK-SAME:     rhs_batching_dimensions = [0],
// CHECK-SAME:     lhs_contracting_dimensions = [3],
// CHECK-SAME:     rhs_contracting_dimensions = [1]
// CHECK-SAME: >}> : (tensor<3x72x1x2048xf32>, tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32>
// CHECK:      return %[[RES]] : tensor<3x72x1x512xf32>
}

// -----

// CHECK-LABEL: testRemoveReshapeAroundDot
func.func @testRemoveReshapeAroundDot(%arg0: tensor<1x1x512xf32>, %arg1: tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x1x13xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x13xf32>
  %2 = "mhlo.reshape"(%1) : (tensor<1x13xf32>) -> tensor<1x1x13xf32>
  func.return %2 : tensor<1x1x13xf32>

// CHECK:      %[[RES:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<1x1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 2.850000e-03>>) -> tensor<1x1x13xf32>
// CHECK:      return %[[RES]] : tensor<1x1x13xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePads
func.func @testTwoConsecutivePads(%arg0: tensor<10x10x10xf32>) -> (tensor<12x12x12xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<0> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<11x11x11xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  return %3 : tensor<12x12x12xf32>
// CHECK:      %[[RES:.*]] = "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
// CHECK:      return %[[RES]] : tensor<12x12x12xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsNegativeLowPad
func.func @testTwoConsecutivePadsNegativeLowPad(%arg0: tensor<10x10x10xf32>) -> (tensor<10x10x10xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<0> : tensor<3xi64>, edge_padding_low = dense<-1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<9x9x9xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  return %3 : tensor<10x10x10xf32>

// CHECK:      %[[RES:.*]] = "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<-1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
// CHECK:      return %[[RES]] : tensor<10x10x10xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsTwoNegativeHighPad
func.func @testTwoConsecutivePadsTwoNegativeHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<9x9x9xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<-1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<-1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
  return %3 : tensor<9x9x9xf32>

// CHECK:      %[[RES:.*]] = "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<-2> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
// CHECK:      return %[[RES]] : tensor<9x9x9xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsPositiveNegativeHighPad
func.func @testTwoConsecutivePadsPositiveNegativeHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<11x11x11xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<-1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  return %3 : tensor<11x11x11xf32>

// CHECK:      %[[RES:.*]] = "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<0> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
// CHECK:      return %[[RES]] : tensor<11x11x11xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsNegativePositiveHighPad
func.func @testTwoConsecutivePadsNegativePositiveHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<11x11x11xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<-1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  return %3 : tensor<11x11x11xf32>

// CHECK:      "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<-1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>

// CHECK:      "mhlo.pad"(%1, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<0> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsDifferentPadVal
func.func @testTwoConsecutivePadsDifferentPadVal(%arg0: tensor<10x10x10xf32>) -> (tensor<14x14x14xf32>) {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<14x14x14xf32>
  return %3 : tensor<14x14x14xf32>

// CHECK:      "mhlo.pad"(%arg0, %1) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>

// CHECK:      "mhlo.pad"(%2, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<14x14x14xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsDifferentUsers
func.func @testTwoConsecutivePadsDifferentUsers(%arg0: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<12x12x12xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = mhlo.exponential %1 : tensor<12x12x12xf32>
  %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = "mhlo.pad"(%1, %3) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
  return %4, %2 : tensor<13x13x13xf32>, tensor<12x12x12xf32>

// CHECK:      "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>

// CHECK:      "mhlo.pad"(%1, %0) <{
// CHECK-SAME:     edge_padding_high = dense<1> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<0> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
}

// -----

// CHECK-LABEL: testTwoConsecutivePadsMultipleDownstreamUsers
  func.func @testTwoConsecutivePadsMultipleDownstreamUsers(%arg0: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<13x13x13xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.pad"(%arg0, %0) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<1> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.pad"(%1, %2) <{edge_padding_high = dense<1> : tensor<3xi64>, edge_padding_low = dense<0> : tensor<3xi64>, interior_padding = dense<0> : tensor<3xi64>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
    %4 = mhlo.exponential %3 : tensor<13x13x13xf32>
    %5 = mhlo.tanh %3 : tensor<13x13x13xf32>
    return %4, %5 : tensor<13x13x13xf32>, tensor<13x13x13xf32>

// CHECK:      "mhlo.pad"(%arg0, %0) <{
// CHECK-SAME:     edge_padding_high = dense<2> : tensor<3xi64>,
// CHECK-SAME:     edge_padding_low = dense<1> : tensor<3xi64>,
// CHECK-SAME:     interior_padding = dense<0> : tensor<3xi64>
// CHECK-SAME: }> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<13x13x13xf32>

// CHECK: mhlo.exponential %1 : tensor<13x13x13xf32>
// CHECK: mhlo.tanh %1 : tensor<13x13x13xf32>
// CHECK: return %2, %3 : tensor<13x13x13xf32>, tensor<13x13x13xf32>
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
  %r = "mhlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1x1x13xf32>, tensor<2x1x13xf32>, tensor<3x1x13xf32>) -> tensor<6x1x13xf32>
  func.return %r : tensor<6x1x13xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg1, %arg2) <{dimension = 0 : i64}> : (tensor<1x1x512xf32>, tensor<2x1x512xf32>, tensor<3x1x512xf32>) -> tensor<6x1x512xf32>
// CHECK:      %[[R1:.*]] = "mhlo.dot_general"(%[[R0]], %arg3) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [0]
// CHECK-SAME: >}> : (tensor<6x1x512xf32>, tensor<512x13xf32>) -> tensor<6x1x13xf32>
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
  %r = "mhlo.concatenate"(%0, %1) <{dimension = 3 : i64}> : (tensor<1x2x3x4x5x5x7xf32>, tensor<1x2x3x100x5x5x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
  func.return %r : tensor<1x2x3x104x5x5x7xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg1) <{dimension = 5 : i64}> : (tensor<1x9x2x3x8x4x10xf32>, tensor<1x9x2x3x8x100x10xf32>) -> tensor<1x9x2x3x8x104x10xf32>
// CHECK:      %[[R1:.*]] = "mhlo.dot_general"(%[[R0]], %arg2) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0, 2],
// CHECK-SAME:     rhs_batching_dimensions = [2, 1],
// CHECK-SAME:     lhs_contracting_dimensions = [4, 1, 6],
// CHECK-SAME:     rhs_contracting_dimensions = [6, 0, 4]
// CHECK-SAME: >}> : (tensor<1x9x2x3x8x104x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
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
  %4 = "mhlo.concatenate"(%0, %1, %2, %3) <{dimension = 0 : i64}> : (tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>) -> tensor<4x72x72xf32>
  func.return %4 : tensor<4x72x72xf32>

// CHECK:      %[[R0:.*]] = "mhlo.concatenate"(%arg0, %arg2, %arg4, %arg6) <{dimension = 0 : i64}> : (tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>) -> tensor<4x72x128xf32>
// CHECK:      %[[R1:.*]] = "mhlo.concatenate"(%arg1, %arg3, %arg5, %arg7) <{dimension = 0 : i64}> : (tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>) -> tensor<4x128x72xf32>
// CHECK:      %[[R2:.*]] = "mhlo.dot_general"(%[[R0]], %[[R1]]) <{
// CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:     lhs_batching_dimensions = [0],
// CHECK-SAME:     rhs_batching_dimensions = [0],
// CHECK-SAME:     lhs_contracting_dimensions = [2],
// CHECK-SAME:     rhs_contracting_dimensions = [1]
// CHECK-SAME:   >}> : (tensor<4x72x128xf32>, tensor<4x128x72xf32>) -> tensor<4x72x72xf32>
// CHECK:      return %[[R2]] : tensor<4x72x72xf32>
}

// -----

// CHECK-LABEL: testSliceConcat
func.func @testSliceConcat(%arg0: tensor<3x1x512xf32>) -> tensor<3x1x512xf32> {
  %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 1, 512]> : tensor<3xi64>, start_indices = dense<[0, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %1 = "mhlo.slice"(%arg0) <{limit_indices = dense<[2, 1, 512]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %2 = "mhlo.slice"(%arg0) <{limit_indices = dense<[3, 1, 512]> : tensor<3xi64>, start_indices = dense<[2, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %r = "mhlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1x1x512xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<3x1x512xf32>
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

// CHECK:      %[[R:.*]] = "mhlo.dot_general"(%arg0, %arg1) <{
// CHECK-SAME: dot_dimension_numbers = #mhlo.dot<
// CHECK-SAME:   lhs_batching_dimensions = [0],
// CHECK-SAME:   rhs_batching_dimensions = [0],
// CHECK-SAME:   lhs_contracting_dimensions = [2],
// CHECK-SAME:   rhs_contracting_dimensions = [1]
// CHECK-SAME: >}> : (tensor<1x72x72xf32>, tensor<1x72x128xf32>) -> tensor<1x72x128xf32>
// CHECK:      return %[[R]] : tensor<1x72x128xf32>
}

// -----

// CHECK-LABEL: broadcast_reshape_one_non_unit_dimnsion
func.func @broadcast_reshape_one_non_unit_dimnsion(%arg0: tensor<1x1x1x63xf32>) -> tensor<32x1x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = mhlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>
  return %1 : tensor<32x1x63xf32>
}

// CHECK: %0 = mhlo.reshape %arg0 : (tensor<1x1x1x63xf32>) -> tensor<63xf32>
// CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<2> : tensor<1xi64>}> : (tensor<63xf32>) -> tensor<32x1x63xf32>
// CHECK: return %1 : tensor<32x1x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_one_non_unit_dimnsion_trailing_zeros
func.func @broadcast_reshape_one_non_unit_dimnsion_trailing_zeros(%arg0: tensor<63x1x1x1xf32>) -> tensor<63x1x2xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
  %1 = mhlo.reshape %0 : (tensor<63x1x1x2xf32>) -> tensor<63x1x2xf32>
  return %1 : tensor<63x1x2xf32>
}

// CHECK: %0 = mhlo.reshape %arg0 : (tensor<63x1x1x1xf32>) -> tensor<63xf32>
// CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<63xf32>) -> tensor<63x1x2xf32>
// CHECK: return %1 : tensor<63x1x2xf32>

// -----

// CHECK-LABEL: broadcast_reshape_multiple_non_unit_dimension
func.func @broadcast_reshape_multiple_non_unit_dimension(%arg0: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x2x1x63xf32>) -> tensor<1x2x3x63xf32>
  %1 = mhlo.reshape %0 : (tensor<1x2x3x63xf32>) -> tensor<2x3x63xf32>
  return %1 : tensor<2x3x63xf32>
}

// CHECK: %0 = mhlo.reshape %arg0 : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}> : (tensor<2x63xf32>) -> tensor<2x3x63xf32>
// CHECK: return %1 : tensor<2x3x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_multiple_non_unit_dimension_unsorted_broadcast_dims
func.func @broadcast_reshape_multiple_non_unit_dimension_unsorted_broadcast_dims(%arg0: tensor<1x2x1x63xf32>) -> tensor<3x2x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 2, 1, 3]> : tensor<4xi64>}> : (tensor<1x2x1x63xf32>) -> tensor<3x1x2x63xf32>
  %1 = mhlo.reshape %0 : (tensor<3x1x2x63xf32>) -> tensor<3x2x63xf32>
  return %1 : tensor<3x2x63xf32>
}

// CHECK: %0 = mhlo.reshape %arg0 : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}> : (tensor<2x63xf32>) -> tensor<3x2x63xf32>
// CHECK: return %1 : tensor<3x2x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_broadcast_increases_rank
func.func @broadcast_reshape_broadcast_increases_rank(%arg0: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 4]> : tensor<4xi64>}> : (tensor<1x2x1x63xf32>) -> tensor<1x2x3x1x63xf32>
  %1 = mhlo.reshape %0 : (tensor<1x2x3x1x63xf32>) -> tensor<2x3x63xf32>
  return %1 : tensor<2x3x63xf32>
}

// CHECK: %0 = mhlo.reshape %arg0 : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK: %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}> : (tensor<2x63xf32>) -> tensor<2x3x63xf32>
// CHECK: return %1 : tensor<2x3x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_not_same_non_unit_dims
func.func @broadcast_reshape_not_same_non_unit_dims(%arg0: tensor<63x1x1x1xf32>) -> tensor<2x1x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
  %1 = mhlo.reshape %0 : (tensor<63x1x1x2xf32>) -> tensor<2x1x63xf32>
  return %1 : tensor<2x1x63xf32>
}

// CHECK: %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
// CHECK: %1 = mhlo.reshape %0 : (tensor<63x1x1x2xf32>) -> tensor<2x1x63xf32>
// CHECK: return %1 : tensor<2x1x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_multi_use
func.func @broadcast_reshape_multi_use(%arg0: tensor<1x1x1x63xf32>) -> (tensor<32x1x63xf32>, tensor<1x32x1x63xf32>) {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = mhlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>
  return %1, %0 : tensor<32x1x63xf32>, tensor<1x32x1x63xf32>
}

// CHECK: %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
// CHECK: %1 = mhlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>

// -----

// CHECK-LABEL: broadcast_reshape_rank_increase
func.func @broadcast_reshape_rank_increase(%arg0: tensor<1x1x1x63xf32>) -> tensor<32x1x1x1x1x63xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = mhlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x1x1x1x63xf32>
  return %1 : tensor<32x1x1x1x1x63xf32>
}

// CHECK: %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
// CHECK: %1 = mhlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x1x1x1x63xf32>


