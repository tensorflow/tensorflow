// RUN: odml-to-stablehlo-opt %s -split-input-file -stablehlo-optimize | FileCheck %s

// CHECK-LABEL:   func.func @testDotToDotGeneralVectorVector(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<3072xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<3072xf32>) -> tensor<f32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], contracting_dims = [0] x [0] : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
// CHECK:           return %[[VAL_2]] : tensor<f32>
// CHECK:         }
func.func @testDotToDotGeneralVectorVector(%arg0: tensor<3072xf32>, %arg1: tensor<3072xf32>) -> tensor<f32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>

}

// -----

// CHECK-LABEL:   func.func @testDotToDotGeneralVectorMatrix(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<3072xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<3072x512xf32>) -> tensor<512xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], contracting_dims = [0] x [0] : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
// CHECK:           return %[[VAL_2]] : tensor<512xf32>
// CHECK:         }
func.func @testDotToDotGeneralVectorMatrix(%arg0: tensor<3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<512xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3072xf32>, tensor<3072x512xf32>) -> tensor<512xf32>
  func.return %0 : tensor<512xf32>

}

// -----

// CHECK-LABEL:   func.func @testDotToDotGeneralMatrixVector(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<2x3072xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<3072xf32>) -> tensor<2xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], contracting_dims = [1] x [0] : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
// CHECK:           return %[[VAL_2]] : tensor<2xf32>
// CHECK:         }
func.func @testDotToDotGeneralMatrixVector(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>

}

// -----

// CHECK-LABEL:   func.func @testDotToDotGeneralMatrixMatrix(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<2x3072xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<3072x512xf32>) -> tensor<2x512xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], contracting_dims = [1] x [0] : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x512xf32>
// CHECK:         }
func.func @testDotToDotGeneralMatrixMatrix(%arg0: tensor<2x3072xf32>, %arg1: tensor<3072x512xf32>) -> tensor<2x512xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3072xf32>, tensor<3072x512xf32>) -> tensor<2x512xf32>
  func.return %0 : tensor<2x512xf32>

}

// -----

// CHECK-LABEL:   func.func @testRemoveReshapeAroundDotGeneral(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: tensor<3x72x1x2048xf32>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], batching_dims = [0] x [0], contracting_dims = [3] x [1] : (tensor<3x72x1x2048xf32>, tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32>
// CHECK:           return %[[VAL_2]] : tensor<3x72x1x512xf32>
// CHECK:         }
func.func @testRemoveReshapeAroundDotGeneral(%arg0: tensor<3x72x1x2048xf32>, %arg1: tensor<3x2048x512xf32>) -> tensor<3x72x1x512xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<3x72x1x2048xf32>) -> tensor<3x72x2048xf32>
  %1 = "stablehlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
    >} : (tensor<3x72x2048xf32>, tensor<3x2048x512xf32>) -> tensor<3x72x512xf32>
  %2 = "stablehlo.reshape"(%1) : (tensor<3x72x512xf32>) -> tensor<3x72x1x512xf32>
  func.return %2 : tensor<3x72x1x512xf32>

}

// -----

// CHECK-LABEL:   func.func @testRemoveReshapeAroundDot(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<1x1x512xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<512x13x!quant.uniform<i8:f32, 2.850000e-03>>) -> tensor<1x1x13xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], contracting_dims = [2] x [0] : (tensor<1x1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 2.850000e-03>>) -> tensor<1x1x13xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x1x13xf32>
// CHECK:         }
func.func @testRemoveReshapeAroundDot(%arg0: tensor<1x1x512xf32>, %arg1: tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x1x13xf32> {
  %0 = "stablehlo.reshape"(%arg0) : (tensor<1x1x512xf32>) -> tensor<1x512xf32>
  %1 = "stablehlo.dot"(%0, %arg1) : (tensor<1x512xf32>, tensor<512x13x!quant.uniform<i8:f32, 0.00285>>) -> tensor<1x13xf32>
  %2 = "stablehlo.reshape"(%1) : (tensor<1x13xf32>) -> tensor<1x1x13xf32>
  func.return %2 : tensor<1x1x13xf32>

}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePads(
// CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<12x12x12xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
// CHECK:           return %[[VAL_2]] : tensor<12x12x12xf32>
// CHECK:         }
func.func @testTwoConsecutivePads(%arg0: tensor<10x10x10xf32>) -> (tensor<12x12x12xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<11x11x11xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  return %3 : tensor<12x12x12xf32>
}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsNegativeLowPad(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [-1, -1, -1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
// CHECK:           return %[[VAL_2]] : tensor<10x10x10xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsNegativeLowPad(%arg0: tensor<10x10x10xf32>) -> (tensor<10x10x10xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: -1, -1, -1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<9x9x9xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  return %3 : tensor<10x10x10xf32>

}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsTwoNegativeHighPad(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<9x9x9xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [-2, -2, -2], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
// CHECK:           return %[[VAL_2]] : tensor<9x9x9xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsTwoNegativeHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<9x9x9xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: -1, -1, -1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: -1, -1, -1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<9x9x9xf32>
  return %3 : tensor<9x9x9xf32>

}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsPositiveNegativeHighPad(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<11x11x11xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
// CHECK:           return %[[VAL_2]] : tensor<11x11x11xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsPositiveNegativeHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<11x11x11xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: -1, -1, -1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  return %3 : tensor<11x11x11xf32>

}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsNegativePositiveHighPad(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<11x11x11xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [-1, -1, -1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.pad %[[VAL_2]], %[[VAL_1]], low = [0, 0, 0], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
// CHECK:           return %[[VAL_3]] : tensor<11x11x11xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsNegativePositiveHighPad(%arg0: tensor<10x10x10xf32>) -> (tensor<11x11x11xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: -1, -1, -1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<11x11x11xf32>
  return %3 : tensor<11x11x11xf32>


}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsDifferentPadVal(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> tensor<14x14x14xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_2]], low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.pad %[[VAL_3]], %[[VAL_1]], low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<14x14x14xf32>
// CHECK:           return %[[VAL_4]] : tensor<14x14x14xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsDifferentPadVal(%arg0: tensor<10x10x10xf32>) -> (tensor<14x14x14xf32>) {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<14x14x14xf32>
  return %3 : tensor<14x14x14xf32>


}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsDifferentUsers(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<12x12x12xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.exponential %[[VAL_2]] : tensor<12x12x12xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.pad %[[VAL_2]], %[[VAL_1]], low = [0, 0, 0], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
// CHECK:           return %[[VAL_4]], %[[VAL_3]] : tensor<13x13x13xf32>, tensor<12x12x12xf32>
// CHECK:         }
func.func @testTwoConsecutivePadsDifferentUsers(%arg0: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<12x12x12xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
  %2 = stablehlo.exponential %1 : tensor<12x12x12xf32>
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = "stablehlo.pad"(%1, %3) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
  return %4, %2 : tensor<13x13x13xf32>, tensor<12x12x12xf32>


}

// -----

// CHECK-LABEL:   func.func @testTwoConsecutivePadsMultipleDownstreamUsers(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<13x13x13xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_1]], low = [1, 1, 1], high = [2, 2, 2], interior = [0, 0, 0] : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<13x13x13xf32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.exponential %[[VAL_2]] : tensor<13x13x13xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.tanh %[[VAL_2]] : tensor<13x13x13xf32>
// CHECK:           return %[[VAL_3]], %[[VAL_4]] : tensor<13x13x13xf32>, tensor<13x13x13xf32>
// CHECK:         }
  func.func @testTwoConsecutivePadsMultipleDownstreamUsers(%arg0: tensor<10x10x10xf32>) -> (tensor<13x13x13xf32>, tensor<13x13x13xf32>) {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "stablehlo.pad"(%arg0, %0) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<12x12x12xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.pad"(%1, %2) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<12x12x12xf32>, tensor<f32>) -> tensor<13x13x13xf32>
    %4 = stablehlo.exponential %3 : tensor<13x13x13xf32>
    %5 = stablehlo.tanh %3 : tensor<13x13x13xf32>
    return %4, %5 : tensor<13x13x13xf32>, tensor<13x13x13xf32>


}

// -----

// CHECK-LABEL:   func.func @testLiftDotConcatLHSSimple(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<1x1x512xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: tensor<2x1x512xf32>,
// CHECK-SAME:                                          %[[VAL_2:.*]]: tensor<3x1x512xf32>,
// CHECK-SAME:                                          %[[VAL_3:.*]]: tensor<512x13xf32>) -> tensor<6x1x13xf32> {
// CHECK:           %[[VAL_4:.*]] = stablehlo.concatenate %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], dim = 0 : (tensor<1x1x512xf32>, tensor<2x1x512xf32>, tensor<3x1x512xf32>) -> tensor<6x1x512xf32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.dot_general %[[VAL_4]], %[[VAL_3]], contracting_dims = [2] x [0] : (tensor<6x1x512xf32>, tensor<512x13xf32>) -> tensor<6x1x13xf32>
// CHECK:           return %[[VAL_5]] : tensor<6x1x13xf32>
// CHECK:         }
func.func @testLiftDotConcatLHSSimple(%arg0: tensor<1x1x512xf32>, %arg1: tensor<2x1x512xf32>, %arg2: tensor<3x1x512xf32>, %arg3: tensor<512x13xf32>) -> tensor<6x1x13xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<1x1x512xf32>, tensor<512x13xf32>) -> tensor<1x1x13xf32>
  %1 = "stablehlo.dot_general"(%arg1, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<2x1x512xf32>, tensor<512x13xf32>) -> tensor<2x1x13xf32>
  %2 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
  >} : (tensor<3x1x512xf32>, tensor<512x13xf32>) -> tensor<3x1x13xf32>
  %r = "stablehlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1x1x13xf32>, tensor<2x1x13xf32>, tensor<3x1x13xf32>) -> tensor<6x1x13xf32>
  func.return %r : tensor<6x1x13xf32>

}

// -----

// CHECK-LABEL:   func.func @testLiftDotConcatLHSComplex(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<1x9x2x3x8x4x10xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<1x9x2x3x8x100x10xf32>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32> {
// CHECK:           %[[VAL_3:.*]] = stablehlo.concatenate %[[VAL_0]], %[[VAL_1]], dim = 5 : (tensor<1x9x2x3x8x4x10xf32>, tensor<1x9x2x3x8x100x10xf32>) -> tensor<1x9x2x3x8x104x10xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.dot_general %[[VAL_3]], %[[VAL_2]], batching_dims = [0, 2] x [2, 1], contracting_dims = [4, 1, 6] x [6, 0, 4] : (tensor<1x9x2x3x8x104x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x2x3x104x5x5x7xf32>
// CHECK:         }
func.func @testLiftDotConcatLHSComplex(%arg0: tensor<1x9x2x3x8x4x10xf32>, %arg1: tensor<1x9x2x3x8x100x10xf32>, %arg2: tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x104x5x5x7xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg2) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [4, 1, 6],
      rhs_contracting_dimensions = [6, 0, 4]
  >} : (tensor<1x9x2x3x8x4x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x4x5x5x7xf32>
  %1 = "stablehlo.dot_general"(%arg1, %arg2) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 2],
      rhs_batching_dimensions = [2, 1],
      lhs_contracting_dimensions = [4, 1, 6],
      rhs_contracting_dimensions = [6, 0, 4]
  >} : (tensor<1x9x2x3x8x100x10xf32>, tensor<9x2x1x5x10x5x8x7xf32>) -> tensor<1x2x3x100x5x5x7xf32>
  %r = "stablehlo.concatenate"(%0, %1) <{dimension = 3 : i64}> : (tensor<1x2x3x4x5x5x7xf32>, tensor<1x2x3x100x5x5x7xf32>) -> tensor<1x2x3x104x5x5x7xf32>
  func.return %r : tensor<1x2x3x104x5x5x7xf32>

}

// -----

// CHECK-LABEL: testLiftDotConcatLHSAndRHS
// CHECK:           %[[VAL_8:.*]] = stablehlo.concatenate %arg0, %arg2, %arg4, %arg6, dim = 0 : (tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>, tensor<1x72x128xf32>) -> tensor<4x72x128xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.concatenate %arg1, %arg3, %arg5, %arg7, dim = 0 : (tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>, tensor<1x128x72xf32>) -> tensor<4x128x72xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.dot_general %[[VAL_8]], %[[VAL_9]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x72x128xf32>, tensor<4x128x72xf32>) -> tensor<4x72x72xf32>
// CHECK:           return %[[VAL_10]] : tensor<4x72x72xf32>
// CHECK:         }
func.func @testLiftDotConcatLHSAndRHS(%arg0: tensor<1x72x128xf32>, %arg1: tensor<1x128x72xf32>, %arg2: tensor<1x72x128xf32>, %arg3: tensor<1x128x72xf32>, %arg4: tensor<1x72x128xf32>, %arg5: tensor<1x128x72xf32>, %arg6: tensor<1x72x128xf32>, %arg7: tensor<1x128x72xf32>) -> tensor<4x72x72xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %1 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %2 = "stablehlo.dot_general"(%arg4, %arg5) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %3 = "stablehlo.dot_general"(%arg6, %arg7) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >} : (tensor<1x72x128xf32>, tensor<1x128x72xf32>) -> tensor<1x72x72xf32>
  %4 = "stablehlo.concatenate"(%0, %1, %2, %3) <{dimension = 0 : i64}> : (tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>, tensor<1x72x72xf32>) -> tensor<4x72x72xf32>
  func.return %4 : tensor<4x72x72xf32>

}

// -----

// CHECK-LABEL:   func.func @testSliceConcat(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<3x1x512xf32>) -> tensor<3x1x512xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.slice %[[VAL_0]] [0:3, 0:1, 0:512] : (tensor<3x1x512xf32>) -> tensor<3x1x512xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.concatenate %[[VAL_1]], dim = 0 : (tensor<3x1x512xf32>) -> tensor<3x1x512xf32>
// CHECK:           return %[[VAL_2]] : tensor<3x1x512xf32>
// CHECK:         }
func.func @testSliceConcat(%arg0: tensor<3x1x512xf32>) -> tensor<3x1x512xf32> {
  %0 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 1, 1, 512>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %1 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 2, 1, 512>, start_indices = array<i64: 1, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %2 = "stablehlo.slice"(%arg0) <{limit_indices = array<i64: 3, 1, 512>, start_indices = array<i64: 2, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<3x1x512xf32>) -> tensor<1x1x512xf32>
  %r = "stablehlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1x1x512xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<3x1x512xf32>
  func.return %r : tensor<3x1x512xf32>

}

// -----

// CHECK-LABEL:   func.func @testConvertReshapeDotRhsToBatchedDot(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<1x72x72xf32>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: tensor<1x72x128xf32>) -> tensor<1x72x128xf32> {
// CHECK:           %[[VAL_2:.*]] = stablehlo.dot_general %[[VAL_0]], %[[VAL_1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x72x72xf32>, tensor<1x72x128xf32>) -> tensor<1x72x128xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x72x128xf32>
// CHECK:         }
func.func @testConvertReshapeDotRhsToBatchedDot(%arg0: tensor<1x72x72xf32>, %arg1: tensor<1x72x128xf32>) -> tensor<1x72x128xf32> {
  %0 = stablehlo.reshape %arg1 : (tensor<1x72x128xf32>) -> tensor<72x128xf32>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >} : (tensor<1x72x72xf32>, tensor<72x128xf32>) -> tensor<1x72x128xf32>
  func.return %1 : tensor<1x72x128xf32>

}

// -----

// CHECK-LABEL:   func.func @broadcast_reshape_one_non_unit_dimnsion(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: tensor<1x1x1x63xf32>) -> tensor<32x1x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<1x1x1x63xf32>) -> tensor<63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [2] : (tensor<63xf32>) -> tensor<32x1x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x1x63xf32>
// CHECK:         }
func.func @broadcast_reshape_one_non_unit_dimnsion(%arg0: tensor<1x1x1x63xf32>) -> tensor<32x1x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>
  return %1 : tensor<32x1x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_one_non_unit_dimnsion_trailing_zeros(
// CHECK-SAME:                                                                      %[[VAL_0:.*]]: tensor<63x1x1x1xf32>) -> tensor<63x1x2xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<63x1x1x1xf32>) -> tensor<63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [0] : (tensor<63xf32>) -> tensor<63x1x2xf32>
// CHECK:           return %[[VAL_2]] : tensor<63x1x2xf32>
// CHECK:         }
func.func @broadcast_reshape_one_non_unit_dimnsion_trailing_zeros(%arg0: tensor<63x1x1x1xf32>) -> tensor<63x1x2xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
  %1 = stablehlo.reshape %0 : (tensor<63x1x1x2xf32>) -> tensor<63x1x2xf32>
  return %1 : tensor<63x1x2xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_multiple_non_unit_dimension(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [0, 2] : (tensor<2x63xf32>) -> tensor<2x3x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x3x63xf32>
// CHECK:         }
func.func @broadcast_reshape_multiple_non_unit_dimension(%arg0: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x2x1x63xf32>) -> tensor<1x2x3x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<1x2x3x63xf32>) -> tensor<2x3x63xf32>
  return %1 : tensor<2x3x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_multiple_non_unit_dimension_unsorted_broadcast_dims(
// CHECK-SAME:                                                                                     %[[VAL_0:.*]]: tensor<1x2x1x63xf32>) -> tensor<3x2x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [1, 2] : (tensor<2x63xf32>) -> tensor<3x2x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<3x2x63xf32>
// CHECK:         }
func.func @broadcast_reshape_multiple_non_unit_dimension_unsorted_broadcast_dims(%arg0: tensor<1x2x1x63xf32>) -> tensor<3x2x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 2, 1, 3>}> : (tensor<1x2x1x63xf32>) -> tensor<3x1x2x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<3x1x2x63xf32>) -> tensor<3x2x63xf32>
  return %1 : tensor<3x2x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_broadcast_increases_rank(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.reshape %[[VAL_0]] : (tensor<1x2x1x63xf32>) -> tensor<2x63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.broadcast_in_dim %[[VAL_1]], dims = [0, 2] : (tensor<2x63xf32>) -> tensor<2x3x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x3x63xf32>
// CHECK:         }
func.func @broadcast_reshape_broadcast_increases_rank(%arg0: tensor<1x2x1x63xf32>) -> tensor<2x3x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 4>}> : (tensor<1x2x1x63xf32>) -> tensor<1x2x3x1x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<1x2x3x1x63xf32>) -> tensor<2x3x63xf32>
  return %1 : tensor<2x3x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_not_same_non_unit_dims(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: tensor<63x1x1x1xf32>) -> tensor<2x1x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.broadcast_in_dim %[[VAL_0]], dims = [0, 1, 2, 3] : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<63x1x1x2xf32>) -> tensor<2x1x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x1x63xf32>
// CHECK:         }
func.func @broadcast_reshape_not_same_non_unit_dims(%arg0: tensor<63x1x1x1xf32>) -> tensor<2x1x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<63x1x1x1xf32>) -> tensor<63x1x1x2xf32>
  %1 = stablehlo.reshape %0 : (tensor<63x1x1x2xf32>) -> tensor<2x1x63xf32>
  return %1 : tensor<2x1x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_multi_use(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<1x1x1x63xf32>) -> (tensor<32x1x63xf32>, tensor<1x32x1x63xf32>) {
// CHECK:           %[[VAL_1:.*]] = stablehlo.broadcast_in_dim %[[VAL_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>
// CHECK:           return %[[VAL_2]], %[[VAL_1]] : tensor<32x1x63xf32>, tensor<1x32x1x63xf32>
// CHECK:         }
func.func @broadcast_reshape_multi_use(%arg0: tensor<1x1x1x63xf32>) -> (tensor<32x1x63xf32>, tensor<1x32x1x63xf32>) {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x63xf32>
  return %1, %0 : tensor<32x1x63xf32>, tensor<1x32x1x63xf32>
}


// -----

// CHECK-LABEL:   func.func @broadcast_reshape_rank_increase(
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<1x1x1x63xf32>) -> tensor<32x1x1x1x1x63xf32> {
// CHECK:           %[[VAL_1:.*]] = stablehlo.broadcast_in_dim %[[VAL_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<1x32x1x63xf32>) -> tensor<32x1x1x1x1x63xf32>
// CHECK:           return %[[VAL_2]] : tensor<32x1x1x1x1x63xf32>
// CHECK:         }
func.func @broadcast_reshape_rank_increase(%arg0: tensor<1x1x1x63xf32>) -> tensor<32x1x1x1x1x63xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x63xf32>) -> tensor<1x32x1x63xf32>
  %1 = stablehlo.reshape %0 : (tensor<1x32x1x63xf32>) -> tensor<32x1x1x1x1x63xf32>
  return %1 : tensor<32x1x1x1x1x63xf32>
}
