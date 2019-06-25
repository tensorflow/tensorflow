// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
func @invalidStatisticsMismatchedLayerType(%arg0: tensor<8x4x3xf32>) ->
    tensor<8x4x3xf32> {
  // expected-error@+1 {{layerStats must have a floating point element type}}
  %0 = "quant.stats"(%arg0) {
    layerStats: dense<[-1, 1]> : tensor<2xi8>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
func @invalidStatisticsMismatchedLayerRank(%arg0: tensor<8x4x3xf32>) ->
    tensor<8x4x3xf32> {
  // expected-error@+1 {{layerStats must have shape [2]}}
  %0 = "quant.stats"(%arg0) {
    layerStats: dense<[[-1.0, 1.0]]> : tensor<1x2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
func @invalidStatisticsMismatchedLayerShape(%arg0: tensor<8x4x3xf32>) ->
    tensor<8x4x3xf32> {
  // expected-error@+1 {{layerStats must have shape [2]}}
  %0 = "quant.stats"(%arg0) {
    layerStats: dense<[-1.0, 1.0, 2.0]> : tensor<3xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// CHECK-LABEL: validStatistics
func @invalidStatisticsMismatchedAxisType(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  // expected-error@+1 {{axisStats must have a floating point element type}}
  %0 = "quant.stats"(%0) {
    layerStats: dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats: dense<[
      [-1, 1],
      [-8, 8],
      [-1, 0]
    ]> : tensor<3x2xi8>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
func @invalidStatisticsMismatchedAxisRank(%arg0: tensor<8x4x3xf32>) ->
    tensor<8x4x3xf32> {
  // expected-error@+1 {{axisStats must have shape [N,2] where N = the argument rank}}
  %0 = "quant.stats"(%arg0) {
    layerStats: dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats: dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5],
      [-2.0, 3.5]
    ]> : tensor<4x2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
func @invalidStatisticsMismatchedAxisShape(%arg0: tensor<8x4x3xf32>) ->
    tensor<8x4x3xf32> {
  // expected-error@+1 {{axisStats must have shape [N,2] where N = the argument rank}}
  %0 = "quant.stats"(%arg0) {
    layerStats: dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats: dense<[
      [-1.0, 1.0, 1.0],
      [-8.0, 8.0, 1.0],
      [-0.5, 0.5, 1.0]
    ]> : tensor<3x3xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}
