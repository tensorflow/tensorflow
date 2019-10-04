// RUN: mlir-opt %s -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: validConstFakeQuant
func @validConstFakeQuant(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.const_fake_quant"(%arg0) {
    min = 0.0 : f32, max = 1.0 : f32, num_bits = 8, narrow_range = true
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quant.const_fake_quant"(%0) {
    min = 0.0 : f32, max = 1.0 : f32, num_bits = 8, narrow_range = false
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %2 = "quant.const_fake_quant"(%1) {
    min = 0.0 : f32, max = 1.0 : f32, num_bits = 8
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %2 : tensor<8x4x3xf32>
}

// -----
// CHECK-LABEL: validConstFakeQuantPerAxis
func @validConstFakeQuantPerAxis(%arg0: tensor<8x4x2xf32>) -> tensor<8x4x2xf32> {
  %0 = "quant.const_fake_quant_per_axis"(%arg0) {
    min = [0.0 : f32, 1.0 : f32], max = [2.0 : f32, 3.0 : f32], axis = 2, num_bits = 8, narrow_range = true
  } : (tensor<8x4x2xf32>) -> tensor<8x4x2xf32>
  %1 = "quant.const_fake_quant_per_axis"(%0) {
    min = [0.0 : f32, 1.0 : f32], max = [2.0 : f32, 3.0 : f32], axis = 2, num_bits = 8, narrow_range = false
  } : (tensor<8x4x2xf32>) -> tensor<8x4x2xf32>
  %2 = "quant.const_fake_quant_per_axis"(%1) {
    min = [0.0 : f32, 1.0 : f32], max = [2.0 : f32, 3.0 : f32], axis = 2, num_bits = 8
  } : (tensor<8x4x2xf32>) -> tensor<8x4x2xf32>
  return %2 : tensor<8x4x2xf32>
}

// -----
// CHECK-LABEL: validStatisticsRef
func @validStatisticsRef(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.stats_ref"(%arg0) { statsKey = "foobar" } :
      (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// CHECK-LABEL: validStatistics
func @validStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.stats"(%arg0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quant.stats"(%0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %1 : tensor<8x4x3xf32>
}

// -----
// CHECK-LABEL: validCoupledRef
func @validCoupledRef(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.coupled_ref"(%arg0) { coupledKey = "foobar" } :
      (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}
