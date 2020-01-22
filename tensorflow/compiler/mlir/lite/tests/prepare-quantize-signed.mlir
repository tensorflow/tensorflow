// RUN: tf-opt %s -tfl-prepare-quantize -tfl-test-quantize-signed | FileCheck %s

// CHECK-LABEL: uint8_to_int8
func @uint8_to_int8(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>} : (tensor<2x2xf32>)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_per_axis
func @uint8_to_int8_per_axis(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32:1, {1.0:128, 1.0}>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8:f32:1, {1.000000e+00,1.000000e+00:-128}>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%0)
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: uint8_to_int8_narrow_range
func @uint8_to_int8_narrow_range(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8<1:255>:f32, 1.0:255>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<i8<-127:127>:f32, 1.000000e+00:127>>}
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: prepareStatistics
func @prepareStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
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

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<8x4x3x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>}
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]])
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[dq1]]) {qtype = tensor<8x4x3x!quant.uniform<i8:f32:2, {0.0078431372549019607:-1,0.062745098039215685:-1,0.0039215686274509803:-1}>>}
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
// CHECK: return %[[dq2]]
}
