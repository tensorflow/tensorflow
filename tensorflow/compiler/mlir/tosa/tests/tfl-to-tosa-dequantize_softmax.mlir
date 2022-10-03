// RUN: tf-opt --tosa-dequantize-tfl-softmax %s | FileCheck %s

// -----

// CHECK-LABEL: test_softmax_qi8
// CHECK-SAME: %[[INPUT:.*]]: tensor<8x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[DEQUANTIZED_INPUT:.*]] = "tfl.dequantize"(%[[INPUT]])
// CHECK: %[[FLOAT_SOFTMAX:.*]] = "tfl.softmax"(%[[DEQUANTIZED_INPUT]])
// CHECK: %[[QUANTIZED_FLOAT_SOFTMAX:.*]] = "tfl.quantize"(%[[FLOAT_SOFTMAX]])
// CHECK: return %[[QUANTIZED_FLOAT_SOFTMAX]]
func.func @test_softmax_qi8(%arg0: tensor<8x!quant.uniform<i8:f32, 0.015>>) -> tensor<8x!quant.uniform<i8:f32, 3.9e-03:-128>> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.2 : f32}  : (tensor<8x!quant.uniform<i8:f32, 0.015>>) -> tensor<8x!quant.uniform<i8:f32, 3.9e-03:-128>>
  func.return %0 : tensor<8x!quant.uniform<i8:f32, 3.9e-03:-128>>
}

// -----
