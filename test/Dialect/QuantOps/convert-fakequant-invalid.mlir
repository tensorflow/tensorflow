// RUN: mlir-opt %s -split-input-file -verify-diagnostics -quant-convert-simulated-quantization

// -----
// Unsupported quantizable type (i1 is currently not a supported element type).
func @fakeQuantArgs(tensor<8x4x3xi1>) -> tensor<8x4x3xi1> {
^bb0(%arg0: tensor<8x4x3xi1>):
  // expected-error@+1 {{op operand #0 must be tensor of 32-bit float values}}
  %0 = "quant.const_fake_quant"(%arg0) {
    min = 1.1 : f32, max = 1.0 : f32, num_bits = 8
  } : (tensor<8x4x3xi1>) -> tensor<8x4x3xi1>
  return %0 : tensor<8x4x3xi1>
}
