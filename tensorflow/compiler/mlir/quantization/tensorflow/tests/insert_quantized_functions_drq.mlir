// RUN: tf-quant-opt %s -quant-insert-quantized-functions='quantization-method=drq target-opset=UNIFORM_QUANTIZED' | FileCheck %s

// Empty module
module {
  func.func @simple_fn(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// CHECK-NOT: func private @internal_calculate_quant_params
// CHECK-NOT: func private @internal_dequantize_i32
// CHECK-NOT: func private @internal_quantize_i8
// CHECK-NOT: func private @internal_matmul_fn
// CHECK: func private @quantized_matmul_fn
