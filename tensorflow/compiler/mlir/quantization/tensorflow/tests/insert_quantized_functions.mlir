// RUN: tf-quant-opt %s -quant-insert-quantized-functions | FileCheck %s

// Empty module
module {
  func.func @simple_fn(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// CHECK-NOT: func private @internal_rescale_fn
// CHECK-NOT: func private @internal_relu_fn
// CHECK-NOT: func private @internal_conv2d_fn
// CHECK-NOT: func private @internal_matmul_fn
// CHECK: func private @quantized_conv2d_with_bias_fn
// CHECK: func private @quantized_conv2d_with_bias_and_relu_fn
// CHECK: func private @quantized_conv2d_with_bias_and_relu6_fn
// CHECK: func private @quantized_conv2d_fn
// CHECK: func private @quantized_conv2d_with_relu_fn
// CHECK: func private @quantized_conv2d_with_relu6_fn
// CHECK: func private @quantized_matmul_with_bias_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu6_fn
// CHECK: func private @quantized_matmul_fn
// CHECK: func private @quantized_matmul_with_relu_fn
// CHECK: func private @quantized_matmul_with_relu6_fn
// CHECK: func private @quantize_i8
// CHECK: func private @dequantize_i8
