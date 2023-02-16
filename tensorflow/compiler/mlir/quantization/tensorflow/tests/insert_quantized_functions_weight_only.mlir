// RUN: tf-quant-opt %s -quant-insert-quantized-functions='quantization-method=weight_only target-opset=XLA' | FileCheck %s

// Empty module
module {
  func.func @simple_fn(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// CHECK-NOT: func private @internal_dequantize_f32
// CHECK-NOT: func private @internal_conv3d_fn
// CHECK-NOT: func private @internal_batch_matmul_fn
// CHECK-NOT: func private @internal_depthwise_conv2d_fn
// CHECK-NOT: func private @internal_matmul_fn
// CHECK-NOT: func private @internal_conv2d_fn

// CHECK: func private @quantized_matmul_fn
// CHECK: func private @quantized_conv2d_fn
// CHECK: func private @quantized_depthwise_conv2d_fn
// CHECK: func private @quantized_conv3d_fn
// CHECK: func private @quantized_batch_matmul_fn
// CHECK: func private @quantized_gather_fn
