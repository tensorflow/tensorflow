// RUN: tf-quant-opt %s -tf-quant-insert-quantized-functions | FileCheck %s
// RUN: tf-quant-opt %s -tf-quant-insert-quantized-functions='quantization-method=ptq target-opset=UNIFORM_QUANTIZED' --mlir-print-ir-after-all | FileCheck --check-prefix=UQ-CHECK %s

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
// CHECK-SAME: tf_quant.quantized_ops = ["Conv2D", "BiasAdd"]
// CHECK: func private @quantized_conv2d_with_bias_and_relu_fn
// CHECK: func private @quantized_conv2d_with_bias_and_relu6_fn
// CHECK: func private @quantized_conv2d_fn
// CHECK: func private @quantized_conv2d_with_relu_fn
// CHECK: func private @quantized_conv2d_with_relu6_fn
// CHECK: func private @quantized_depthwise_conv2d_with_bias_and_relu_float_output_fn
// CHECK-SAME: tf_quant.quantized_ops = ["DepthwiseConv2D", "BiasAdd", "Relu"]
// CHECK: func private @quantized_matmul_with_bias_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu_fn
// CHECK: func private @quantized_matmul_with_bias_and_relu6_fn
// CHECK: func private @quantized_matmul_fn
// CHECK-SAME: tf_quant.quantized_ops = ["MatMul"]
// CHECK: func private @quantized_matmul_with_relu_fn
// CHECK: func private @quantized_matmul_with_relu6_fn
// CHECK: func private @quantized_conv3d_with_bias_fn
// CHECK-SAME: tf_quant.quantized_ops = ["Conv3D", "BiasAdd"]
// CHECK: func private @quantized_batch_matmul_with_bias_fn
// CHECK-SAME: tf_quant.quantized_ops = ["BatchMatMul", "BiasAdd"]
// CHECK: func private @quantize_i8
// CHECK: func private @dequantize_i8

// UQ-CHECK-NOT: func private @internal_conv2d_fn
// UQ-CHECK-NOT: func private @internal_requantize_qi8_fn
// UQ-CHECK-NOT: func private @internal_requantize_no_activation_fn
// UQ-CHECK-NOT: func private @internal_requantize_and_relu_fn
// UQ-CHECK: func private @quantized_conv2d_with_bias_fn
// UQ-CHECK-SAME: tf_quant.quantized_ops = ["Conv2D", "BiasAdd"]
// UQ-CHECK: func private @quantized_conv2d_with_bias_and_relu_fn
// UQ-CHECK: func private @quantized_conv2d_with_bias_and_relu6_fn
// UQ-CHECK: func private @quantized_conv2d_with_relu_fn
// UQ-CHECK: func private @quantized_conv2d_with_relu6_fn
// UQ-CHECK: func private @quantized_depthwise_conv2d_with_bias_fn
// UQ-CHECK-SAME: tf_quant.quantized_ops = ["DepthwiseConv2D", "BiasAdd"]
// UQ-CHECK: func private @quantized_depthwise_conv2d_with_bias_and_relu_fn
// UQ-CHECK: func private @quantized_depthwise_conv2d_with_bias_and_relu6_fn
// UQ-CHECK: func private @quantized_depthwise_conv2d_with_relu_fn
// UQ-CHECK: func private @quantized_depthwise_conv2d_with_relu6_fn
// UQ-CHECK: func private @quantized_matmul_with_bias_fn
// UQ-CHECK-SAME: tf_quant.quantized_ops = ["MatMul", "BiasAdd"]
// UQ-CHECK: func private @quantized_matmul_with_bias_and_relu_fn
// UQ-CHECK: func private @quantized_matmul_with_bias_and_relu6_fn
// UQ-CHECK: func private @quantized_matmul_with_relu_fn
// UQ-CHECK: func private @quantized_matmul_with_relu6_fn
// UQ-CHECK: func private @quantize_i8
// UQ-CHECK: func private @quantize_i32
// UQ-CHECK: func private @dequantize_i8
