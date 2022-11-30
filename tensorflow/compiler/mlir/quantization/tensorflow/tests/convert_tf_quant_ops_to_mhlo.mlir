// RUN: tf-quant-opt %s -quant-convert-tf-quant-ops-to-mhlo | FileCheck %s

func.func @quantized_matmul_fn(%input: tensor<*xf32>) -> tensor<*xf32> {
  %weight = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2032207D207D2074656E736F725F636F6E74656E743A20225C3030315C3030325C3030335C30303422"> : tensor<2x2x!tf_type.qint8> } : () -> tensor<2x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  %0 = "tf.AddV2"(%input, %input) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %1 = "tf.UniformQuantizedDotHybrid"(%0, %weight, %weight_scales, %weight_zps) {rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64} : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// CHECK: func @quantized_matmul_fn
// CHECK: mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
// CHECK-NEXT: "tf.AddV2"
// CHECK-NEXT: "mhlo.dot"(%1, %0) : (tensor<*xf32>, tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<*xf32>
