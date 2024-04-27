// RUN: stablehlo-quant-opt %s -quant-convert-tf-quant-ops-to-mhlo | FileCheck %s

// CHECK-LABEL: func @quantized_matmul_fn
func.func @quantized_matmul_fn(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %weight = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2032207D207D2074656E736F725F636F6E74656E743A20225C3030315C3030325C3030335C30303422"> : tensor<2x2x!tf_type.qint8> } : () -> tensor<2x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: "tf.AddV2"
  // CHECK: mhlo.constant
  // CHECK-SAME{LITERAL}: dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
  %0 = "tf.AddV2"(%input, %input) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: "mhlo.dot"
  // CHECK-SAME: (tensor<?x?xf32>, tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  %1 = "tf.UniformQuantizedDotHybrid"(%0, %weight, %weight_scales, %weight_zps) {rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64} : (tensor<?x?xf32>, tensor<2x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_add
func.func @uniform_quantized_add(%input: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  // tensor_proto that points to dense<127> of type !tf_type.qint32.
  // CHECK-DAG: %[[RHS:.*]] = mhlo.constant() <{value = dense<127> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  %bias = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>
  %bias_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %bias_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  %output_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[LHS:.*]] = mhlo.uniform_quantize %arg0 : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  // CHECK-DAG: %[[LHS1:.*]] = mhlo.bitcast_convert %[[LHS]] : (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>) -> tensor<3x2xi32>
  // CHECK-DAG: %[[LHS2:.*]] = mhlo.bitcast_convert %[[LHS1]] : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  %0 = "tf.UniformQuantize"(%input, %input_scales, %input_zps) {
    quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
  } : (tensor<3x2xf32>, tensor<f32>, tensor<i32>) -> tensor<3x2x!tf_type.qint32>

  // CHECK: chlo.broadcast_add %[[LHS2]], %[[RHS]] {broadcast_dimensions = array<i64: 1>} :
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>, tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>)
  // CHECK-SAME: -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  %1 = "tf.UniformQuantizedAdd"(
    %0, %bias,
    %input_scales, %input_zps,
    %bias_scales, %bias_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -2147483648 : i64,
      lhs_quantization_max_val = 2147483647 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -2147483648 : i64,
      rhs_quantization_max_val = 2147483647 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<3x2x!tf_type.qint32>
  %2 = "tf.UniformDequantize"(%1, %output_scales, %output_zps) {
    quantization_axis = -1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
  } : (tensor<3x2x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<3x2xf32>
  func.return %2 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_add_bias_not_const
func.func @uniform_quantized_add_bias_not_const(%input1: tensor<3x2xi32>, %input2: tensor<2xi32>) -> tensor<3x2xi32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  %bias_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %bias_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  %output_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[LHS_1:.*]] = mhlo.convert %arg0 : tensor<3x2xi32>
  // CHECK-DAG: %[[LHS_2:.*]] = mhlo.bitcast_convert %[[LHS_1]] : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  // CHECK-DAG: %[[RHS_1:.*]] = mhlo.convert %arg1 : tensor<2xi32>
  // CHECK-DAG: %[[RHS_2:.*]] = mhlo.bitcast_convert %[[RHS_1]] : (tensor<2xi32>) -> tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  %input1_qint = "tf.Cast"(%input1) {Truncate = false} : (tensor<3x2xi32>) -> tensor<3x2x!tf_type.qint32>
  %input2_qint = "tf.Cast"(%input2) {Truncate = false} : (tensor<2xi32>) -> tensor<2x!tf_type.qint32>

  // CHECK: %[[RES:.*]] = chlo.broadcast_add %[[LHS_2]], %[[RHS_2]] {broadcast_dimensions = array<i64: 1>} :
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>, tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>)
  // CHECK-SAME: -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  %result = "tf.UniformQuantizedAdd"(
    %input1_qint, %input2_qint,
    %input_scales, %input_zps,
    %bias_scales, %bias_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -2147483648 : i64,
      lhs_quantization_max_val = 2147483647 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -2147483648 : i64,
      rhs_quantization_max_val = 2147483647 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<3x2x!tf_type.qint32>

  // CHECK: %[[RES_INT_1:.*]] = mhlo.bitcast_convert %[[RES]] : (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>) -> tensor<3x2xi32>
  // CHECK: %[[RES_INT_2:.*]] = mhlo.convert %[[RES_INT_1]] : tensor<3x2xi32>
  %result_int = "tf.Cast"(%result) {Truncate = false} : (tensor<3x2x!tf_type.qint32>) -> tensor<3x2xi32>
  // CHECK: return %[[RES_INT_2]] : tensor<3x2xi32>
  func.return %result_int : tensor<3x2xi32>
}
