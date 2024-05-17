// RUN: stablehlo-quant-opt %s -verify-quant-legalization -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @legal_i8
func.func @legal_i8(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  func.return %arg0: tensor<1xi8>
}

// -----

func.func @illegal_qint8(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8> {
  // expected-error@+1 {{'func.return' op is illegal as it is a UQ op or contains uq/qint types}}
  func.return %arg0: tensor<1x!tf_type.qint8>
}

// -----

func.func @illegal_cast(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xi8> {
  // expected-error@+1 {{'tf.Cast' op is illegal as it is a UQ op or contains uq/qint types}}
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  func.return %0: tensor<1xi8>
}

// -----

func.func @illegal_mhlo_uniform_quantize(%arg0: tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>> {
  // expected-error@+1 {{'mhlo.uniform_quantize' op is illegal as it is a UQ op or contains uq/qint types}}
  %0 = mhlo.uniform_quantize %arg0 : (tensor<?x?xf32>) -> tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>
}

// -----

func.func @illegal_mhlo_uniform_dequantize(%arg0: tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32> {
  // expected-error@+1 {{'mhlo.uniform_dequantize' op is illegal as it is a UQ op or contains uq/qint types}}
  %0 = mhlo.uniform_dequantize %arg0 : (tensor<?x?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_tf_uniform_quantize(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error@+1 {{'tf.UniformQuantize' op is illegal as it is a UQ op or contains uq/qint types}}
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// -----

func.func @illegal_tf_uniform_dequantize(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xf32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error@+1 {{'tf.UniformDequantize' op is illegal as it is a UQ op or contains uq/qint types}}
  %0 = "tf.UniformDequantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}
