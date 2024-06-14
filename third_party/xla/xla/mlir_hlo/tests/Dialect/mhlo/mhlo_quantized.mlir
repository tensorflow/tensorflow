// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----

// CHECK-LABEL: @uniform_quantized_c1_valid
func.func @uniform_quantized_c1_valid(%arg0: tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.1>> {
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<2x!quant.uniform<i8:f32, 0.1>>
}

// -----

func.func @uniform_quantized_c1(%arg0: tensor<2xf32>) {
  // expected-error@+1 {{Expressed type of result expected to be 'f32', but got 'f64'}}
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f64, 0.1>>
  func.return
}

// -----

func.func @uniform_quantized_c1(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1>>) {
  // expected-error@+1 {{Expressed type of result expected to be 'f32', but got 'f64'}}
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<2x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x!quant.uniform<i8:f64, 0.1>>
  func.return
}