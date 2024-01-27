// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-convert-func-to-bfloat16 -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @add_f32(%arg0: tensor<3x3xbf16>, %arg1: tensor<3x3xbf16>) -> tensor<3x3xbf16>
func.func @add_f32(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK-NOT: f32
  // CHECK: stablehlo.add
  %0 = stablehlo.add %arg0, %arg1: (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: @add_f64(%arg0: tensor<3x3xbf16>, %arg1: tensor<3x3xbf16>) -> tensor<3x3xbf16>
func.func @add_f64(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // CHECK-NOT: f64
  // CHECK: stablehlo.add
  %0 = stablehlo.add %arg0, %arg1: (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xf64>
  return %0 : tensor<3x3xf64>
}

// -----

// CHECK-LABEL: @constant_f32() -> tensor<2x2xbf16>
func.func @constant_f32() -> tensor<2x2xf32> {
  // CHECK-NOT: f32
  // CHECK{LITERAL}: stablehlo.constant dense<[[1.398440e+00, 0.000000e+00], [3.093750e+00, -2.001950e-01]]> : tensor<2x2xbf16>
  %0 = stablehlo.constant dense<[[1.4, 0.0], [3.1, -0.2]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @constant_elided() -> tensor<2x2xf32> {
  // expected-error @+1 {{failed to legalize operation 'stablehlo.constant' that was explicitly marked illegal}}
  %0 = stablehlo.constant dense_resource<__elided__> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @reduce_window_f32(%arg0: tensor<2x3x1x3xbf16>) -> tensor<2x3x1x3xbf16>
func.func @reduce_window_f32(%arg0: tensor<2x3x1x3xf32>) -> tensor<2x3x1x3xf32> {
  // CHECK-NOT: f32
  // CHECK: stablehlo.reduce_window
  %0 = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
  }) {padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>} : (tensor<2x3x1x3xf32>, tensor<f32>) -> tensor<2x3x1x3xf32>
  return %1 : tensor<2x3x1x3xf32>
}

// -----

