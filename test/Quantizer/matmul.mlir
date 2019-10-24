// RUN: mlir-opt %s -quantizer-infer-quantized-types -quant-convert-const -quantizer-remove-instrumentation -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

// ----
// A matmul without fused clamp or bias.
// CHECK-LABEL: @matmul
// CHECK: %cst = constant dense{{.*}}tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>
// CHECK-NEXT: %2 = "fxpmath.real_matmul"(%0, %1) : (tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>, tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>
// CHECK-NEXT: %3 = "quant.dcast"(%2) : (tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>) -> tensor<300x5xf32>
func @matmul(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats = dense<[-6.123e+00, 3.45e+00]> : tensor<2xf32>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name = "constant.35"} dense<[[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]> : tensor<3x5xf32>
  %1 = "fxpmath.real_matmul"(%0, %cst) : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  %2 = "quant.stats"(%1) {layerStats = dense<[-8.000000e+00, 8.000000e+00]> : tensor<2xf32>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
  return %2 : tensor<300x5xf32>
}

// ----
// A matmul with fused clamp which serves as statistics for the result.
// CHECK-LABEL: @matmul_clamp
// CHECK: %cst = constant dense{{.*}}tensor<3x5xi8>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>
// CHECK-NEXT: %2 = "fxpmath.real_matmul"(%0, %1) {clamp_max = 6.100000e+00 : f64, clamp_min = -1.225000e+01 : f64} : (tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>, tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.072058823529412216:42>>
// CHECK-NEXT: %3 = "quant.dcast"(%2) : (tensor<300x5x!quant.uniform<i8:f32, 0.072058823529412216:42>>) -> tensor<300x5xf32>
func @matmul_clamp(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats = dense<[-6.123e+00, 3.45e+00]> : tensor<2xf32>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name = "constant.35"} dense<[[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]> : tensor<3x5xf32>
  %1 = "fxpmath.real_matmul"(%0, %cst) {clamp_max = 6.10, clamp_min = -12.25} : (tensor<300x3xf32>, tensor<3x5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

// ----
// A matmul with bias and clamp.
// CHECK-LABEL: @matmul_add_clamp
// CHECK: %cst = constant dense{{.*}}tensor<3x5xi8>
// CHECK-NEXT: %cst_0 = constant dense<[14, 28, 42, 56, 69]> : tensor<5xi32>
// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>
// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>
// CHECK-NEXT: %2 = "quant.scast"(%cst_0) : (tensor<5xi32>) -> tensor<5x!quant.uniform<i32:f32, 0.072058823529412216>>
// CHECK-NEXT: %3 = "fxpmath.real_matmul_bias"(%0, %1, %2) {clamp_max = 6.100000e+00 : f64, clamp_min = -1.225000e+01 : f64} : (tensor<300x3x!quant.uniform<i8:f32, 0.037564418067230126:35>>, tensor<3x5x!quant.uniform<i8:f32, 0.0062823070315864236:-1>>, tensor<5x!quant.uniform<i32:f32, 0.072058823529412216>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.072058823529412216:42>>
// CHECK-NEXT: %4 = "quant.dcast"(%3) : (tensor<300x5x!quant.uniform<i8:f32, 0.072058823529412216:42>>) -> tensor<300x5xf32>
func @matmul_add_clamp(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
  %0 = "quant.stats"(%arg0) {layerStats = dense<[-6.123e+00, 3.45e+00]> : tensor<2xf32>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
  %cst = constant  {name = "constant.35"} dense<[[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]> : tensor<3x5xf32>
  %cst_0 = constant  {name = "constant.37"} dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>
  %1 = "fxpmath.real_matmul_bias"(%0, %cst, %cst_0) {clamp_max = 6.10, clamp_min = -12.25} : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
  return %1 : tensor<300x5xf32>
}

