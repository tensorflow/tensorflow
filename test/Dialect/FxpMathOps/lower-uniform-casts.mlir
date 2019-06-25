// RUN: mlir-opt %s -split-input-file -fxpmath-lower-uniform-casts | FileCheck %s --dump-input=always

// -----
// CHECK-LABEL: dequantize_per_layer_fixedpoint
!type_input = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4xf32>
func @dequantize_per_layer_fixedpoint(%arg0 : !type_input) -> !type_result {
  // CHECK: %cst = constant dense<6.250000e-02> : tensor<4xf32>
  // CHECK-NEXT: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi32>
  // CHECK-NEXT: %2 = "fxpmath.convertistof"(%1) : (tensor<4xi32>) -> tensor<4xf32>
  // CHECK-NEXT: %3 = mulf %2, %cst : tensor<4xf32>
  // CHECK-NEXT: return %3 : tensor<4xf32>
  %0 = "quant.dcast"(%arg0) : (!type_input) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: dequantize_per_layer_affine
!type_input = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-36>>
!type_result = type tensor<4xf32>
func @dequantize_per_layer_affine(%arg0 : !type_input) -> !type_result {
  // CHECK: %cst = constant dense<36> : tensor<4xi32>
  // CHECK-NEXT: %cst_0 = constant dense<6.250000e-02> : tensor<4xf32>
  // CHECK-NEXT: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02:-36>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi32>
  // CHECK-NEXT: %2 = addi %1, %cst : tensor<4xi32>
  // CHECK-NEXT: %3 = "fxpmath.convertistof"(%2) : (tensor<4xi32>) -> tensor<4xf32>
  // CHECK-NEXT: %4 = mulf %3, %cst_0 : tensor<4xf32>
  // CHECK-NEXT: return %4 : tensor<4xf32>
  %0 = "quant.dcast"(%arg0) : (!type_input) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: dequantize_per_axis_fixedpoint
!type_input = type tensor<4x!quant.uniform<i8:f32:0, {6.25e-2,3.26e-2,4.25e-2,1.23e-2}>>
!type_result = type tensor<4xf32>
func @dequantize_per_axis_fixedpoint(%arg0 : !type_input) -> !type_result {
  // expected-warning@+1 {{unimplemented: per-axis uniform dequantization}}
  %0 = "quant.dcast"(%arg0) : (!type_input) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: dequantize_per_axis_affine
!type_input = type tensor<4x!quant.uniform<i8:f32:0, {6.25e-2,3.26e-2,4.25e-2,1.23e-2}>>
!type_result = type tensor<4xf32>
func @dequantize_per_axis_affine(%arg0 : !type_input) -> !type_result {
  // expected-warning@+1 {{unimplemented: per-axis uniform dequantization}}
  %0 = "quant.dcast"(%arg0) : (!type_input) -> (!type_result)
  return %0 : !type_result
}

// -----
// Noop dequantize should be skipped (will be canonicalized away later).
// CHECK-LABEL: dequantize_noop
!type_input = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-36>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-36>>
func @dequantize_noop(%arg0 : !type_input) -> !type_result {
  // CHECK: %0 = "quant.dcast"(%arg0)
  %0 = "quant.dcast"(%arg0) : (!type_input) -> (!type_result)
  return %0 : !type_result
}
