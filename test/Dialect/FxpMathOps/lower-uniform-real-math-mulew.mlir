// RUN: mlir-opt %s -split-input-file -fxpmath-lower-uniform-real-math -pass-pipeline='func(canonicalize)' -verify-diagnostics | FileCheck %s --dump-input=always

// -----
// Verify lowering when operands and result have the same fixedpoint scale.
// CHECK-LABEL: real_mulew_fixedpoint
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 3.875e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 1.065e-1>>
func @real_mulew_fixedpoint(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant.uniform<i8:f32, 3.875000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %2 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi32>
  // CHECK-NEXT: %3 = "fxpmath.convertis"(%1) : (tensor<4xi8>) -> tensor<4xi32>
  // CHECK-NEXT: %4 = muli %2, %3 : tensor<4xi32>
  // CHECK-NEXT: %5 = "fxpmath.vs_saturating_rounding_doubling_high_mulis"(%4) {b = 1562722842 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: %6 = "fxpmath.rounding_divide_by_potis"(%5) {exponent = 5 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: %7 = "fxpmath.clampis"(%6) {clamp_max = 127 : i32, clamp_min = -128 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK-NEXT: %8 = "fxpmath.convertis"(%7) : (tensor<4xi32>) -> tensor<4xi8>
  // CHECK-NEXT: %9 = "quant.scast"(%8) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 1.065000e-01>>
  // CHECK-NEXT: return %9 : tensor<4x!quant.uniform<i8:f32, 1.065000e-01>>
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// Verify lowering when operands and result have the same fixedpoint scale
// and non-zero zero points.
// CHECK-LABEL: real_mulew_affine_clamp
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-3>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-5>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-9>>
func @real_mulew_affine_clamp(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // Just verify that the affine adds/constants and clamps are present.
  // CHECK: %cst = constant dense<3> : tensor<4xi32>
  // CHECK: %cst_0 = constant dense<5> : tensor<4xi32>
  // CHECK: %cst_1 = constant dense<-9> : tensor<4xi32>
  // CHECK: addi %2, %cst : tensor<4xi32>
  // CHECK: addi %3, %cst_0 : tensor<4xi32>
  // CHECK: muli %4, %5 : tensor<4xi32>
  // CHECK: addi %8, %cst_1 : tensor<4xi32>
  // CHECK: {clamp_max = 55 : i32, clamp_min = -73 : i32}
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) { clamp_min = -4.0, clamp_max = 4.0 } : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_mulew_unquantized_lhs
// Verifies that leaves as-is for unquantized lhs.
!type_lhs = type tensor<4xf32>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_mulew_unquantized_lhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_mul_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_mulew_unquantized_rhs
// Verifies that leaves as-is for unquantized rhs.
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4xf32>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_mulew_unquantized_rhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_mul_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_mulew_unquantized_result
// Verifies that leaves as-is for unquantized result.
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4xf32>
func @real_mulew_unquantized_result(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_mul_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// Verify lowering when operands and result have the same fixedpoint scale.
// Note that the multiplier = lhs_scale * rhs_scale / result_scale
//   = 22.740610328638496
// CHECK-LABEL: real_mulew_multiplier_gt_1
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 3.875e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 1.065e-4>>
func @real_mulew_multiplier_gt_1(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // expected-warning@+1 {{unimplemented: cannot multiply with multiplier > 1.0}}
  %0 = "fxpmath.real_mul_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}
