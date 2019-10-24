// RUN: mlir-opt %s -split-input-file -fxpmath-lower-uniform-real-math -pass-pipeline='func(canonicalize)' | FileCheck %s --dump-input=always

// -----
// Verify lowering when operands and result have the same fixedpoint scale.
// CHECK-LABEL: real_addew_fixedpoint_isomorphic
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_addew_fixedpoint_isomorphic(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK-NEXT: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %2 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %3 = "fxpmath.convertis"(%1) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %4 = addi %2, %3 : tensor<4xi16>
  // CHECK-NEXT: %5 = "fxpmath.clampis"(%4) {clamp_max = 127 : i16, clamp_min = -128 : i16} : (tensor<4xi16>) -> tensor<4xi16>
  // CHECK-NEXT: %6 = "fxpmath.convertis"(%5) : (tensor<4xi16>) -> tensor<4xi8>
  // CHECK-NEXT: %7 = "quant.scast"(%6) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>
  // CHECK-NEXT: return %7 : tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// Verify lowering when operands and result have the same fixedpoint scale
// and non-zero zero points.
// CHECK-LABEL: real_addew_affine_isomorphic
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-5>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-5>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2:-5>>
func @real_addew_affine_isomorphic(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK-NEXT: %cst = constant dense<5> : tensor<4xi16>
  // CHECK-NEXT: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02:-5>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02:-5>>) -> tensor<4xi8>
  // CHECK-NEXT: %2 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %3 = "fxpmath.convertis"(%1) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %4 = addi %2, %3 : tensor<4xi16>
  // CHECK-NEXT: %5 = addi %4, %cst : tensor<4xi16>
  // CHECK-NEXT: %6 = "fxpmath.clampis"(%5) {clamp_max = 127 : i16, clamp_min = -128 : i16} : (tensor<4xi16>) -> tensor<4xi16>
  // CHECK-NEXT: %7 = "fxpmath.convertis"(%6) : (tensor<4xi16>) -> tensor<4xi8>
  // CHECK-NEXT: %8 = "quant.scast"(%7) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 6.250000e-02:-5>>
  // CHECK-NEXT: return %8 : tensor<4x!quant.uniform<i8:f32, 6.250000e-02:-5>>
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// The RHS quant parameters proscribe a range of [-8..8) so an explicit clamp
// of [-4..4] should result in an integral clamp range of [-64..64].
// CHECK-LABEL: real_addew_fixedpoint_clamp
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_addew_fixedpoint_clamp(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK-NEXT: %0 = "quant.scast"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>) -> tensor<4xi8>
  // CHECK-NEXT: %2 = "fxpmath.convertis"(%0) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %3 = "fxpmath.convertis"(%1) : (tensor<4xi8>) -> tensor<4xi16>
  // CHECK-NEXT: %4 = addi %2, %3 : tensor<4xi16>
  // CHECK-NEXT: %5 = "fxpmath.clampis"(%4) {clamp_max = 64 : i16, clamp_min = -64 : i16} : (tensor<4xi16>) -> tensor<4xi16>
  // CHECK-NEXT: %6 = "fxpmath.convertis"(%5) : (tensor<4xi16>) -> tensor<4xi8>
  // CHECK-NEXT: %7 = "quant.scast"(%6) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>
  // CHECK-NEXT: return %7 : tensor<4x!quant.uniform<i8:f32, 6.250000e-02>>
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) { clamp_min = -4.0, clamp_max = 4.0 }
      : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_lhs
// Verifies that leaves as-is for unquantized lhs.
!type_lhs = type tensor<4xf32>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_addew_unquantized_lhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_add_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_rhs
// Verifies that leaves as-is for unquantized rhs.
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4xf32>
!type_result = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
func @real_addew_unquantized_rhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_add_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_result
// Verifies that leaves as-is for unquantized result.
!type_lhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_rhs = type tensor<4x!quant.uniform<i8:f32, 6.25e-2>>
!type_result = type tensor<4xf32>
func @real_addew_unquantized_result(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "fxpmath.real_add_ew"(%arg0, %arg1)
  %0 = "fxpmath.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}
