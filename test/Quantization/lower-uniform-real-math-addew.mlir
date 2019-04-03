// RUN: mlir-opt %s -split-input-file -quant-lower-uniform-real-math | FileCheck %s --dump-input=fail

// -----
// Verify lowering when operands and result have the same fixedpoint pot scale.
// CHECK-LABEL: real_addew_fixedpoint_same_scale
//      CHECK: %0 = "quant.scast"(%arg0) : (tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>) -> tensor<4xi8>
// CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>) -> tensor<4xi8>
// CHECK-NEXT: %2 = "quant.saturating_addi"(%0, %1) {clamp_max: 127 : i32, clamp_min: -128 : i32} : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %3 = "quant.scast"(%2) : (tensor<4xi8>) -> tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
// CHECK-NEXT: return %3 : tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_fixedpoint_same_scale(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// Verify lowering when the rhs is a shifted pot scale compared to lhs and result.
// CHECK-LABEL: real_addew_fixedpoint_rhs_shift
//      CHECK: %0 = "quant.scast"(%arg0) : (tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>) -> tensor<4xi8>
// CHECK-NEXT: %1 = "quant.scast"(%arg1) : (tensor<4x!quant<"uniform[i8:f32]{7.812500e-03}">>) -> tensor<4xi8>
// CHECK-NEXT: %2 = "quant.rounding_divide_by_poti"(%1) {exponent: 3 : i32} : (tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %3 = "quant.saturating_addi"(%0, %2) {clamp_max: 127 : i32, clamp_min: -128 : i32} : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %4 = "quant.scast"(%3) : (tensor<4xi8>) -> tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
// CHECK-NEXT: return %4 : tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{7.8125e-03}">>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_fixedpoint_rhs_shift(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// Verify lowering when the lhs is a shifted pot scale compared to lhs and result.
// CHECK-LABEL: real_addew_fixedpoint_lhs_shift
//      CHECK: %0 = "quant.scast"(%arg1) : (tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>) -> tensor<4xi8>
// CHECK-NEXT: %1 = "quant.scast"(%arg0) : (tensor<4x!quant<"uniform[i8:f32]{7.812500e-03}">>) -> tensor<4xi8>
// CHECK-NEXT: %2 = "quant.rounding_divide_by_poti"(%1) {exponent: 3 : i32} : (tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %3 = "quant.saturating_addi"(%0, %2) {clamp_max: 127 : i32, clamp_min: -128 : i32} : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %4 = "quant.scast"(%3) : (tensor<4xi8>) -> tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
// CHECK-NEXT: return %4 : tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{7.8125e-03}">>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_fixedpoint_lhs_shift(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// The RHS quant parameters proscribe a range of [-8..8) so an explicit clamp
// of [-4..4] should result in an integral clamp range of [-64..64].
// CHECK-LABEL: real_addew_fixedpoint_clamp
//      CHECK: %0 = "quant.scast"(%arg1) : (tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>) -> tensor<4xi8>
// CHECK-NEXT: %1 = "quant.scast"(%arg0) : (tensor<4x!quant<"uniform[i8:f32]{7.812500e-03}">>) -> tensor<4xi8>
// CHECK-NEXT: %2 = "quant.rounding_divide_by_poti"(%1) {exponent: 3 : i32} : (tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %3 = "quant.saturating_addi"(%0, %2) {clamp_max: 64 : i32, clamp_min: -64 : i32} : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %4 = "quant.scast"(%3) : (tensor<4xi8>) -> tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
// CHECK-NEXT: return %4 : tensor<4x!quant<"uniform[i8:f32]{6.250000e-02}">>
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{7.8125e-03}">>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_fixedpoint_clamp(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  %0 = "quant.real_add_ew"(%arg0, %arg1) { clamp_min:-4.0, clamp_max:4.0 }
      : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_lhs
// Verifies that leaves as-is for unquantized lhs.
!type_lhs = type tensor<4xf32>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_unquantized_lhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "quant.real_add_ew"(%arg0, %arg1)
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_rhs
// Verifies that leaves as-is for unquantized rhs.
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_rhs = type tensor<4xf32>
!type_result = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
func @real_addew_unquantized_rhs(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "quant.real_add_ew"(%arg0, %arg1)
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}

// -----
// CHECK-LABEL: real_addew_unquantized_result
// Verifies that leaves as-is for unquantized result.
!type_lhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_rhs = type tensor<4x!quant<"uniform[i8:f32]{6.25e-2}">>
!type_result = type tensor<4xf32>
func @real_addew_unquantized_result(%arg0 : !type_lhs, %arg1: !type_rhs) -> !type_result {
  // CHECK: %0 = "quant.real_add_ew"(%arg0, %arg1)
  %0 = "quant.real_add_ew"(%arg0, %arg1) : (!type_lhs, !type_rhs) -> (!type_result)
  return %0 : !type_result
}
