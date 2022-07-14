// RUN: mlir-hlo-opt %s -split-input-file | mlir-hlo-opt | FileCheck %s

// CHECK-LABEL: func @chlo_acos(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.acos %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_acos(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.acos %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_acosh(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.acosh %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_acosh(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.acosh %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_asin(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.asin %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_asin(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.asin %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_asinh(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.asinh %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_asinh(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.asinh %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----


// CHECK-LABEL: func @chlo_atan(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.atan %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_atan(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.atan %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_atanh(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.atanh %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_atanh(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.atanh %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_bessel_i1e(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.bessel_i1e %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_bessel_i1e(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.bessel_i1e %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_conj(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.conj %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_conj(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.conj %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_cosh(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.cosh %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_cosh(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.cosh %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_digamma(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.digamma %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_digamma(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.digamma %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_erf(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.erf %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_erf(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.erf %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_erfc(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.erfc %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_erfc(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.erfc %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_lgamma(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.lgamma %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_lgamma(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.lgamma %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_sinh(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.sinh %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_sinh(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.sinh %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_tan(
// CHECK-SAME:  %[[A:.*]]: tensor<8x8xf64>
// CHECK:       %[[T:.*]] = chlo.tan %[[A]] : tensor<8x8xf64> -> tensor<8x8xf64>
// CHECK:       return %[[T]] : tensor<8x8xf64>
func.func @chlo_tan(%arg0: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %0 = chlo.tan %arg0 : tensor<8x8xf64> -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_add(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_add %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_add(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_atan2(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_atan2 %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_atan2(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_divide(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_divide %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_divide(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_maximum(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_maximum %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_maximum(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_minimum(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_minimum %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_minimum(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_multiply(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_multiply %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_multiply(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_next_after(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_next_after %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_next_after(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_next_after %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_polygamma(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_polygamma %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_polygamma(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_polygamma %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_power(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_power %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_power(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_remainder(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_remainder %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_remainder(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_shift_left(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_shift_left %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_shift_left(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_shift_right_arithmetic(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_shift_right_arithmetic %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_shift_right_arithmetic(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_shift_right_logical(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_shift_right_logical %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_shift_right_logical(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_subtract(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_subtract %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_subtract(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_zeta(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_zeta %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_zeta(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_zeta %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}
