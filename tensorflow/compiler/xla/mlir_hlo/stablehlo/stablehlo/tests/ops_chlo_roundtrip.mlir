// RUN: stablehlo-opt %s -split-input-file | FileCheck %s

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

// -----

// CHECK-LABEL: func @chlo_broadcast_and(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xi8>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xi8>
// CHECK:       %[[T:.*]] = chlo.broadcast_and %[[A0]], %[[A1]] : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
// CHECK:       return %[[T]] : tensor<2x3x4xi8>
func.func @chlo_broadcast_and(%arg0: tensor<2x3x4xi8>, %arg1: tensor<2x3x4xi8>) -> tensor<2x3x4xi8> {
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
  return %0 : tensor<2x3x4xi8>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_or(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xi8>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xi8>
// CHECK:       %[[T:.*]] = chlo.broadcast_or %[[A0]], %[[A1]] : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
// CHECK:       return %[[T]] : tensor<2x3x4xi8>
func.func @chlo_broadcast_or(%arg0: tensor<2x3x4xi8>, %arg1: tensor<2x3x4xi8>) -> tensor<2x3x4xi8> {
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
  return %0 : tensor<2x3x4xi8>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_xor(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xi8>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xi8>
// CHECK:       %[[T:.*]] = chlo.broadcast_xor %[[A0]], %[[A1]] : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
// CHECK:       return %[[T]] : tensor<2x3x4xi8>
func.func @chlo_broadcast_xor(%arg0: tensor<2x3x4xi8>, %arg1: tensor<2x3x4xi8>) -> tensor<2x3x4xi8> {
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<2x3x4xi8>, tensor<2x3x4xi8>) -> tensor<2x3x4xi8>
  return %0 : tensor<2x3x4xi8>
}

// -----

// CHECK-LABEL: func @chlo_next_after(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.next_after %[[A0]], %[[A1]] : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_next_after(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.next_after %arg0, %arg1 : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_polygamma(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.polygamma %[[A0]], %[[A1]] : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_polygamma(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.polygamma %arg0, %arg1 : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_zeta(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.zeta %[[A0]], %[[A1]] : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
// CHECK:       return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_zeta(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xf64> {
  %0 = chlo.zeta %arg0, %arg1 : tensor<2x3x4xf64>, tensor<2x3x4xf64> -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL: func @chlo_broadcast_complex(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_complex %[[A0]], %[[A1]] : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xcomplex<f32>>
// CHECK:       return %[[T]] : tensor<2x3x4xcomplex<f32>>
func.func @chlo_broadcast_complex(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xcomplex<f32>> {
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xcomplex<f32>>
  return %0 : tensor<2x3x4xcomplex<f32>>
}


// -----

// CHECK-LABEL: func @chlo_broadcast_compare(
// CHECK-SAME:  %[[A0:.*]]: tensor<2x3x4xf64>,
// CHECK-SAME:  %[[A1:.*]]: tensor<2x3x4xf64>
// CHECK:       %[[T:.*]] = chlo.broadcast_compare %[[A0]], %[[A1]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xi1>
// CHECK:       return %[[T]] : tensor<2x3x4xi1>
func.func @chlo_broadcast_compare(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> tensor<2x3x4xi1> {
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction LT>} : (tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// -----

// CHECK-LABEL:  func @chlo_broadcast_select
// CHECK-SAME:   %[[A0:.*0]]: tensor<2x3x4xf64>,
// CHECK-SAME:   %[[A1:.*1]]: tensor<2x3x4xf64>,
// CHECK-SAME:   %[[A2:.*2]]: tensor<2x3x4xi1>)
// CHECK:        %[[T:.*]] = chlo.broadcast_select %[[A2]], %[[A0]], %[[A1]] : (tensor<2x3x4xi1>, tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
// CHECK:        return %[[T]] : tensor<2x3x4xf64>
func.func @chlo_broadcast_select(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>, %arg2: tensor<2x3x4xi1>) -> tensor<2x3x4xf64> {
  %0 = chlo.broadcast_select %arg2, %arg0, %arg1 : (tensor<2x3x4xi1>, tensor<2x3x4xf64>, tensor<2x3x4xf64>) -> tensor<2x3x4xf64>
  return %0 : tensor<2x3x4xf64>
}

// -----

// CHECK-LABEL:  func @chlo_top_k(
// CHECK-SAME:   %[[A0:.*]]: tensor<16x16xi32>)
// CHECK:        %[[V:.*]], %[[I:.*]] = chlo.top_k(%[[A0]], k = 8) : tensor<16x16xi32> -> (tensor<16x8xi32>, tensor<16x8xi32>)
// CHECK:        return %[[V]], %[[I]] : tensor<16x8xi32>, tensor<16x8xi32>
func.func @chlo_top_k(%arg : tensor<16x16xi32>) -> (tensor<16x8xi32>, tensor<16x8xi32>) {
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xi32> -> (tensor<16x8xi32>, tensor<16x8xi32>)
  return %1#0, %1#1 : tensor<16x8xi32>, tensor<16x8xi32>
}

// -----

// CHECK-LABEL:  func @chlo_minimum_broadcast_shapes(
// CHECK-SAME:   %[[A0:.*]]: tensor<?xindex>,
// CHECK-SAME:   %[[A1:.*]]: tensor<?xindex>
// CHECK:        %[[T:.*]]:2 = chlo.minimum_broadcast_shapes %[[A0]], %[[A1]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
// CHECK:        return %[[T]]#0, %[[T]]#1 : tensor<?xindex>, tensor<?xindex>
func.func @chlo_minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) -> (tensor<?xindex>, tensor<?xindex>) {
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  func.return %0, %1 : tensor<?xindex>, tensor<?xindex>
}

// -----

// CHECK-LABEL:  func @chlo_reshape_dynamic(
// CHECK-SAME:   %[[A0:.*]]: tensor<?xf32>,
// CHECK-SAME:   %[[A1:.*]]: tensor<2xi32>
// CHECK:        %[[T:.*]] = "chlo.dynamic_reshape"(%[[A0]], %[[A1]]) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:        return %[[T]] : tensor<?x?xf32>
func.func @chlo_reshape_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "chlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL:  func @chlo_rank_specialization_cluster
// CHECK-SAME:   %[[A0:.*0]]: tensor<*xf32>,
// CHECK-SAME:   %[[A1:.*1]]: tensor<*xf32>,
// CHECK-SAME:   %[[A2:.*2]]: tensor<*xf32>)
// CHECK-NEXT:   %[[T:.*]] = "chlo.rank_specialization_cluster"(%[[A0]], %[[A1]], %[[A2]])
// CHECK:        ^bb0(%[[A3:.*]]: tensor<*xf32>, %[[A4:.*]]: tensor<*xf32>, %[[A5:.*]]: tensor<*xf32>):
// CHECK:          "chlo.rank_specialization_cluster_yield"(%[[A3]]) : (tensor<*xf32>) -> ()
// CHECK:        }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:        return %[[T]] : tensor<*xf32>
func.func @chlo_rank_specialization_cluster(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
    "chlo.rank_specialization_cluster_yield"(%arg0_) : (tensor<*xf32>) -> ()
  }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
