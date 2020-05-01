// RUN: xla-opt -test-xla-chlo-legalize-to-hlo -split-input-file -verify-diagnostics %s -o - | FileCheck --dump-input=fail %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.
// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.add %arg0, %arg1
  %0 = xla_chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcast
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ARG0_S:.+]] = "shape.shape_of"(%[[ARG0]])
  // CHECK-DAG: %[[ARG1_S:.+]] = "shape.shape_of"(%[[ARG1]])
  // CHECK-DAG: %[[RESULT_S:.+]] = "shape.broadcast"(%[[ARG0_S]], %[[ARG1_S]])
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_S]])
  // CHECK-DAG: %[[ARG0_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: %[[ARG1_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[RESULT:.+]] = xla_hlo.add %[[ARG0_B]], %[[ARG1_B]]
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  %0 = xla_chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  // CHECK-DAG: %[[ARG0_S:.+]] = "shape.shape_of"(%[[ARG0]])
  // CHECK-DAG: %[[ARG1_S:.+]] = "shape.shape_of"(%[[ARG1]])
  // CHECK-DAG: %[[RESULT_S:.+]] = "shape.broadcast"(%[[ARG0_S]], %[[ARG1_S]])
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_S]])
  // CHECK-DAG: %[[ARG0_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[ARG1_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = "xla_hlo.complex"(%[[ARG0_B]], %[[ARG1_B]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK: return %[[RESULT]] : tensor<?x?xcomplex<f32>>
  %0 = xla_chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // CHECK-DAG: %[[ARG0_S:.+]] = "shape.shape_of"(%[[ARG0]])
  // CHECK-DAG: %[[ARG1_S:.+]] = "shape.shape_of"(%[[ARG1]])
  // CHECK-DAG: %[[RESULT_S:.+]] = "shape.broadcast"(%[[ARG0_S]], %[[ARG1_S]])
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_S]])
  // CHECK-DAG: %[[ARG0_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[ARG1_B:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = "xla_hlo.compare"(%[[ARG0_B]], %[[ARG1_B]]) {comparison_direction = "EQ"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: return %[[RESULT]] : tensor<?x?xi1>
  %0 = xla_chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarBroadcastDimensions
func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK: xla_hlo.add
  %0 = xla_chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  // CHECK: xla_hlo.add
  %0 = xla_chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = xla_chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = xla_chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.and %arg0, %arg1
  %0 = xla_chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.atan2 %arg0, %arg1
  %0 = xla_chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %0 = xla_chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // CHECK: "xla_hlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %0 = xla_chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.divide %arg0, %arg1
  %0 = xla_chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.maximum %arg0, %arg1
  %0 = xla_chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.minimum %arg0, %arg1
  %0 = xla_chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.multiply %arg0, %arg1
  %0 = xla_chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.or %arg0, %arg1
  %0 = xla_chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.power %arg0, %arg1
  %0 = xla_chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.remainder %arg0, %arg1
  %0 = xla_chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_leftWithoutBroadcast
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_left %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_arithmeticWithoutBroadcast
func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_right_arithmetic %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_logicalWithoutBroadcast
func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_right_logical %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.subtract %arg0, %arg1
  %0 = xla_chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.xor %arg0, %arg1
  %0 = xla_chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
