// RUN: xla-opt -test-xla-chlo-legalize-to-hlo -split-input-file -verify-diagnostics %s -o - | FileCheck --dump-input=fail %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.
// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.add %arg0, %arg1
  %0 = xla_chlo.broadcast_add(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.atan2 %arg0, %arg1
  %0 = xla_chlo.broadcast_atan2(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.divide %arg0, %arg1
  %0 = xla_chlo.broadcast_divide(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.maximum %arg0, %arg1
  %0 = xla_chlo.broadcast_maximum(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.minimum %arg0, %arg1
  %0 = xla_chlo.broadcast_minimum(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.multiply %arg0, %arg1
  %0 = xla_chlo.broadcast_multiply(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.power %arg0, %arg1
  %0 = xla_chlo.broadcast_power(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.remainder %arg0, %arg1
  %0 = xla_chlo.broadcast_remainder(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_leftWithoutBroadcast
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_left %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_left(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_arithmeticWithoutBroadcast
func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_right_arithmetic %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_right_arithmetic(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_logicalWithoutBroadcast
func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.shift_right_logical %arg0, %arg1
  %0 = xla_chlo.broadcast_shift_right_logical(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: xla_hlo.subtract %arg0, %arg1
  %0 = xla_chlo.broadcast_subtract(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @andWithoutBroadcast
func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.and %arg0, %arg1
  %0 = xla_chlo.broadcast_and(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.or %arg0, %arg1
  %0 = xla_chlo.broadcast_or(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: xla_hlo.xor %arg0, %arg1
  %0 = xla_chlo.broadcast_xor(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// TODO(laurenzo): Implement broadcast conversions.
func @addBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error @+1 {{failed to legalize operation 'xla_chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = xla_chlo.broadcast_add(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
