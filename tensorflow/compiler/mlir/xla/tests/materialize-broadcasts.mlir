// RUN: xla-opt -test-xla-materialize-broadcasts -split-input-file %s -o - | FileCheck --dump-input=fail %s

// CHECK-LABEL: @addBroadcastRhs
func @addBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastLhs
func @addBroadcastLhs(%arg0: tensor<4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %arg1 : tensor<1x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastEqual
func @addBroadcastEqual(%arg0: tensor<4x1xf32>, %arg1: tensor<1x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x1xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<4x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4x1xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastMultidimension
func @addBroadcastMultidimension(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1x4xf32>) -> tensor<1x1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %arg1 : tensor<1x1x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  return %0 : tensor<1x1x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastBothArgs
func @addBroadcastBothArgs(%arg0: tensor<1x2xf32>, %arg1: tensor<3x2x1xf32>) -> tensor<3x2x2xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<3x2x2xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<3x2x1xf32>) -> tensor<3x2x2xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<3x2x2xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xf32>, tensor<3x2x1xf32>) -> tensor<3x2x2xf32>
  return %0 : tensor<3x2x2xf32>
}

// -----

// CHECK-LABEL: @addBroadcastScalar
func @addBroadcastScalar(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %arg0, %[[BROADCAST1]] : tensor<4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %arg0, %arg1 : tensor<4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @addUnranked
func @addUnranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %arg0, %arg1 : tensor<*xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @atan2BroadcastRhs
func @atan2BroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.atan2 %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.atan2"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @divBroadcastRhs
func @divBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.divide %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @maxBroadcastRhs
func @maxBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.maximum %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.maximum"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @minBroadcastRhs
func @minBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.minimum %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.minimum"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @mulBroadcastRhs
func @mulBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.multiply %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.multiply"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @powBroadcastRhs
func @powBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.power %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.power"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @remainderBroadcastRhs
func @remainderBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.remainder %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.remainder"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @shiftLeftBroadcastRhs
func @shiftLeftBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.shift_left %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.shift_left"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @shiftRightArithmeticBroadcastRhs
func @shiftRightArithmeticBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.shift_right_arithmetic %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.shift_right_arithmetic"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @shiftRightLogicalBroadcastRhs
func @shiftRightLogicalBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.shift_right_logical %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.shift_right_logical"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @subBroadcastRhs
func @subBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.subtract %arg0, %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.subtract"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @andBroadcastRhs
func @andBroadcastRhs(%arg0: tensor<1x4xi32>, %arg1: tensor<4xi32>) -> tensor<1x4xi32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.and %arg0, %[[BROADCAST1]] : tensor<1x4xi32>
  %0 = "xla_hlo.and"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @orBroadcastRhs
func @orBroadcastRhs(%arg0: tensor<1x4xi32>, %arg1: tensor<4xi32>) -> tensor<1x4xi32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.or %arg0, %[[BROADCAST1]] : tensor<1x4xi32>
  %0 = "xla_hlo.or"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @xorBroadcastRhs
func @xorBroadcastRhs(%arg0: tensor<1x4xi32>, %arg1: tensor<4xi32>) -> tensor<1x4xi32> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.xor %arg0, %[[BROADCAST1]] : tensor<1x4xi32>
  %0 = "xla_hlo.xor"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xi32>, tensor<4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @clampBroadcast
// CHECK-SAME: (%[[MIN:.+]]: tensor<f32>, %[[VAL:.+]]: tensor<4xf32>, %[[MAX:.+]]: tensor<f32>)
func @clampBroadcast(%min: tensor<f32>, %value: tensor<4xf32>, %max: tensor<f32>) -> tensor<4xf32> {
  // CHECK-DAG: %[[MIN_BC:.+]] = "xla_hlo.broadcast"(%[[MIN]]) {broadcast_sizes = dense<4> : tensor<1xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK-DAG: %[[MAX_BC:.+]] = "xla_hlo.broadcast"(%[[MAX]]) {broadcast_sizes = dense<4> : tensor<1xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK: "xla_hlo.clamp"(%[[MIN_BC]], %[[VAL]], %[[MAX_BC]]) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = "xla_hlo.clamp"(%min, %value, %max) : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @compareBroadcastRhs
func @compareBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xi1> {
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = "xla_hlo.compare"(%arg0, %[[BROADCAST1]]) {comparison_direction = "NE"} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xi1>
  return %0 : tensor<1x4xi1>
}

// -----

// CHECK-LABEL: @dynamicCompareBroadcastRhs
func @dynamicCompareBroadcastRhs(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xi1> {
  // CHECK-NEXT: %[[DIM0:.*]] = dim %arg0, 0 : tensor<?x?xf32>
  // CHECK-NEXT: %c1 = constant 1 : index
  // CHECK-NEXT: %[[DIM1_0:.*]] = dim %arg0, 1 : tensor<?x?xf32>
  // CHECK-NEXT: %[[DIM1_1:.*]] = dim %arg1, 0 : tensor<?xf32>
  // CHECK-NEXT: %[[CMPI:.*]] = cmpi "eq", %[[DIM1_0]], %c1 : index
  // CHECK-NEXT: %[[DIM1:.*]] = select %[[CMPI]], %[[DIM1_0]], %[[DIM1_1]] : index
  // CHECK-NEXT: %[[SHAPE:.*]] = "xla_hlo.scalars_to_dimension_tensor"(%[[DIM0]], %[[DIM1]]) : (index, index) -> tensor<2xindex>
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[SHAPE]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: "xla_hlo.compare"(%[[BROADCAST0]], %[[BROADCAST1]]) {comparison_direction = "NE"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  %0 = "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----

// CHECK-LABEL: @dynamicBroadcastAdd
func @dynamicBroadcastAdd(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
  // CHECK-NEXT: %[[DIM0:.*]] = dim %arg0, 0 : tensor<?x?xf32>
  // CHECK-NEXT: %c1 = constant 1 : index
  // CHECK-NEXT: %[[DIM1_0:.*]] = dim %arg0, 1 : tensor<?x?xf32>
  // CHECK-NEXT: %[[DIM1_1:.*]] = dim %arg1, 0 : tensor<?xf32>
  // CHECK-NEXT: %[[CMPI:.*]] = cmpi "eq", %[[DIM1_0]], %c1 : index
  // CHECK-NEXT: %[[DIM1:.*]] = select %[[CMPI]], %[[DIM1_0]], %[[DIM1_1]] : index
  // CHECK-NEXT: %[[SHAPE:.*]] = "xla_hlo.scalars_to_dimension_tensor"(%[[DIM0]], %[[DIM1]]) : (index, index) -> tensor<2xindex>
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[SHAPE]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<?x?xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamicBroadcastAddScalar
func @dynamicBroadcastAddScalar(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) -> tensor<?x?xf32> {
  // CHECK-NEXT: %[[DIM0:.*]] = dim %arg0, 0 : tensor<?x?xf32>
  // CHECK-NEXT: %[[DIM1:.*]] = dim %arg0, 1 : tensor<?x?xf32>
  // CHECK-NEXT: %[[SHAPE:.*]] = "xla_hlo.scalars_to_dimension_tensor"(%[[DIM0]], %[[DIM1]]) : (index, index) -> tensor<2xindex>
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %[[SHAPE]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[SHAPE]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT: xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<?x?xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
