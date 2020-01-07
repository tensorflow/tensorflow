// RUN: tf-opt -test-xla-materialize-broadcasts -split-input-file %s -o - | FileCheck --dump-input=fail %s

// CHECK-LABEL: @addBroadcastRhs
func @addBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastLhs
func @addBroadcastLhs(%arg0: tensor<4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @addBroadcastMultidimension
func @addBroadcastMultidimension(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1x4xf32>) -> tensor<1x1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<1x1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x1x4xf32>
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
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// TODO(scotttodd): Check if this use of dynamic shapes should pass verification
// CHECK-LABEL: @addBroadcastLhsDynamicShape
func @addBroadcastLhsDynamicShape(%arg0: tensor<?xf32>, %arg1: tensor<1x3xf32>) -> tensor<?x3xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>) -> tensor<?x3xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x3xf32>) -> tensor<?x3xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.add %[[BROADCAST0]], %[[BROADCAST1]] : tensor<?x3xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?xf32>, tensor<1x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
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

// CHECK-LABEL: @divBroadcastRhs
func @divBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.div %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.div"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @maxBroadcastRhs
func @maxBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.max %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.max"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @minBroadcastRhs
func @minBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.min %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.min"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @mulBroadcastRhs
func @mulBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.mul %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.mul"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @powBroadcastRhs
func @powBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.pow %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.pow"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: @subBroadcastRhs
func @subBroadcastRhs(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NEXT: %[[BROADCAST0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[BROADCAST1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<1x4xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = xla_hlo.sub %[[BROADCAST0]], %[[BROADCAST1]] : tensor<1x4xf32>
  %0 = "xla_hlo.sub"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
