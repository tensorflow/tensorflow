// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo=convert-to-lmhlo-only=true \
// RUN:  -canonicalize -lhlo-legalize-tensor-load-op %s -o - | FileCheck %s

// CHECK-LABEL: func @dynamic_reshape
// CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32>, %[[SHAPE:.*]]: memref<3xindex>) -> memref<?x?x?xf32>
func @dynamic_reshape(%lhs: tensor<?x?xf32>, %rhs: tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: tensor_load
  // CHECK: %[[DIM0:.*]] = memref.load %[[SHAPE]][%c0]
  // CHECK: %[[DIM1:.*]] = memref.load %[[SHAPE]][%c1]
  // CHECK: %[[DIM2:.*]] = memref.load %[[SHAPE]][%c2]
  // CHECK: %[[OUTPUT:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]], %[[DIM2]])
  // CHECK: "lmhlo.dynamic_reshape"(%[[ARG]], %[[SHAPE]], %[[OUTPUT]])
  // CHECK: return %[[OUTPUT]]
  %result = "mhlo.dynamic_reshape"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
// CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32>, %[[SHAPE:.*]]: memref<3xindex>) -> memref<?x?x?xf32>
func @dynamic_broadcast_in_dim(%operand: tensor<?x?xf32>, %shape: tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: tensor_load
  // CHECK: %[[DIM0:.*]] = memref.load %[[SHAPE]][%c0]
  // CHECK: %[[DIM1:.*]] = memref.load %[[SHAPE]][%c1]
  // CHECK: %[[DIM2:.*]] = memref.load %[[SHAPE]][%c2]
  // CHECK: %[[OUTPUT:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]], %[[DIM2]])
  // CHECK: "lmhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[SHAPE]], %[[OUTPUT]])
  // CHECK: return %[[OUTPUT]]
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}