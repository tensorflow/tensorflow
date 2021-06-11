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

// -----

// CHECK-LABEL: func @dynamic_iota
// CHECK-SAME: (%[[SHAPE:.*]]: memref<2xindex>) -> memref<5x?xi32>
func @dynamic_iota(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK-NOT: tensor_load
  // CHECK: %[[DIM0:.*]] = memref.load %[[SHAPE]][%c1]
  // CHECK: %[[OUTPUT:.*]] = memref.alloc(%[[DIM0]])
  // CHECK: "lmhlo.dynamic_iota"(%[[SHAPE]], %[[OUTPUT]])
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<5x?xi32>
  return %0 : tensor<5x?xi32>
}

// -----

// CHECK-LABEL: func @dynamic_pad
// CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32>, %[[VAL:.*]]: memref<f32>,
// CHECK-SAME:  %[[LOW:.*]]: memref<2xindex>, %[[HIGH:.*]]: memref<2xindex>, %[[INTER:.*]]: memref<2xindex>) -> memref<?x?xf32>
func @dynamic_pad(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: tensor<2xindex>, %arg3: tensor<2xindex>, %arg4: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK-NOT: tensor_load
  // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG]], %c0 : memref<?x?xf32>
  // CHECK: %[[TMP1:.*]] = memref.load %[[LOW]][%c0] : memref<2xindex>
  // CHECK: %[[TMP2:.*]] = memref.load %[[HIGH]][%c0] : memref<2xindex>
  // CHECK: %[[TMP3:.*]] = memref.load %[[INTER]][%c0] : memref<2xindex>
  // CHECK: %[[TMP4:.*]] = cmpi slt, %[[DIM0]], %c1 : index
  // CHECK: %[[TMP5:.*]] = subi %[[DIM0]], %c1 : index
  // CHECK: %[[TMP6:.*]] = select %[[TMP4]], %c0, %[[TMP5]] : index
  // CHECK: %[[TMP7:.*]] = muli %[[TMP3]], %[[TMP6]] : index
  // CHECK: %[[TMP8:.*]] = addi %[[TMP7]], %[[DIM0]] : index
  // CHECK: %[[TMP9:.*]] = addi %[[TMP8]], %[[TMP1]] : index
  // CHECK: %[[TMP10:.*]] = addi %[[TMP9]], %[[TMP2]] : index
  // CHECK: %[[TMP11:.*]] = memref.dim %[[ARG]], %c1 : memref<?x?xf32>
  // CHECK: %[[TMP12:.*]] = memref.load %[[LOW]][%c1] : memref<2xindex>
  // CHECK: %[[TMP13:.*]] = memref.load %[[HIGH]][%c1] : memref<2xindex>
  // CHECK: %[[TMP14:.*]] = memref.load %[[INTER]][%c1] : memref<2xindex>
  // CHECK: %[[TMP15:.*]] = cmpi slt, %[[TMP11]], %c1 : index
  // CHECK: %[[TMP16:.*]] = subi %[[TMP11]], %c1 : index
  // CHECK: %[[TMP17:.*]] = select %[[TMP15]], %c0, %[[TMP16]] : index
  // CHECK: %[[TMP18:.*]] = muli %[[TMP14]], %[[TMP17]] : index
  // CHECK: %[[TMP19:.*]] = addi %[[TMP18]], %[[TMP11]] : index
  // CHECK: %[[TMP20:.*]] = addi %[[TMP19]], %[[TMP12]] : index
  // CHECK: %[[TMP21:.*]] = addi %[[TMP20]], %[[TMP13]] : index
  // CHECK: %[[OUT:.*]] = memref.alloc(%[[TMP10]], %[[TMP21]]) : memref<?x?xf32>
  // CHECK: "lmhlo.dynamic_pad"(%[[ARG]], %[[VAL]], %[[LOW]], %[[HIGH]], %[[INTER]], %[[OUT]])
  %0 = "mhlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?x?xf32>, tensor<f32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}
