// RUN: mlir-hlo-opt -lhlo-legalize-to-tensor-op %s -o - | FileCheck %s

// test: `memref -> bufferization.to_tensor -> tensor.extract` -> `memref -> memref.load`
// CHECK-LABEL: forward_extract_op
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: memref<3xindex>)
func @forward_extract_op(%arg0: memref<?x?xf32>, %arg1: memref<3xindex>) -> memref<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK-NOT: bufferization.to_tensor
  // CHECK-NOT: tensor.extract
  // CHECK: %[[DIM0:.*]] = memref.load %[[ARG1]][%c0]
  // CHECK: %[[DIM1:.*]] = memref.load %[[ARG1]][%c1]
  // CHECK: %[[DIM2:.*]] = memref.load %[[ARG1]][%c2]
  // CHECK: memref.alloc(%[[DIM0]], %[[DIM1]], %[[DIM2]])
  %0 = bufferization.to_tensor %arg1 : memref<3xindex>
  %1 = tensor.extract %0[%c0] : tensor<3xindex>
  %2 = tensor.extract %0[%c1] : tensor<3xindex>
  %3 = tensor.extract %0[%c2] : tensor<3xindex>
  %4 = memref.alloc(%1, %2, %3) : memref<?x?x?xf32>
  "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %4) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (memref<?x?xf32>, memref<3xindex>, memref<?x?x?xf32>) -> ()
  return %4 : memref<?x?x?xf32>
}

// -----

// test: `memref -> bufferization.to_tensor -> shape.shape_of` -> `memref -> shape.shape_of`
// CHECK-LABEL: forward_shape_of_op
// CHECK-SAME: (%[[ARG:.*]]: memref<?x?xf32>)
func @forward_shape_of_op(%arg0: memref<?x?xf32>) -> tensor<2xindex> {
  // CHECK-NOT: bufferization.to_tensor
  // CHECK: shape.shape_of %[[ARG]] : memref<?x?xf32> -> tensor<2xindex>
  %0 = bufferization.to_tensor %arg0 : memref<?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  return %1 : tensor<2xindex>
}
