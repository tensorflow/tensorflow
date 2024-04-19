// RUN: mlir-hlo-opt %s --hlo-one-shot-bufferize \
// RUN:  --cse --canonicalize --split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: @tensor.extract
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func.func @tensor.extract(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = arith.constant 0 : index
  %result = tensor.extract %arg[%c0] : tensor<?xf32>
  func.return %result : f32
}

// -----

// CHECK-LABEL: @tensor.from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
func.func @tensor.from_elements(%a : f32) -> f32 {
  // CHECK-DAG: %[[B:.*]] = arith.constant 1.2
  // CHECK-DAG: %[[C:.*]] = arith.constant 2.3
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[MEM:.*]] = memref.alloc
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  %b = arith.constant 1.2 : f32
  %c = arith.constant 2.3 : f32
  %tfe = tensor.from_elements %a, %b, %c : tensor<3xf32>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<3xf32>
  func.return %result : f32
}

// -----

// CHECK-LABEL: @tensor.generate
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> index
func.func @tensor.generate(%arg : tensor<*xf32>) -> index {
  // CHECK: %[[SIZE:.*]] = memref.rank %[[ARG]] : memref<*xf32>
  // CHECK: %[[MEM:.*]] = memref.alloc(%[[SIZE]]) {{.*}} : memref<?xindex>
  // CHECK: linalg.map
  // CHECK: outs(%[[MEM]] : memref<?xindex>)
  // CHECK:   %[[INDEX:.*]] = linalg.index 0
  // CHECK:   %[[ELEM:.*]] = memref.dim %[[ARG]], %[[INDEX]] : memref<*xf32>
  // CHECK:   linalg.yield %[[ELEM]]
  // CHECK: }
  %size = tensor.rank %arg : tensor<*xf32>
  %tfe = tensor.generate %size {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<?xindex>
  func.return %result : index
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]>
// CHECK-SAME: alignment = 64
// CHECK: @const
// CHECK-SAME: -> memref<3xf32>
func.func @const() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<4.000000e+00>
// CHECK-SAME: alignment = 64
// CHECK: @const_splat
// CHECK-SAME: -> memref<3xf32>
func.func @const_splat() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<4.0> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK-LABEL: @tensor_reshape
// CHECK-SAME: (%[[T:.*]]: memref<1x2x2xf32>)
func.func @tensor_reshape(%t : tensor<1x2x2xf32>) -> tensor<4xf32> {
  // CHECK: memref.collapse_shape %[[T]] {{.*}} : memref<1x2x2xf32> into memref<4xf32>
  %result = tensor.collapse_shape %t [[0, 1, 2]] : tensor<1x2x2xf32> into tensor<4xf32>
  func.return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @slice
// CHECK-SAME: (%[[T:.*]]: memref<3xi32>)
func.func @slice(%t : tensor<3xi32>) -> tensor<1xi32> {
  // CHECK: memref.subview %[[T]][0] [1] [1] : memref<3xi32> to memref<1xi32, strided<[1]>>
  %result = tensor.extract_slice %t[0] [1] [1] : tensor<3xi32> to tensor<1xi32>
  func.return %result : tensor<1xi32>
}

// -----

func.func @dynamic_broadcast_return(%t : tensor<?x?xf32>, %shape : tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK: memref.copy
  %bcast = "mhlo.dynamic_broadcast_in_dim"(%t, %shape) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %bcast : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @arith_select
// CHECK-SAME: %[[C:.*]]: memref<i1>,
// CHECK-SAME: %[[LHS:.*]]: memref<1xf32>,
// CHECK-SAME: %[[RHS:.*]]: memref<1xf32>
func.func @arith_select(%c : tensor<i1>, %lhs: tensor<1xf32>, %rhs: tensor<1xf32>)
                  -> tensor<1xf32> {
  // CHECK: %[[COND:.*]] = memref.load %[[C]][]
  // CHECK: %[[RESULT:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
  // CHECK-SAME:             : memref<1xf32>
  %cond = tensor.extract %c[] : tensor<i1>
  %result = arith.select %cond, %lhs, %rhs : tensor<1xf32>
  func.return %result : tensor<1xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>
func.func @init_tensor_multiple_users(%lhs: tensor<10xf32>,
    %rhs: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %init = bufferization.alloc_tensor() : tensor<10xf32>
  %add = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %a = arith.addf %l, %r : f32
    linalg.yield %a : f32
  } -> tensor<10xf32>
  %sub = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %s = arith.subf %l, %r : f32
    linalg.yield %s : f32
  } -> tensor<10xf32>
  func.return %add, %sub : tensor<10xf32>, tensor<10xf32>
}
// CHECK-LABEL: func @init_tensor_multiple_users

// -----

// Test that scf ops are bufferized
// CHECK-LABEL:   func @if(
// CHECK-SAME:             %[[PRED:.*]]: i1,
// CHECK-SAME:             %[[TRUE_TENSOR:.*]]: memref<?xf32>,
// CHECK-SAME:             %[[FALSE_TENSOR:.*]]: memref<?xf32>) -> memref<?xf32> {
// CHECK:             %[[IF_RES:.*]] = arith.select %[[PRED]], %[[TRUE_TENSOR]], %[[FALSE_TENSOR]]
func.func @if(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %0 = scf.if %pred -> (tensor<?xf32>) {
    scf.yield %true_val : tensor<?xf32>
  } else {
    scf.yield %false_val : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}
