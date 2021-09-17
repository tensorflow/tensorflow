// RUN: kernel-gen-opt %s --copy-cleanup --split-input-file | FileCheck %s

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  builtin.func @Copy(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %size = memref.dim %rhs, %c0 : memref<?xi16>
    %lhsCasted = memref.reinterpret_cast %lhs to offset: [0], sizes: [%size], strides: [%c0] : memref<?xi16> to memref<?xi16, #map0>
    %lhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %lhsCasted, %lhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %rhsCasted = memref.reinterpret_cast %rhs to offset: [0], sizes: [%size], strides: [%c1] : memref<?xi16> to memref<?xi16, #map0>
    %rhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %rhsCasted, %rhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %outputAlloc = memref.alloc(%size) : memref<?xi16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%lhsAlloc, %rhsAlloc : memref<?xi16>, memref<?xi16>) outs(%outputAlloc : memref<?xi16>) {
    ^bb0(%arg1: i16, %arg2: i16, %arg3 : i16):
      %and = and %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    return
  }
}

// CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @Copy(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[OUTPUT:.*]] = memref.alloc(%0) : memref<?xi16>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_CASTED]], %[[RHS_CASTED]] : memref<?xi16, #[[$MAP0]]>, memref<?xi16, #[[$MAP0]]>) outs(%{{.*}} : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = and %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

// -----

// The target of the copy is also used to write.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  builtin.func @CopyWithWrite(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %size = memref.dim %rhs, %c0 : memref<?xi16>
    %lhsCasted = memref.reinterpret_cast %lhs to offset: [0], sizes: [%size], strides: [%c0] : memref<?xi16> to memref<?xi16, #map0>
    %lhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %lhsCasted, %lhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %rhsCasted = memref.reinterpret_cast %rhs to offset: [0], sizes: [%size], strides: [%c1] : memref<?xi16> to memref<?xi16, #map0>
    %rhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %rhsCasted, %rhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%lhsAlloc, %rhsAlloc : memref<?xi16>, memref<?xi16>) outs(%rhsAlloc : memref<?xi16>) {
    ^bb0(%arg1: i16, %arg2: i16, %arg3 : i16):
      %and = and %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    return
  }
}

// CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @CopyWithWrite(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_ALLOC:.*]] = memref.alloc(%0) : memref<?xi16>
// CHECK: memref.copy %[[RHS_CASTED]], %[[RHS_ALLOC]]
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_CASTED]], %[[RHS_ALLOC]] : memref<?xi16, #[[$MAP0]]>, memref<?xi16>) outs(%[[RHS_ALLOC]] : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = and %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

// -----

// The source of the copy is mutated.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  builtin.func @CopyWithMutation(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c42 = constant 42 : i16
    %size = memref.dim %rhs, %c0 : memref<?xi16>
    %lhsCasted = memref.reinterpret_cast %lhs to offset: [0], sizes: [%size], strides: [%c0] : memref<?xi16> to memref<?xi16, #map0>
    %lhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %lhsCasted, %lhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %rhsCasted = memref.reinterpret_cast %rhs to offset: [0], sizes: [%size], strides: [%c1] : memref<?xi16> to memref<?xi16, #map0>
    %rhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %rhsCasted, %rhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %outputAlloc = memref.alloc(%size) : memref<?xi16>
    memref.store %c42, %lhs[%c0] : memref<?xi16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%lhsAlloc, %rhsAlloc : memref<?xi16>, memref<?xi16>) outs(%outputAlloc : memref<?xi16>) {
    ^bb0(%arg1: i16, %arg2: i16, %arg3 : i16):
      %and = and %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    return
  }
}

// CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @CopyWithMutation(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[LHS_ALLOC:.*]] = memref.alloc(%0) : memref<?xi16>
// CHECK: memref.copy %[[LHS_CASTED]], %[[LHS_ALLOC]]
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_ALLOC:.*]] = memref.alloc(%0) : memref<?xi16>
// CHECK: memref.copy %[[RHS_CASTED]], %[[RHS_ALLOC]]
// CHECK: %[[OUTPUT_ALLOC:.*]] = memref.alloc(%0) : memref<?xi16>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_ALLOC]], %[[RHS_ALLOC]] : memref<?xi16>, memref<?xi16>) outs(%[[OUTPUT_ALLOC]] : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = and %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

