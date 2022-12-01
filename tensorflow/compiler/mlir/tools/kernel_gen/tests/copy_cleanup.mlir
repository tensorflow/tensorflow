// RUN: kernel-gen-opt %s --copy-cleanup --split-input-file | FileCheck %s

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  func.func @Copy(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
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
      %and = arith.andi %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    func.return
  }
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @Copy(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[OUTPUT:.*]] = memref.alloc(%[[SIZE]]) : memref<?xi16>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_CASTED]], %[[RHS_CASTED]] : memref<?xi16, #[[$MAP0]]>, memref<?xi16, #[[$MAP0]]>) outs(%{{.*}} : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = arith.andi %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

// -----

// The target of the copy is also used to write.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  func.func @CopyWithWrite(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %size = memref.dim %rhs, %c0 : memref<?xi16>
    %lhsCasted = memref.reinterpret_cast %lhs to offset: [0], sizes: [%size], strides: [%c0] : memref<?xi16> to memref<?xi16, #map0>
    %lhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %lhsCasted, %lhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    %rhsCasted = memref.reinterpret_cast %rhs to offset: [0], sizes: [%size], strides: [%c1] : memref<?xi16> to memref<?xi16, #map0>
    %rhsAlloc = memref.alloc(%size) : memref<?xi16>
    memref.copy %rhsCasted, %rhsAlloc : memref<?xi16, #map0> to memref<?xi16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%lhsAlloc, %rhsAlloc : memref<?xi16>, memref<?xi16>) outs(%rhsAlloc : memref<?xi16>) {
    ^bb0(%arg1: i16, %arg2: i16, %arg3 : i16):
      %and = arith.andi %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    func.return
  }
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @CopyWithWrite(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?xi16>
// CHECK: memref.copy %[[RHS_CASTED]], %[[RHS_ALLOC]]
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_CASTED]], %[[RHS_ALLOC]] : memref<?xi16, #[[$MAP0]]>, memref<?xi16>) outs(%[[RHS_ALLOC]] : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = arith.andi %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

// -----

// The source of the copy is mutated.

#map0 = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0) -> (d0)>
builtin.module {
  func.func @CopyWithMutation(%lhs: memref<?xi16>, %rhs: memref<?xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c42 = arith.constant 42 : i16
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
      %and = arith.andi %arg1, %arg2 : i16
      linalg.yield %and : i16
    }
    func.return
  }
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @CopyWithMutation(
// CHECK-SAME:              %[[LHS:.*]]: memref<?xi16>, %[[RHS:.*]]: memref<?xi16>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[SIZE:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xi16>
// CHECK: %[[LHS_CASTED:.*]] = memref.reinterpret_cast %[[LHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C0]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[LHS_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?xi16>
// CHECK: memref.copy %[[LHS_CASTED]], %[[LHS_ALLOC]]
// CHECK: %[[RHS_CASTED:.*]] = memref.reinterpret_cast %[[RHS]] to offset: [0], sizes: [%[[SIZE]]], strides: [%[[C1]]] : memref<?xi16> to memref<?xi16, #[[$MAP0]]>
// CHECK: %[[RHS_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?xi16>
// CHECK: memref.copy %[[RHS_CASTED]], %[[RHS_ALLOC]]
// CHECK: %[[OUTPUT_ALLOC:.*]] = memref.alloc(%[[SIZE]]) : memref<?xi16>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%[[LHS_ALLOC]], %[[RHS_ALLOC]] : memref<?xi16>, memref<?xi16>) outs(%[[OUTPUT_ALLOC]] : memref<?xi16>) {
// CHECK: ^bb0(%[[ARG1:.*]]: i16, %[[ARG2:.*]]: i16, %[[ARG3:.*]]: i16):
// CHECK:   %[[AND:.*]] = arith.andi %[[ARG1]], %[[ARG2]] : i16
// CHECK:   linalg.yield %[[AND]] : i16

// -----

#map0 = affine_map<(d0) -> (d0)>
builtin.module  {
  func.func @testCopyAfterLinalg(%arg2: memref<4xi32>, %arg3: memref<4xi32>, %arg4: memref<4xi32>) {
    %0 = memref.alloc() : memref<4xi32>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg2, %arg3 : memref<4xi32>, memref<4xi32>) outs(%0 : memref<4xi32>) {
    ^bb0(%arg5: i32, %arg6: i32, %arg7: i32):
      %1 = arith.addi %arg5, %arg6 : i32
      linalg.yield %1 : i32
    }
    memref.copy %0, %arg4 : memref<4xi32> to memref<4xi32>
    func.return
  }
}

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:func @testCopyAfterLinalg(
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi32>, %[[ARG1:.*]]: memref<4xi32>, %[[ARG2:.*]]: memref<4xi32>) {
// CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : memref<4xi32>, memref<4xi32>) outs(%[[ARG2]] : memref<4xi32>) {
// CHECK: ^bb0(%[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32):
// CHECK:   %[[R1:.*]] = arith.addi %[[ARG3]], %[[ARG4]] : i32
// CHECK:   linalg.yield %[[R1]] : i32
// CHECK: }
// CHECK: return
// CHECK: }

// -----

#map0 = affine_map<(d0) -> (d0)>
builtin.module  {
  func.func @testCopyAfterLinalgMutated(%arg2: memref<4xi32>, %arg3: memref<4xi32>, %arg4: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %0 = memref.alloc() : memref<4xi32>
    linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%arg2, %arg3 : memref<4xi32>, memref<4xi32>) outs(%0 : memref<4xi32>) {
    ^bb0(%arg5: i32, %arg6: i32, %arg7: i32):
      %1 = arith.addi %arg5, %arg6 : i32
      linalg.yield %1 : i32
    }
    memref.copy %0, %arg4 : memref<4xi32> to memref<4xi32>
    memref.store %c42, %0[%c0] : memref<4xi32>
    func.return
  }
}

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:func @testCopyAfterLinalgMutated(
// CHECK-SAME: %[[ARG0:.*]]: memref<4xi32>, %[[ARG1:.*]]: memref<4xi32>, %[[ARG2:.*]]: memref<4xi32>) {
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi32>
// CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : memref<4xi32>, memref<4xi32>) outs(%[[ALLOC]] : memref<4xi32>) {
// CHECK: ^bb0(%[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32):
// CHECK:   %[[R1:.*]] = arith.addi %[[ARG3]], %[[ARG4]] : i32
// CHECK:   linalg.yield %[[R1]] : i32
// CHECK: }
// CHECK: memref.copy %[[ALLOC]], %[[ARG2]]
// CHECK: return
// CHECK: }
