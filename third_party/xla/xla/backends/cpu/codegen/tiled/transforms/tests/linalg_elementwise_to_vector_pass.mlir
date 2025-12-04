// RUN: fusion_compiler_opt %s -split-input-file \
// RUN: -xtile-cpu-linalg-elementwise-to-vector -fold-memref-alias-ops \
// RUN: | FileCheck %s

func.func @elementwise_add_to_vector(
    %arg0 : memref<8x1024xf32>,
    %arg1 : memref<8x1024xf32>,
    %arg2 : memref<8x1024xf32>) {
  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK: scf.for %[[IV0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
  // CHECK:   scf.for %[[IV1:.*]] = %[[C0]] to %[[C1024]] step %[[C8]] {
  // CHECK:     %[[LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[IV1]]],
  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>
  // CHECK:     %[[RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[IV1]]],
  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x1024xf32>, vector<8xf32>
  // CHECK:     %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8xf32>
  // CHECK:     vector.transfer_write %[[OUT]], %arg2[%[[IV0]], %[[IV1]]]
  // CHECK-SAME:  {in_bounds = [true]} : vector<8xf32>, memref<8x1024xf32>
  // CHECK:   }
  // CHECK: }
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%arg0, %arg1 : memref<8x1024xf32>, memref<8x1024xf32>)
    outs(%arg2 : memref<8x1024xf32>)
  return
}

//------

func.func @elementwise_add_to_vector_non_multiple_of_8(
    %arg0 : memref<8x100xf32>,
    %arg1 : memref<8x100xf32>,
    %arg2 : memref<8x100xf32>) {
  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C96:.*]] = arith.constant 96 : index
  // CHECK: scf.for %[[IV0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
  // CHECK:   scf.for %[[IV1:.*]] = %[[C0]] to %[[C96]] step %[[C8]] {
  // CHECK:     %[[LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[IV1]]],
  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x100xf32>, vector<8xf32>
  // CHECK:     %[[RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[IV1]]],
  // CHECK-SAME:  %[[MASK]] {in_bounds = [true]} : memref<8x100xf32>, vector<8xf32>
  // CHECK:     %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<8xf32>
  // CHECK:     vector.transfer_write %[[OUT]], %arg2[%[[IV0]], %[[IV1]]]
  // CHECK-SAME:  {in_bounds = [true]} : vector<8xf32>, memref<8x100xf32>
  // CHECK:   }
  // CHECK: %[[UNROLL_LHS:.*]] = vector.transfer_read %arg0[%[[IV0]], %[[C96]]], %[[MASK]]
  // CHECK-SAME: {in_bounds = [true]} : memref<8x100xf32>, vector<4xf32>
  // CHECK: %[[UNROLL_RHS:.*]] = vector.transfer_read %arg1[%[[IV0]], %[[C96]]], %[[MASK]]
  // CHECK-SAME: {in_bounds = [true]} : memref<8x100xf32>, vector<4xf32>
  // CHECK: %[[UNROLL_OUT:.*]] = arith.addf %[[UNROLL_LHS]], %[[UNROLL_RHS]] : vector<4xf32>
  // CHECK: vector.transfer_write %[[UNROLL_OUT]], %arg2[%[[IV0]], %[[C96]]]
  // CHECK-SAME: {in_bounds = [true]} : vector<4xf32>, memref<8x100xf32>
  // CHECK: }
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%arg0, %arg1 : memref<8x100xf32>, memref<8x100xf32>)
    outs(%arg2 : memref<8x100xf32>)
  return
}

//------

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fused(%arg0: memref<8x1024xf32>,
                 %arg1: memref<8x1024xf32>,
                 %arg2: memref<8x1024xf32>) {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK-NEXT: vector.transfer_read
  // CHECK-NEXT: vector.transfer_read
  // CHECK-NEXT: arith.mulf
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: vector.transfer_write
  linalg.generic
    {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<8x1024xf32>, memref<8x1024xf32>)
    outs(%arg2 : memref<8x1024xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %mul = arith.mulf %lhs, %rhs : f32
    %res = arith.addf %mul, %rhs : f32
    linalg.yield %res : f32
  }
  return
}

// -----

func.func @elementwise_add_to_vector_small_minor(
    %arg0 : memref<8x3xf32>,
    %arg1 : memref<8x3xf32>,
    %arg2 : memref<8x3xf32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%arg0, %arg1 : memref<8x3xf32>, memref<8x3xf32>)
    outs(%arg2 : memref<8x3xf32>)
  return
}
