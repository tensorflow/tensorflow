// RUN: fusion_compiler_opt %s -xtile-cpu-linalg-elementwise-to-vector -split-input-file | FileCheck %s

func.func @elementwise_add_to_vector(
    %arg0 : memref<8x1024xf32>,
    %arg1 : memref<8x1024xf32>,
    %arg2 : memref<8x1024xf32>) {
  // CHECK-DAG: %[[MASK:.*]] = ub.poison : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
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
