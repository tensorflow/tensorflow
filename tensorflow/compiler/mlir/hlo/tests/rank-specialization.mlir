// RUN: mlir-hlo-opt %s --mhlo-rank-specialization-cluster | FileCheck %s

// CHECK-LABEL: @add_mul
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<*xf32>)
func @add_mul(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG2]], %[[ARG0]], %[[ARG1]]) ( {
  // CHECK: ^bb0(%[[ARG2_:.*]]: tensor<*xf32>, %[[ARG0_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_multiply %[[ARG0_]], %[[ARG1_]]
  // CHECK:   %[[INNER_RES:.*]] = chlo.broadcast_add %[[TMP]], %[[ARG2_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[INNER_RES]])
  // CHECK: }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[RES]]
  %0 = chlo.broadcast_multiply %arg0, %arg1
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %1 = chlo.broadcast_add %0, %arg2
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}
