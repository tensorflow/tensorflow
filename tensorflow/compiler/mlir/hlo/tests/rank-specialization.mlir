// RUN: mlir-hlo-opt %s --split-input-file --mhlo-rank-specialization-cluster | FileCheck %s

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

// -----

// Unary MHLO operation.
// CHECK-LABEL: @sqrt
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @sqrt(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP0:.*]] = "mhlo.sqrt"(%[[ARG_]])
  // CHECK:   %[[TMP1:.*]] = "mhlo.sqrt"(%[[TMP0]])
  // CHECK:   %[[TMP2:.*]] = "mhlo.sqrt"(%[[TMP1]])
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = "mhlo.sqrt"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "mhlo.sqrt"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// -----

// Don't cluster single ranked operation.
// CHECK-LABEL: @sqrt_ranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<3x?xf32>)
func @sqrt_ranked(%arg: tensor<3x?xf32>) -> tensor<3x?xf32> {
  // CHECK-NOT: rank_specialization_cluster
  %0 = "mhlo.sqrt"(%arg) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  %2 = "mhlo.sqrt"(%1) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  return %2 : tensor<3x?xf32>
}

// -----

// Ternary operation.
// CHECK-LABEL: @select_mixed
// CHECK-SAME: (%[[PRED:.*]]: tensor<*xi1>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<2xf32>)
func @select_mixed(%pred: tensor<*xi1>, %arg1: tensor<*xf32>,
    %arg2: tensor<2xf32>)  -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[PRED]], %[[ARG1]], %[[ARG2]])
  // CHECK: ^bb0(%[[PRED_:.*]]: tensor<*xi1>, %[[ARG1_:.*]]: tensor<*xf32>, %[[ARG2_:.*]]: tensor<2xf32>)
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_select %[[PRED_]], %[[ARG1_]], %[[ARG2_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = "chlo.broadcast_select"(%pred, %arg1, %arg2)
      : (tensor<*xi1>, tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Unary CHLO operation.
// CHECK-LABEL: @tan
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>) -> tensor<*xf32>
func @tan(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]]) ( {
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>)
  // CHECK:   %[[TMP0:.*]] = chlo.tan %[[ARG_]]
  // CHECK:   %[[TMP1:.*]] = chlo.tan %[[TMP0]]
  // CHECK:   %[[TMP2:.*]] = chlo.tan %[[TMP1]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = chlo.tan %arg : tensor<*xf32> -> tensor<*xf32>
  %1 = chlo.tan %0 : tensor<*xf32> -> tensor<*xf32>
  %2 = chlo.tan %1 : tensor<*xf32> -> tensor<*xf32>
  return %2 : tensor<*xf32>
}
