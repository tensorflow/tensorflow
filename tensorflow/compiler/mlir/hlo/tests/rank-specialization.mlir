// RUN: mlir-hlo-opt %s --split-input-file --mhlo-rank-specialization-cluster | FileCheck %s
// RUN: mlir-hlo-opt %s --split-input-file --mhlo-rank-specialization-cluster --mhlo-rank-specialization-to-scf | FileCheck %s --check-prefix CHECK-SCF

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

// CHECK-SCF-LABEL: @sqrt
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-SCF:       %[[TMP0:.*]] = "mhlo.sqrt"(%[[FLAT_ARG]]) : (tensor<?xf32>)
// CHECK-SCF:       %[[TMP1:.*]] = "mhlo.sqrt"(%[[TMP0]]) : (tensor<?xf32>)
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = "mhlo.sqrt"(%[[TMP1]]) : (tensor<?xf32>)
// CHECK-SCF:       %[[RES_SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[RES_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// Don't cluster ranked operations.
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
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
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

// CHECK-SCF-LABEL: @tan
// CHECK-SCF-SAME: (%[[ARG:.*]]: tensor<*xf32>)
// CHECK-SCF:      %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:      %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:      %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:      %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-SCF:      %[[TMP0:.*]] = chlo.tan %[[FLAT_ARG]] : tensor<?xf32>
// CHECK-SCF:      %[[TMP1:.*]] = chlo.tan %[[TMP0]] : tensor<?xf32>
// CHECK-SCF:      %[[UNSHAPED_RES:.*]] = chlo.tan %[[TMP1]] : tensor<?xf32>
// CHECK-SCF:      %[[RES_SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:      %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[RES_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:      return %[[RES]]

// -----

// Composition of unary/binary CHLO and unary MHLO ops.
// CHECK-LABEL: @mixed
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<*xf32>)
func @mixed(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>)
    -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG2]], %[[ARG1]], %[[ARG0]])
  // CHECK: ^bb0(%[[ARG2_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>, %[[ARG0_:.*]]: tensor<*xf32>)
  // CHECK:   %[[TMP0:.*]] = chlo.tan %[[ARG0_]]
  // CHECK:   %[[TMP1:.*]] = "mhlo.sqrt"(%[[ARG1_]])
  // CHECK:   %[[TMP2:.*]] = chlo.broadcast_multiply %[[TMP0]], %[[TMP1]]
  // CHECK:   %[[TMP3:.*]] = chlo.broadcast_add %[[TMP2]], %[[ARG2_]]
  // CHECK:   %[[TMP4:.*]] = "mhlo.sqrt"(%[[TMP3]])
  // CHECK:   %[[TMP5:.*]] = chlo.tan %[[TMP4]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP5]])
  // CHECK: return %[[RES]]
  %0 = chlo.tan %arg0 : tensor<*xf32> -> tensor<*xf32>
  %1 = "mhlo.sqrt"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = chlo.broadcast_multiply %0, %1
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = chlo.broadcast_add %2, %arg2
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %4 = "mhlo.sqrt"(%3) : (tensor<*xf32>) -> tensor<*xf32>
  %5 = chlo.tan %4 : tensor<*xf32> -> tensor<*xf32>
  return %5 : tensor<*xf32>
}

// -----

// Constant cluster operand.
// CHECK-LABEL: @relu
// CHECK-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
func @relu(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]], %[[C0]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>, %[[C0_:.*]]: tensor<f32>):
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_maximum %[[ARG_]], %[[C0_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = chlo.broadcast_maximum %0, %arg
      : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @relu
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
// CHECK-SCF:       %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = chlo.broadcast_maximum %[[FLAT_ARG]], %[[C0]] : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF:       %[[RES_SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[RES_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// Cluster with binary non-broadcasting operation.
// CHECK-LABEL: @angle
// CHECK-SAME:  (%[[ARG:.*]]: tensor<*xcomplex<f32>>)
func @angle(%arg : tensor<*xcomplex<f32>>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xcomplex<f32>>):
  // CHECK:   %[[IMAG:.*]] = "mhlo.imag"(%[[ARG_]])
  // CHECK:   %[[REAL:.*]] = "mhlo.real"(%[[ARG_]])
  // CHECK:   %[[TMP:.*]] = mhlo.atan2 %[[IMAG]], %[[REAL]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = "mhlo.imag"(%arg) : (tensor<*xcomplex<f32>>) -> tensor<*xf32>
  %1 = "mhlo.real"(%arg) : (tensor<*xcomplex<f32>>) -> tensor<*xf32>
  %2 = mhlo.atan2 %0, %1 : tensor<*xf32>
  return %2 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @angle
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xcomplex<f32>>)
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xcomplex<f32>>, tensor<1xindex>) -> tensor<?xcomplex<f32>>
// CHECK-SCF:       %[[IMAG:.*]] = "mhlo.imag"(%[[FLAT_ARG]]) : (tensor<?xcomplex<f32>>)
// CHECK-SCF:       %[[REAL:.*]] = "mhlo.real"(%[[FLAT_ARG]]) : (tensor<?xcomplex<f32>>)
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = mhlo.atan2 %[[IMAG]], %[[REAL]] : tensor<?xf32>
// CHECK-SCF:       %[[RES_SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[RES_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// CHECK-LABEL: @xlogy
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func @xlogy(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[C0]], %[[ARG0]], %[[ARG1]])
  // CHECK: ^bb0(%[[C0_:.*]]: tensor<f32>, %[[ARG0_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP0:.*]] = chlo.broadcast_compare %[[ARG0_]], %[[C0_]] {comparison_direction = "EQ"}
  // CHECK:   %[[TMP1:.*]] = "mhlo.log"(%[[ARG1_]])
  // CHECK:   %[[TMP2:.*]] = chlo.broadcast_multiply %[[ARG0_]], %[[TMP1]]
  // CHECK:   %[[TMP3:.*]] = chlo.broadcast_select %[[TMP0]], %[[C0_]], %[[TMP2]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP3]])
  // CHECK: return %[[RES]]
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = tensor.cast %0 : tensor<f32> to tensor<f32>
  %2 = chlo.broadcast_compare %arg0, %1 {comparison_direction = "EQ"}
      : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  %3 = "mhlo.log"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = chlo.broadcast_multiply %arg0, %3
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %5 = chlo.broadcast_select %2, %1, %4
      : (tensor<*xi1>, tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  return %5 : tensor<*xf32>
}
