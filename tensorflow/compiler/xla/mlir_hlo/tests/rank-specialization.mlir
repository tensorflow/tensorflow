// RUN: mlir-hlo-opt %s --split-input-file --mhlo-rank-specialization-cluster | FileCheck %s
// RUN: mlir-hlo-opt %s --split-input-file --mhlo-rank-specialization-cluster --mhlo-rank-specialization-to-scf=max-target-rank=3 | FileCheck %s --check-prefix CHECK-SCF

// CHECK-LABEL: @add_mul
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<*xf32>)
func.func @add_mul(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>,
    %arg2 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG2]], %[[ARG0]], %[[ARG1]]) ({
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
  func.return %1 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @add_mul
// CHECK-SCF-SAME: (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<*xf32>)
// CHECK-SCF-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-SCF-DAG:  %[[C2:.*]] = arith.constant 2
// CHECK-SCF-DAG:  %[[C3:.*]] = arith.constant 3
// CHECK-SCF-DAG:  %[[ONE_SHAPE_1:.*]] = shape.const_shape [1]
// CHECK-SCF-DAG:  %[[ONE_SHAPE_2:.*]] = shape.const_shape [1, 1]
// CHECK-SCF-DAG:  %[[ONE_SHAPE_3:.*]] = shape.const_shape [1, 1, 1]
// CHECK-SCF-DAG:  %[[SHAPE_ARG0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG2:.*]] = shape.shape_of %[[ARG2]]
//                 Equal shapes case:
// CHECK-SCF-DAG:  %[[EQ20:.*]] = shape.shape_eq %[[SHAPE_ARG2]], %[[SHAPE_ARG0]]
// CHECK-SCF-DAG:  %[[EQ21:.*]] = shape.shape_eq %[[SHAPE_ARG2]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:  %[[SHAPES_EQ:.*]] = arith.andi %[[EQ20]], %[[EQ21]]
// CHECK-SCF:      %[[UNSHAPED_RES_EQ_SHAPES:.*]] = scf.if %[[SHAPES_EQ]]
// CHECK-SCF-DAG:    %[[ANY_SHAPE:.*]] = shape.any %[[SHAPE_ARG2]], %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[N:.*]] = shape.num_elements %[[ANY_SHAPE]]
// CHECK-SCF-DAG:    %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:    %[[FLAT_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:    %[[FLAT_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:    %[[FLAT_ARG2:.*]] = "mhlo.dynamic_reshape"(%[[ARG2]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:    %[[TMP:.*]] = chlo.broadcast_multiply %[[FLAT_ARG0]], %[[FLAT_ARG1]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:    %[[INNER_RES:.*]] = chlo.broadcast_add %[[TMP]], %[[FLAT_ARG2]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:    %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:        scf.yield %[[INNER_RES_]]
// CHECK-SCF:      else
//                 Find maximum reduced rank.
// CHECK-SCF-DAG:    %[[REDUCED_SHAPES:.*]]:3 = chlo.minimum_broadcast_shapes %[[SHAPE_ARG2]], %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[REDUCED_RANK0:.*]] = shape.rank %[[REDUCED_SHAPES]]#1
// CHECK-SCF-DAG:    %[[REDUCED_RANK1:.*]] = shape.rank %[[REDUCED_SHAPES]]#2
// CHECK-SCF-DAG:    %[[REDUCED_RANK2:.*]] = shape.rank %[[REDUCED_SHAPES]]#0
// CHECK-SCF-DAG:    %[[R2_GT_R0:.*]] = arith.cmpi sgt, %[[REDUCED_RANK2]], %[[REDUCED_RANK0]]
// CHECK-SCF-DAG:    %[[R20:.*]] = arith.select %[[R2_GT_R0]], %[[REDUCED_RANK2]], %[[REDUCED_RANK0]]
// CHECK-SCF-DAG:    %[[R20_GT_R1:.*]] = arith.cmpi sgt, %[[R20]], %[[REDUCED_RANK1]]
// CHECK-SCF-DAG:    %[[MAX_RED_RANK:.*]] = arith.select %[[R20_GT_R1]], %[[R20]], %[[REDUCED_RANK1]]
//                 Generic case 1:
// CHECK-SCF:        %[[MAX_RED_RANK_LE_1:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C1]]
// CHECK-SCF:        %[[UNSHAPED_RES_1:.*]] = scf.if %[[MAX_RED_RANK_LE_1]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#2, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG2:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:      %[[EXT_SHAPE_ARG2_:.*]] = tensor.cast %[[EXT_SHAPE_ARG2]]
// CHECK-SCF-DAG:      %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:      %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:      %[[REDUCED_ARG2:.*]] = "mhlo.dynamic_reshape"(%[[ARG2]], %[[EXT_SHAPE_ARG2_]])
// CHECK-SCF-DAG:      %[[TMP:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:      %[[INNER_RES:.*]] = chlo.broadcast_add %[[TMP]], %[[REDUCED_ARG2]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:      %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:          scf.yield %[[INNER_RES_]]
// CHECK-SCF:        else
//                 Generic case 2:
// CHECK-SCF:          %[[MAX_RED_RANK_LE_2:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C2]]
// CHECK-SCF:          %[[UNSHAPED_RES_2:.*]] = scf.if %[[MAX_RED_RANK_LE_2]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_2]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#2, %[[ONE_SHAPE_2]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG2:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_2]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG2_:.*]] = tensor.cast %[[EXT_SHAPE_ARG2]]
// CHECK-SCF-DAG:        %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:        %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:        %[[REDUCED_ARG2:.*]] = "mhlo.dynamic_reshape"(%[[ARG2]], %[[EXT_SHAPE_ARG2_]])
// CHECK-SCF-DAG:        %[[TMP:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES:.*]] = chlo.broadcast_add %[[TMP]], %[[REDUCED_ARG2]] : (tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:            scf.yield %[[INNER_RES_]]
// CHECK-SCF:          else
//                 Generic case 3:
// CHECK-SCF:            %[[MAX_RED_RANK_LE_3:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C3]]
// CHECK-SCF:            assert %[[MAX_RED_RANK_LE_3]], "Input for dynamic binary or n-ary op lowering was of a rank greater than 3"
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_3]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#2, %[[ONE_SHAPE_3]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG2:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_3]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[EXT_SHAPE_ARG2_:.*]] = tensor.cast %[[EXT_SHAPE_ARG2]]
// CHECK-SCF-DAG:        %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:        %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:        %[[REDUCED_ARG2:.*]] = "mhlo.dynamic_reshape"(%[[ARG2]], %[[EXT_SHAPE_ARG2_]])
// CHECK-SCF-DAG:        %[[TMP:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES:.*]] = chlo.broadcast_add %[[TMP]], %[[REDUCED_ARG2]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:            scf.yield %[[INNER_RES_]]
// CHECK-SCF:          scf.yield %[[UNSHAPED_RES_2]]
// CHECK-SCF:        scf.yield %[[UNSHAPED_RES_1]]
//                 Reshape the result.
// CHECK-SCF-DAG:  %[[SHAPE_ARG0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:  %[[TMP:.*]] = shape.broadcast %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG2:.*]] = shape.shape_of %[[ARG2]]
// CHECK-SCF-DAG:  %[[RES_SHAPE:.*]] = shape.broadcast %[[TMP]], %[[SHAPE_ARG2]]
// CHECK-SCF-DAG:  %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES_EQ_SHAPES]], %[[RES_SHAPE]])
// CHECK-SCF:      return %[[RES]]

// -----

// CHECK-LABEL: @compare_const_like
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>)
func.func @compare_const_like(%arg0 : tensor<*xf32>) -> tensor<*xi1> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG0]]) ({
  // CHECK: ^bb0(%[[ARG1:.*]]: tensor<*xf32>):
  // CHECK:   %[[ZERO:.*]] = "chlo.constant_like"(%[[ARG1]]) {value = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:   %[[CMP_GT:.*]] = chlo.broadcast_compare %[[ARG1]], %[[ZERO]] {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[CMP_GT]]) : (tensor<*xi1>) -> ()
  // CHECK: }) : (tensor<*xf32>) -> tensor<*xi1>
  // CHECK: return %[[RES]] : tensor<*xi1>
  %0 = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = chlo.broadcast_compare %arg0, %0 {comparison_direction = #mhlo<comparison_direction GT>}
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
  func.return %1 : tensor<*xi1>
}

// -----

// Unary MHLO operation.
// CHECK-LABEL: @sqrt
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @sqrt(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP0:.*]] = mhlo.sqrt %[[ARG_]]
  // CHECK:   %[[TMP1:.*]] = mhlo.sqrt %[[TMP0]]
  // CHECK:   %[[TMP2:.*]] = mhlo.sqrt %[[TMP1]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = mhlo.sqrt(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = mhlo.sqrt(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = mhlo.sqrt(%1) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @sqrt
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-SCF:       %[[TMP0:.*]] = mhlo.sqrt %[[FLAT_ARG]] : tensor<?xf32>
// CHECK-SCF:       %[[TMP1:.*]] = mhlo.sqrt %[[TMP0]] : tensor<?xf32>
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = mhlo.sqrt %[[TMP1]] : tensor<?xf32>
// CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// Don't cluster ranked operations.
// CHECK-LABEL: @sqrt_ranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<3x?xf32>)
func.func @sqrt_ranked(%arg: tensor<3x?xf32>) -> tensor<3x?xf32> {
  // CHECK-NOT: rank_specialization_cluster
  %0 = mhlo.sqrt(%arg) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  %1 = mhlo.sqrt(%0) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  %2 = mhlo.sqrt(%1) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  func.return %2 : tensor<3x?xf32>
}

// CHECK-SCF-LABEL: @sqrt_ranked
// CHECK-SCF-NOT:   dynamic_reshape
// CHECK-SCF:       return

// -----

// Operation with mixed ranked and unranked operands.
// CHECK-LABEL: @select_mixed
// CHECK-SAME: (%[[PRED:.*]]: tensor<*xi1>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<2xf32>)
func.func @select_mixed(%pred: tensor<*xi1>, %arg1: tensor<*xf32>,
    %arg2: tensor<2xf32>)  -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[PRED]], %[[ARG1]], %[[ARG2]])
  // CHECK: ^bb0(%[[PRED_:.*]]: tensor<*xi1>, %[[ARG1_:.*]]: tensor<*xf32>, %[[ARG2_:.*]]: tensor<2xf32>)
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_select %[[PRED_]], %[[ARG1_]], %[[ARG2_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = "chlo.broadcast_select"(%pred, %arg1, %arg2)
      : (tensor<*xi1>, tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @select_mixed
// CHECK-SCF:       chlo.broadcast_select %{{.*}}, %{{.*}}, %{{.*}} : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF:       return

// -----

// Unary CHLO operation.
// CHECK-LABEL: @tan
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @tan(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]]) ({
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>)
  // CHECK:   %[[TMP0:.*]] = chlo.tan %[[ARG_]]
  // CHECK:   %[[TMP1:.*]] = chlo.tan %[[TMP0]]
  // CHECK:   %[[TMP2:.*]] = chlo.tan %[[TMP1]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = chlo.tan %arg : tensor<*xf32> -> tensor<*xf32>
  %1 = chlo.tan %0 : tensor<*xf32> -> tensor<*xf32>
  %2 = chlo.tan %1 : tensor<*xf32> -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
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
// CHECK-SCF:      %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:      return %[[RES]]

// -----

// Composition of unary/binary CHLO and unary MHLO ops.
// CHECK-LABEL: @mixed
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>, %[[ARG2:.*]]: tensor<*xf32>)
func.func @mixed(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>)
    -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG2]], %[[ARG1]], %[[ARG0]])
  // CHECK: ^bb0(%[[ARG2_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>, %[[ARG0_:.*]]: tensor<*xf32>)
  // CHECK:   %[[TMP0:.*]] = chlo.tan %[[ARG0_]]
  // CHECK:   %[[TMP1:.*]] = mhlo.sqrt %[[ARG1_]]
  // CHECK:   %[[TMP2:.*]] = chlo.broadcast_multiply %[[TMP0]], %[[TMP1]]
  // CHECK:   %[[TMP3:.*]] = chlo.broadcast_add %[[TMP2]], %[[ARG2_]]
  // CHECK:   %[[TMP4:.*]] = mhlo.sqrt %[[TMP3]]
  // CHECK:   %[[TMP5:.*]] = chlo.tan %[[TMP4]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP5]])
  // CHECK: return %[[RES]]
  %0 = chlo.tan %arg0 : tensor<*xf32> -> tensor<*xf32>
  %1 = mhlo.sqrt(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
  %2 = chlo.broadcast_multiply %0, %1
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = chlo.broadcast_add %2, %arg2
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %4 = mhlo.sqrt(%3) : (tensor<*xf32>) -> tensor<*xf32>
  %5 = chlo.tan %4 : tensor<*xf32> -> tensor<*xf32>
  func.return %5 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @mixed
// CHECK-SCF-DAG:   %[[TMP0:.*]] = chlo.tan %{{.*}} : tensor<?xf32>
// CHECK-SCF-DAG:   %[[TMP1:.*]] = mhlo.sqrt %{{.*}} : tensor<?xf32>
// CHECK-SCF-DAG:   %[[TMP2:.*]] = chlo.broadcast_multiply %[[TMP0]], %[[TMP1]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:   %[[TMP3:.*]] = chlo.broadcast_add %[[TMP2]], %{{.*}} : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:   %[[TMP4:.*]] = mhlo.sqrt %[[TMP3]] : tensor<?xf32>
// CHECK-SCF:       chlo.tan %[[TMP4]] : tensor<?xf32>

// -----

// Constant cluster operand.
// CHECK-LABEL: @relu
// CHECK-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
func.func @relu(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]], %[[C0]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xf32>, %[[C0_:.*]]: tensor<f32>):
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_maximum %[[ARG_]], %[[C0_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = chlo.broadcast_maximum %0, %arg
      : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @relu
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xf32>)
// CHECK-SCF:       %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = chlo.broadcast_maximum %[[FLAT_ARG]], %[[C0]] : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// Cluster with binary non-broadcasting operation.
// CHECK-LABEL: @angle
// CHECK-SAME:  (%[[ARG:.*]]: tensor<*xcomplex<f32>>)
func.func @angle(%arg : tensor<*xcomplex<f32>>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG]])
  // CHECK: ^bb0(%[[ARG_:.*]]: tensor<*xcomplex<f32>>):
  // CHECK:   %[[IMAG:.*]] = mhlo.imag(%[[ARG_]])
  // CHECK:   %[[REAL:.*]] = mhlo.real(%[[ARG_]])
  // CHECK:   %[[TMP:.*]] = mhlo.atan2 %[[IMAG]], %[[REAL]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = mhlo.imag(%arg) : (tensor<*xcomplex<f32>>) -> tensor<*xf32>
  %1 = mhlo.real(%arg) : (tensor<*xcomplex<f32>>) -> tensor<*xf32>
  %2 = mhlo.atan2 %0, %1 : tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @angle
// CHECK-SCF-SAME:  (%[[ARG:.*]]: tensor<*xcomplex<f32>>)
// CHECK-SCF:       %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
// CHECK-SCF:       %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF:       %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF:       %[[FLAT_ARG:.*]] = "mhlo.dynamic_reshape"(%[[ARG]], %[[FLAT_SHAPE]]) : (tensor<*xcomplex<f32>>, tensor<1xindex>) -> tensor<?xcomplex<f32>>
// CHECK-SCF:       %[[IMAG:.*]] = mhlo.imag(%[[FLAT_ARG]]) : (tensor<?xcomplex<f32>>)
// CHECK-SCF:       %[[REAL:.*]] = mhlo.real(%[[FLAT_ARG]]) : (tensor<?xcomplex<f32>>)
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = mhlo.atan2 %[[IMAG]], %[[REAL]] : tensor<?xf32>
  // CHECK-SCF:       %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:       return %[[RES]]

// -----

// Scalar cluster operand.
// CHECK-LABEL: @xlogy
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @xlogy(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[C0]], %[[ARG0]], %[[ARG1]])
  // CHECK: ^bb0(%[[C0_:.*]]: tensor<f32>, %[[ARG0_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP0:.*]] = chlo.broadcast_compare %[[ARG0_]], %[[C0_]] {comparison_direction = #mhlo<comparison_direction EQ>}
  // CHECK:   %[[TMP1:.*]] = mhlo.log %[[ARG1_]]
  // CHECK:   %[[TMP2:.*]] = chlo.broadcast_multiply %[[ARG0_]], %[[TMP1]]
  // CHECK:   %[[TMP3:.*]] = chlo.broadcast_select %[[TMP0]], %[[C0_]], %[[TMP2]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP3]])
  // CHECK: return %[[RES]]
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = tensor.cast %0 : tensor<f32> to tensor<f32>
  %2 = chlo.broadcast_compare %arg0, %1 {comparison_direction = #mhlo<comparison_direction EQ>}
      : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  %3 = mhlo.log(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = chlo.broadcast_multiply %arg0, %3
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %5 = chlo.broadcast_select %2, %1, %4
      : (tensor<*xi1>, tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %5 : tensor<*xf32>
}

// CHECK-SCF:      @xlogy
// CHECK-SCF-SAME: (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
// CHECK-SCF-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-SCF-DAG:  %[[ONE_SHAPE_1:.*]] = shape.const_shape [1]
// CHECK-SCF-DAG:  %[[SHAPE_ARG0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:  %[[ZERO:.*]] = mhlo.constant dense<0.00{{.*}}>
//                 Lhs scalar case:
// CHECK-SCF-DAG:  %[[LHS_N:.*]] = shape.num_elements %[[SHAPE_ARG0]]
// CHECK-SCF-DAG:  %[[LHS_SCALAR:.*]] = arith.cmpi eq, %[[LHS_N]], %[[C1]]
// CHECK-SCF:      %[[UNSHAPED_RES:.*]] = scf.if %[[LHS_SCALAR]]
// CHECK-SCF-DAG:    %[[N:.*]] = shape.num_elements %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:    %[[FLAT_NON_SCALAR:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:    %[[SCALAR:.*]] = "mhlo.reshape"(%[[ARG0]])
// CHECK-SCF-DAG:    %[[PRED:.*]] = chlo.broadcast_compare %[[SCALAR]], %[[ZERO]] {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>)
// CHECK-SCF-DAG:    %[[TMP0:.*]] = mhlo.log %[[FLAT_NON_SCALAR]] : tensor<?xf32>
// CHECK-SCF-DAG:    %[[TMP1:.*]] = chlo.broadcast_multiply %[[SCALAR]], %[[TMP0]] : (tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:    %[[INNER_RES:.*]] = chlo.broadcast_select %[[PRED]], %[[ZERO]], %[[TMP1]] : (tensor<i1>, tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:    %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:        scf.yield %[[INNER_RES_]]
// CHECK-SCF:      else
//                   Rhs scalar case:
// CHECK-SCF-DAG:    %[[RHS_N:.*]] = shape.num_elements %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[RHS_SCALAR:.*]] = arith.cmpi eq, %[[RHS_N]], %[[C1]]
// CHECK-SCF:        %{{.*}} = scf.if %[[RHS_SCALAR]]
// CHECK-SCF-DAG:      %[[N:.*]] = shape.num_elements %[[SHAPE_ARG0]]
// CHECK-SCF-DAG:      %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:      %[[FLAT_NON_SCALAR:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:      %[[SCALAR:.*]] = "mhlo.reshape"(%[[ARG1]])
// CHECK-SCF-DAG:      %[[PRED:.*]] = chlo.broadcast_compare %[[FLAT_NON_SCALAR]], %[[ZERO]] {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF-DAG:      %[[TMP0:.*]] = mhlo.log %[[SCALAR]] : tensor<f32>
// CHECK-SCF-DAG:      %[[TMP1:.*]] = chlo.broadcast_multiply %[[FLAT_NON_SCALAR]], %[[TMP0]] : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF-DAG:      %[[INNER_RES:.*]] = chlo.broadcast_select %[[PRED]], %[[ZERO]], %[[TMP1]] : (tensor<?xi1>, tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:      %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:          scf.yield %[[INNER_RES_]]
// CHECK-SCF:        else
//                 Equal shapes case:
// CHECK-SCF-DAG:      %[[SHAPES_EQ:.*]] = shape.shape_eq %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF:          %{{.*}} = scf.if %[[SHAPES_EQ]]
// CHECK-SCF-DAG:        %[[SHAPE:.*]] = shape.any %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF-DAG:        %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:        %[[FLAT_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:        %[[FLAT_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:        %[[PRED:.*]] = chlo.broadcast_compare %[[FLAT_ARG0]], %[[ZERO]] {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF-DAG:        %[[TMP0:.*]] = mhlo.log %[[FLAT_ARG1]] : tensor<?xf32>
// CHECK-SCF-DAG:        %[[TMP1:.*]] = chlo.broadcast_multiply %[[FLAT_ARG0]], %[[TMP0]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES:.*]] = chlo.broadcast_select %[[PRED]], %[[ZERO]], %[[TMP1]] : (tensor<?xi1>, tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:            scf.yield %[[INNER_RES_]]
// CHECK-SCF:          else
//                 Find maximum reduced rank.
// CHECK-SCF-DAG:        %[[REDUCED_SHAPES:.*]]:2 = chlo.minimum_broadcast_shapes %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[REDUCED_RANK0:.*]] = shape.rank %[[REDUCED_SHAPES]]#0
// CHECK-SCF-DAG:        %[[REDUCED_RANK1:.*]] = shape.rank %[[REDUCED_SHAPES]]#1
// CHECK-SCF-DAG:        %[[R0_GT_R1:.*]] = arith.cmpi sgt, %[[REDUCED_RANK0]], %[[REDUCED_RANK1]]
// CHECK-SCF-DAG:        %[[MAX_RED_RANK:.*]] = arith.select %[[R0_GT_R1]], %[[REDUCED_RANK0]], %[[REDUCED_RANK1]]
//                 Generic case 1:
// CHECK-SCF:            %[[MAX_RED_RANK_LE_1:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C1]]
// CHECK-SCF:            %{{.*}} = scf.if %[[MAX_RED_RANK_LE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:          %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:          %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:          %[[PRED:.*]] = chlo.broadcast_compare %[[REDUCED_ARG0]], %[[ZERO]] {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF-DAG:          %[[TMP0:.*]] = mhlo.log %[[REDUCED_ARG1]] : tensor<?xf32>
// CHECK-SCF-DAG:          %[[TMP1:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[TMP0]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:          %[[INNER_RES:.*]] = chlo.broadcast_select %[[PRED]], %[[ZERO]], %[[TMP1]] : (tensor<?xi1>, tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:          %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:              scf.yield %[[INNER_RES_]]
// CHECK-SCF:            else
//                         ...
//                 Reshape the result.
// CHECK-SCF:      %[[S0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF:      %[[S0_:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF:      %[[S1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF:      %[[TMP:.*]] = shape.broadcast %[[S0_]], %[[S1]]
// CHECK-SCF:      %[[RES_SHAPE:.*]] = shape.broadcast %[[S0]], %[[TMP]]
// CHECK-SCF:      %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[RES_SHAPE]]) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-SCF:      return %[[RES]]

// -----

// CHECK-LABEL: @mul
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @mul(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG0]], %[[ARG1]])
  // CHECK: ^bb0(%[[ARG0_:.*]]: tensor<*xf32>, %[[ARG1_:.*]]: tensor<*xf32>):
  // CHECK:   %[[TMP:.*]] = chlo.broadcast_multiply %[[ARG0_]], %[[ARG1_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP]])
  // CHECK: return %[[RES]]
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @mul
// CHECK-SCF-SAME: (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
// CHECK-SCF-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-SCF-DAG:  %[[C2:.*]] = arith.constant 2
// CHECK-SCF-DAG:  %[[C3:.*]] = arith.constant 3
// CHECK-SCF-DAG:  %[[ONE_SHAPE_1:.*]] = shape.const_shape [1]
// CHECK-SCF-DAG:  %[[ONE_SHAPE_2:.*]] = shape.const_shape [1, 1]
// CHECK-SCF-DAG:  %[[ONE_SHAPE_3:.*]] = shape.const_shape [1, 1, 1]
// CHECK-SCF-DAG:  %[[SHAPE_ARG0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG1:.*]] = shape.shape_of %[[ARG1]]
//                 Lhs scalar case:
// CHECK-SCF-DAG:  %[[LHS_N:.*]] = shape.num_elements %[[SHAPE_ARG0]]
// CHECK-SCF-DAG:  %[[LHS_SCALAR:.*]] = arith.cmpi eq, %[[LHS_N]], %[[C1]]
// CHECK-SCF:      %[[UNSHAPED_RES_LHS_SCALAR:.*]] = scf.if %[[LHS_SCALAR]]
// CHECK-SCF-DAG:    %[[N:.*]] = shape.num_elements %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:    %[[FLAT_NON_SCALAR:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:    %[[SCALAR:.*]] = "mhlo.reshape"(%[[ARG0]])
// CHECK-SCF-DAG:    %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[SCALAR]], %[[FLAT_NON_SCALAR]] : (tensor<f32>, tensor<?xf32>)
// CHECK-SCF-DAG:    %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:        scf.yield %[[INNER_RES_]]
// CHECK-SCF:      else
//                   Rhs scalar case:
// CHECK-SCF-DAG:    %[[RHS_N:.*]] = shape.num_elements %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:    %[[RHS_SCALAR:.*]] = arith.cmpi eq, %[[RHS_N]], %[[C1]]
// CHECK-SCF:        %[[UNSHAPED_RES_RHS_SCALAR:.*]] = scf.if %[[RHS_SCALAR]]
// CHECK-SCF-DAG:      %[[N:.*]] = shape.num_elements %[[SHAPE_ARG0]]
// CHECK-SCF-DAG:      %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:      %[[FLAT_NON_SCALAR:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:      %[[SCALAR:.*]] = "mhlo.reshape"(%[[ARG1]])
// CHECK-SCF-DAG:      %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[FLAT_NON_SCALAR]], %[[SCALAR]] : (tensor<?xf32>, tensor<f32>)
// CHECK-SCF-DAG:      %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:          scf.yield %[[INNER_RES_]]
// CHECK-SCF:        else
//                 Equal shapes case:
// CHECK-SCF-DAG:      %[[SHAPES_EQ:.*]] = shape.shape_eq %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF:          %[[UNSHAPED_RES_EQ_SHAPES:.*]] = scf.if %[[SHAPES_EQ]]
// CHECK-SCF-DAG:        %[[SHAPE:.*]] = shape.any %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[N:.*]] = shape.num_elements %[[SHAPE]]
// CHECK-SCF-DAG:        %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:        %[[FLAT_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:        %[[FLAT_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:        %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[FLAT_ARG0]], %[[FLAT_ARG1]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:        %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:            scf.yield %[[INNER_RES_]]
// CHECK-SCF:          else
//                 Find maximum reduced rank.
// CHECK-SCF-DAG:        %[[REDUCED_SHAPES:.*]]:2 = chlo.minimum_broadcast_shapes %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:        %[[REDUCED_RANK0:.*]] = shape.rank %[[REDUCED_SHAPES]]#0
// CHECK-SCF-DAG:        %[[REDUCED_RANK1:.*]] = shape.rank %[[REDUCED_SHAPES]]#1
// CHECK-SCF-DAG:        %[[R0_GT_R1:.*]] = arith.cmpi sgt, %[[REDUCED_RANK0]], %[[REDUCED_RANK1]]
// CHECK-SCF-DAG:        %[[MAX_RED_RANK:.*]] = arith.select %[[R0_GT_R1]], %[[REDUCED_RANK0]], %[[REDUCED_RANK1]]
//                 Generic case 1:
// CHECK-SCF:            %[[MAX_RED_RANK_LE_1:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C1]]
// CHECK-SCF:            %[[UNSHAPED_RES_1:.*]] = scf.if %[[MAX_RED_RANK_LE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_1]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:          %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:          %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:          %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:          %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?xf32>, tensor<?xf32>)
// CHECK-SCF-DAG:          %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:              scf.yield %[[INNER_RES_]]
// CHECK-SCF:            else
//                 Generic case 2:
// CHECK-SCF:              %[[MAX_RED_RANK_LE_2:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C2]]
// CHECK-SCF:              %[[UNSHAPED_RES_2:.*]] = scf.if %[[MAX_RED_RANK_LE_2]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_2]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_2]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:            %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:            %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:            %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SCF-DAG:            %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:                scf.yield %[[INNER_RES_]]
// CHECK-SCF:              else
//                 Generic case 3:
// CHECK-SCF:                %[[MAX_RED_RANK_LE_3:.*]] = arith.cmpi ule, %[[MAX_RED_RANK]], %[[C3]]
// CHECK-SCF:                assert %[[MAX_RED_RANK_LE_3]], "Input for dynamic binary or n-ary op lowering was of a rank greater than 3"
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG0:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#0, %[[ONE_SHAPE_3]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG1:.*]] = shape.broadcast %[[REDUCED_SHAPES]]#1, %[[ONE_SHAPE_3]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG0_:.*]] = tensor.cast %[[EXT_SHAPE_ARG0]]
// CHECK-SCF-DAG:            %[[EXT_SHAPE_ARG1_:.*]] = tensor.cast %[[EXT_SHAPE_ARG1]]
// CHECK-SCF-DAG:            %[[REDUCED_ARG0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[EXT_SHAPE_ARG0_]])
// CHECK-SCF-DAG:            %[[REDUCED_ARG1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[EXT_SHAPE_ARG1_]])
// CHECK-SCF-DAG:            %[[INNER_RES:.*]] = chlo.broadcast_multiply %[[REDUCED_ARG0]], %[[REDUCED_ARG1]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SCF-DAG:            %[[INNER_RES_:.*]] = tensor.cast %[[INNER_RES]]
// CHECK-SCF:                scf.yield %[[INNER_RES_]]
// CHECK-SCF:              scf.yield %[[UNSHAPED_RES_2]]
// CHECK-SCF:            scf.yield %[[UNSHAPED_RES_1]]
// CHECK-SCF:          scf.yield %[[UNSHAPED_RES_EQ_SHAPES]]
// CHECK-SCF:        scf.yield %[[UNSHAPED_RES_RHS_SCALAR]]
//                 Reshape the result.
// CHECK-SCF-DAG:  %[[SHAPE_ARG0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:  %[[SHAPE_ARG1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:  %[[RES_SHAPE:.*]] = shape.broadcast %[[SHAPE_ARG0]], %[[SHAPE_ARG1]]
// CHECK-SCF-DAG:  %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES_LHS_SCALAR]], %[[RES_SHAPE]])
// CHECK-SCF:      return %[[RES]]

// -----

// CHECK-LABEL: @merge_clusters
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf64>, %[[ARG1:.*]]: tensor<*xf64>)
func.func @merge_clusters(%arg0: tensor<*xf64>, %arg1 : tensor<*xf64>)
    -> tensor<*xf64> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG0]], %[[ARG1]])
  // CHECK: ^bb0(%[[ARG0_:.*]]: tensor<*xf64>, %[[ARG1_:.*]]: tensor<*xf64>):
  // CHECK:   %[[TMP0:.*]] = mhlo.tanh %[[ARG0_]]
  // CHECK:   %[[TMP1:.*]] = chlo.broadcast_add %[[TMP0]], %[[ARG0_]]
  // CHECK:   %[[TMP2:.*]] = chlo.broadcast_add %[[TMP1]], %[[ARG1_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = "chlo.rank_specialization_cluster"(%arg0) ({
  ^bb0(%arg0_: tensor<*xf64>):
    %1 = mhlo.tanh(%arg0_) : (tensor<*xf64>) -> tensor<*xf64>
    "chlo.rank_specialization_cluster_yield"(%1) : (tensor<*xf64>) -> ()
  }) : (tensor<*xf64>) -> (tensor<*xf64>)
  %2 = "chlo.rank_specialization_cluster"(%0, %arg0, %arg1) ({
  ^bb0(%3: tensor<*xf64>, %4: tensor<*xf64>, %5: tensor<*xf64>):
    %6 = "chlo.broadcast_add"(%3, %4)
        : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %7 = "chlo.broadcast_add"(%6, %5)
        : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "chlo.rank_specialization_cluster_yield"(%7) : (tensor<*xf64>) -> ()
  }) : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> (tensor<*xf64>)
  func.return %2 : tensor<*xf64>
}

// -----

// CHECK-LABEL: @all_equal_shapes_inferrable
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf64>, %[[ARG1:.*]]: tensor<*xf64>)
func.func @all_equal_shapes_inferrable(%arg0: tensor<*xf64>, %arg1 : tensor<*xf64>)
    -> tensor<*xf64> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG0]], %[[ARG1]])
  // CHECK: ^bb0(%[[ARG0_:.*]]: tensor<*xf64>, %[[ARG1_:.*]]: tensor<*xf64>)
  // CHECK:   %[[INNER_RES:.*]] = mhlo.add %[[ARG0_]], %[[ARG1_]]
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[INNER_RES]])
  // CHECK: return %[[RES]]
  %0 = "mhlo.add"(%arg0, %arg1)
      : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  func.return %0 : tensor<*xf64>
}

// CHECK-SCF-LABEL: @all_equal_shapes_inferrable
// CHECK-SCF-SAME:  (%[[ARG0:.*]]: tensor<*xf64>, %[[ARG1:.*]]: tensor<*xf64>)
// CHECK-SCF-DAG:   %[[S0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:   %[[S1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:   %[[S:.*]] = shape.any %[[S0]], %[[S1]]
// CHECK-SCF-DAG:   %[[N:.*]] = shape.num_elements %[[S]]
// CHECK-SCF-DAG:   %[[FLAT_S:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:   %[[FLAT0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_S]])
// CHECK-SCF-DAG:   %[[FLAT1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_S]])
// CHECK-SCF:       %[[FLAT_RES:.*]] = mhlo.add %[[FLAT0]], %[[FLAT1]]
// CHECK-SCF-DAG:   %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[FLAT_RES]], %[[S0]])
// CHECK-SCF:       return %[[RES]]

// -----

// All shapes are equal, which is inferrable through the select op.
// CHECK-LABEL: @relu_grad
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @relu_grad(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG1]], %[[ARG0]])
  // CHECK: ^bb0(%[[ARG1_:.*]]: tensor<*xf32>, %[[ARG0_:.*]]: tensor<*xf32>)
  // CHECK:   %[[TMP0:.*]] = "chlo.constant_like"(%[[ARG0_]]) {value = 0.0{{.*}}e+00 : f32}
  // CHECK:   %[[TMP1:.*]] = mhlo.compare GT, %[[ARG0_]], %[[TMP0]]
  // CHECK:   %[[TMP2:.*]] = "mhlo.select"(%[[TMP1]], %[[ARG1_]], %[[TMP0]])
  // CHECK:   "chlo.rank_specialization_cluster_yield"(%[[TMP2]])
  // CHECK: return %[[RES]]
  %0 = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "mhlo.compare"(%arg0, %0) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
  %2 = "mhlo.select"(%1, %arg1, %0) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @relu_grad
// CHECK-SCF-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
// CHECK-SCF-DAG:   %[[S0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:   %[[S1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:   %[[S:.*]] = shape.any %[[S1]], %[[S0]]
// CHECK-SCF-DAG:   %[[N:.*]] = shape.num_elements %[[S]]
// CHECK-SCF-DAG:   %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:   %[[FLAT0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:   %[[FLAT1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:   %[[ZERO:.*]] = "chlo.constant_like"(%[[FLAT0]]) {value = 0.0{{.*}}+00 : f32}
// CHECK-SCF-DAG:   %[[PRED:.*]] = mhlo.compare GT, %[[FLAT0]], %[[ZERO]]
// CHECK-SCF:       %[[UNSHAPED_RES:.*]] = "mhlo.select"(%[[PRED]], %[[FLAT1]], %[[ZERO]])
// CHECK-SCF-DAG:   %[[RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[S1]])
// CHECK-SCF:       return %[[RES]]

// -----

// Find shape equivalences through surrounding constraints.
// CHECK-LABEL: @relu_grad
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @relu_grad(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[CSTR_EQ:.*]] = shape.cstr_eq %[[S0]], %[[S1]]
  // CHECK:     %[[RES:.*]] = shape.assuming %[[CSTR_EQ]]
  // CHECK:       %[[INNER_RES:.*]] = "chlo.rank_specialization_cluster"(%[[ARG1]], %[[ARG0]])
  // CHECK:       ^bb0(%[[ARG1_:.*]]: tensor<*xf32>, %[[ARG0_:.*]]: tensor<*xf32>):
  // CHECK-DAG:     %[[ZERO:.*]] = "chlo.constant_like"(%[[ARG0_]]) {value = 0.0{{.*}}+00 : f32}
  // CHECK-DAG:     %[[PRED:.*]] = mhlo.compare GT, %[[ARG0_]], %[[ZERO]]
  // CHECK-DAG:     %[[INNER_INNER_RES:.*]] = "mhlo.select"(%[[PRED]], %[[ARG1_]], %[[ZERO]])
  // CHECK:         "chlo.rank_specialization_cluster_yield"(%[[INNER_INNER_RES]])
  // CHECK:       shape.assuming_yield %[[INNER_RES]]
  // CHECK:     return %[[RES]]
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = shape.shape_of %arg1 : tensor<*xf32> -> tensor<?xindex>
  %2 = shape.cstr_eq %0, %1 : tensor<?xindex>, tensor<?xindex>
  %3 = shape.assuming %2 -> tensor<*xf32> {
    %4 = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32}
        : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "mhlo.compare"(%arg0, %4) {comparison_direction = #mhlo<comparison_direction GT>}
        : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
    %6 = "mhlo.select"(%5, %arg1, %4)
        : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    shape.assuming_yield %6 : tensor<*xf32>
  }
  func.return %3 : tensor<*xf32>
}

// CHECK-SCF-LABEL: @relu_grad
// CHECK-SCF-SAME:  (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
// CHECK-SCF-DAG:   %[[S0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:   %[[S1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:   %[[CSTR_EQ:.*]] = shape.cstr_eq %0, %1
// CHECK-SCF:       %[[RES:.*]] = shape.assuming %[[CSTR_EQ]]
// CHECK-SCF-DAG:     %[[S0:.*]] = shape.shape_of %[[ARG0]]
// CHECK-SCF-DAG:     %[[S1:.*]] = shape.shape_of %[[ARG1]]
// CHECK-SCF-DAG:     %[[S:.*]] = shape.any %[[S1]], %[[S0]]
// CHECK-SCF-DAG:     %[[N:.*]] = shape.num_elements %[[S]]
// CHECK-SCF-DAG:     %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[N]]
// CHECK-SCF-DAG:     %[[FLAT0:.*]] = "mhlo.dynamic_reshape"(%[[ARG0]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:     %[[FLAT1:.*]] = "mhlo.dynamic_reshape"(%[[ARG1]], %[[FLAT_SHAPE]])
// CHECK-SCF-DAG:     %[[ZERO:.*]] = "chlo.constant_like"(%[[FLAT0]]) {value = 0.0{{.*}}+00 : f32}
// CHECK-SCF-DAG:     %[[PRED:.*]] = mhlo.compare GT, %[[FLAT0]], %[[ZERO]]
// CHECK-SCF:         %[[UNSHAPED_RES:.*]] = "mhlo.select"(%[[PRED]], %[[FLAT1]], %[[ZERO]])
// CHECK-SCF-DAG:     %[[INNER_RES:.*]] = "mhlo.dynamic_reshape"(%[[UNSHAPED_RES]], %[[S1]])
// CHECK-SCF:         shape.assuming_yield %[[INNER_RES]]
// CHECK-SCF:       return %[[RES]]
