// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=generic-2d" \
// RUN: --gml-tiling="tile-sizes=4,1 distribute=false op-label=generic-2d" | \
// RUN: FileCheck %s

#id2d = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add
// CHECK-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>
func.func @add(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // CHECK:      %[[C4:.*]] = arith.constant 4
  // CHECK:      %[[C1:.*]] = arith.constant 1
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[C256:.*]] = arith.constant 256
  // CHECK:      %[[C512:.*]] = arith.constant 512
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%{{.*}}, %{{.*}}]
  // CHECK:      %[[UB0:.*]] = tensor.dim %[[LHS]], %[[C0]]
  // CHECK:      %[[UB1:.*]] = tensor.dim %[[LHS]], %[[C1]]
  // CHECK:      %[[FOR:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
  // CHECK-SAME:     to (%[[UB0]], %[[UB1]])
  // CHECK-SAME:     step (%[[C256]], %[[C512]])
  // CHECK-SAME:     outs (%[[ACC:.*]] = %[[INIT]]: tensor<?x?xf32>)
  // CHECK:        %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]]
  // CHECK:        %[[ACC_SUB:.*]] = gml_st.materialize %[[ACC]]
  // CHECK:        %[[UB0_:.*]] = tensor.dim %[[LHS_SUB]], %[[C0]]
  // CHECK:        %[[UB1_:.*]] = tensor.dim %[[LHS_SUB]], %[[C1]]
  // CHECK:        %[[FOR_:.*]] = gml_st.for (%[[K:.*]], %[[L:.*]]) = (%[[C0]], %[[C0]])
  // CHECK-SAME:       to (%[[UB0_]], %[[UB1_]])
  // CHECK-SAME:       step (%[[C4]], %[[C1]])
  // CHECK-SAME:       outs (%[[ACC_:.*]] = %[[ACC_SUB]]: tensor<?x?xf32>)
  // CHECK:          %[[GENERIC:.*]] = linalg.generic
  // CHECK:          gml_st.set_yield %[[GENERIC]] into %[[ACC_]]
  // CHECK:        gml_st.set_yield %[[FOR_]] into %[[ACC]]
  // CHECK:      return %[[FOR]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [#id2d, #id2d, #id2d],
      iterator_types = ["parallel", "parallel"],
      op_label = "generic-2d" }
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  func.return %add : tensor<?x?xf32>
}
