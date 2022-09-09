// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=true op-label=generic-2d" \
// RUN: --gml-tiling="tile-sizes=4,1 distribute=true op-label=generic-2d" | \
// RUN: FileCheck %s

#id2d = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add
// CHECK-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>
func.func @add(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // CHECK:     %[[INIT:.*]] = linalg.init_tensor

  // CHECK:     %[[OUTER_LOOP:.*]] = gml_st.parallel
  // CHECK:       %[[INIT_SUB:.*]] = gml_st.materialize

  // CHECK:       %[[INNER_LOOP:.*]] = gml_st.parallel
  // CHECK:     %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]]
  // CHECK:     %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]]
  // CHECK:     %[[INIT_SUB_SUB:.*]] = gml_st.materialize %[[INIT_SUB]]
  // CHECK:         %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:      ins(%[[LHS_SUB]], %[[RHS_SUB]]
  // CHECK-SAME:      outs(%[[INIT_SUB_SUB]]
  // CHECK:         gml_st.set_yield %[[GENERIC]] into %[[INIT_SUB]]
  // CHECK:       gml_st.set_yield %[[INNER_LOOP]] into %[[INIT]]
  // CHECK:     return %[[OUTER_LOOP]]
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
