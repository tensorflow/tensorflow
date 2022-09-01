// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=8 distribute=false op-label="sum"" \
// RUN: --gml-fusion="producer-label="mul" consumer-label="sum"" \
// RUN: | FileCheck %s

#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#project_2d = affine_map<(d0, d1) -> (d0)>

func.func @reduce_cwise(%lhs: tensor<32x16xf32>, %rhs: tensor<32x16xf32>)
    -> tensor<32xf32> {
  %init_mul = linalg.init_tensor [32, 16] : tensor<32x16xf32>
  %mul = linalg.generic {
      indexing_maps = [#id_2d, #id_2d, #id_2d],
      iterator_types = ["parallel", "parallel"],
      op_label = "mul"}
      ins(%lhs, %rhs : tensor<32x16xf32>, tensor<32x16xf32>)
      outs(%init_mul : tensor<32x16xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %mul = arith.mulf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %mul : f32
  } -> tensor<32x16xf32>

  %c0_f32 = arith.constant 0.0 : f32
  %init_reduce = linalg.init_tensor [32] : tensor<32xf32>
  %fill = linalg.fill{op_label = "fill"}
    ins(%c0_f32 : f32) outs(%init_reduce : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.generic {
      indexing_maps = [#id_2d, #project_2d],
      iterator_types = ["parallel", "reduction"],
      op_label = "sum"}
      ins(%mul : tensor<32x16xf32>)
      outs(%fill : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sum = arith.addf %in, %out : f32
    linalg.yield %sum : f32
  } -> tensor<32xf32>
  return %sum: tensor<32xf32>
}
// CHECK-LABEL:   func.func @reduce_cwise(
// CHECK-SAME:     %[[LHS:.*]]: tensor<32x16xf32>,
// CHECK-SAME:     %[[RHS:.*]]: tensor<32x16xf32>)

// CHECK: linalg.fill {op_label = "fill"}

// CHECK: gml_st.for (%[[I:.*]]) =
// CHECK:      %[[TILE_SIZE:.*]] = affine.min
// CHECK:      %[[LHS_TILE:.*]] = gml_st.tile
// CHECK-SAME:   [%[[I]], 0] [%[[TILE_SIZE]], 16]
// CHECK:      %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[LHS_TILE]]]
// CHECK-SAME: tensor<32x16xf32>[!gml_st.tile<?x16>]

// CHECK:      %[[RHS_TILE:.*]] = gml_st.tile
// CHECK-SAME:  [%[[I]], 0] [%[[TILE_SIZE]], 16] [1, 1]
// CHECK-SAME:  !gml_st.tile<32x16> to !gml_st.tile<?x16>
// CHECK:      %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[RHS_TILE]]]
// CHECK-SAME:  tensor<32x16xf32>[!gml_st.tile<?x16>]

// CHECK:      %[[INIT_TILE:.*]] = gml_st.tile
// CHECK-SAME:   [%[[I]], 0] [%[[TILE_SIZE]], 16] [1, 1]
// CHECK-SAME:   : !gml_st.tile<32x16> to !gml_st.tile<?x16>
// CHECK:      %[[INIT_MUL:.*]] = gml_st.materialize
// CHECK-SAME:   [%[[INIT_TILE]]] : tensor<32x16xf32>[!gml_st.tile<?x16>]

// CHECK:      %[[MUL:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]]
// CHECK-SAME:   outs(%[[INIT_MUL]] : tensor<?x16xf32>)
// CHECK-SAME:   attrs = {op_label = "mul"}

// CHECK:      %[[FILL_TILE:.*]] = gml_st.tile
// CHECK-SAME:  [%[[I]]] [%[[TILE_SIZE]]] [1] : !gml_st.tile<32> to !gml_st.tile<?>
// CHECK:      %[[FILL_SUB:.*]] = gml_st.materialize
// CHECK-SAME:  [%[[FILL_TILE]]] : tensor<32xf32>[!gml_st.tile<?>]

// CHECK:      %[[REDUCE_SUB:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[VAL_31:.*]] : tensor<?x16xf32>)
// CHECK-SAME:   outs(%[[FILL_SUB]] : tensor<?xf32>) attrs = {op_label = "sum"}

// CHECK:      gml_st.set_yield %[[REDUCE_SUB:.*]] into %{{.*}}[%[[FILL_TILE]]]
// CHECK-SAME:  tensor<?xf32> into tensor<32xf32>[!gml_st.tile<?>]

