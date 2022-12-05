// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --gml-tiling="tile-sizes=8 distribute=false op-label=sum" \
// RUN:     --gml-fusion="producer-label=mul consumer-label=sum" | \
// RUN: FileCheck %s

#id_2d = affine_map<(d0, d1) -> (d0, d1)>
#project_2d = affine_map<(d0, d1) -> (d0)>

func.func @reduce_cwise(%lhs: tensor<32x16xf32>, %rhs: tensor<32x16xf32>)
    -> tensor<32xf32> {
  %init_mul = tensor.empty() : tensor<32x16xf32>
  %mul = linalg.generic {
      indexing_maps = [#id_2d, #id_2d, #id_2d],
      iterator_types = ["parallel", "parallel"],
      op_label = "mul"}
      ins(%lhs, %rhs : tensor<32x16xf32>, tensor<32x16xf32>)
      outs(%init_mul : tensor<32x16xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %mul_scalar = arith.mulf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %mul_scalar : f32
  } -> tensor<32x16xf32>
  %c0_f32 = arith.constant 0.0 : f32
  %init_reduce = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%c0_f32 : f32)
      outs(%init_reduce : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.generic {
      indexing_maps = [#id_2d, #project_2d],
      iterator_types = ["parallel", "reduction"],
      op_label = "sum"}
      ins(%mul : tensor<32x16xf32>)
      outs(%fill : tensor<32xf32>) {
  ^bb0(%mul_scalar: f32, %acc_scalar: f32):
    %sum_scalar = arith.addf %mul_scalar, %acc_scalar : f32
    linalg.yield %sum_scalar : f32
  } -> tensor<32xf32>
  return %sum: tensor<32xf32>
}

// CHECK-LABEL: @reduce_cwise
// CHECK-SAME:  %[[ARG0:.*]]: tensor<32x16xf32>, %[[ARG1:.*]]: tensor<32x16xf32>

// CHECK:       %[[C8:.*]] = arith.constant 8
// CHECK:       %[[C0:.*]] = arith.constant 0
// CHECK:       %[[C32:.*]] = arith.constant 32
// CHECK:       %[[CST:.*]] = arith.constant 0.000000e+00
// CHECK:       %[[INIT:.*]] = tensor.empty()
// CHECK:       %[[INIT_0:.*]] = tensor.empty()
// CHECK:       %[[FILL:.*]] = linalg.fill
// CHECK-SAME:      ins(%[[CST]] : f32)
// CHECK-SAME:      outs(%[[INIT_0]] : tensor<32xf32>)
// CHECK:       %[[FOR:.*]] = gml_st.for (%[[ARG2:.*]]) = (%[[C0]])
// CHECK-SAME:      to (%[[C32]])
// CHECK-SAME:      step (%[[C8]])
// CHECK-SAME:      outs (%[[ARG3:.*]] = %[[FILL]]: tensor<32xf32>)
// CHECK:         %[[TILE:.*]] = gml_st.tile [%[[ARG2]], 0] [8, 16] [1, 1]
// CHECK:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK:         %[[TILE_0:.*]] = gml_st.tile [%[[ARG2]], 0] [8, 16] [1, 1]
// CHECK:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
// CHECK:         %[[TILE_1:.*]] = gml_st.tile [%[[ARG2]], 0] [8, 16] [1, 1]
// CHECK:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE_1]]]
// CHECK:         %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<8x16xf32>, tensor<8x16xf32>)
// CHECK-SAME:        outs(%[[MATERIALIZE_1]] : tensor<8x16xf32>)
// CHECK-SAME:        attrs =  {op_label = "mul"}
// CHECK:         ^bb0(%[[ARG4:.*]]: f32, %[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32):
// CHECK:           %[[MULF:.*]] = arith.mulf %[[ARG4]], %[[ARG5]]
// CHECK:           linalg.yield %[[MULF]]
// CHECK:         %[[TILE_2:.*]] = gml_st.tile [%[[ARG2]]] [8] [1]
// CHECK:         %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[ARG3]][%[[TILE_2]]]
// CHECK:         %[[GENERIC_0:.*]] = linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "reduction"]
// CHECK-SAME:        ins(%[[GENERIC]] : tensor<8x16xf32>)
// CHECK-SAME:        outs(%[[MATERIALIZE_2]] : tensor<8xf32>)
// CHECK-SAME:        attrs =  {op_label = "sum"}
// CHECK:         ^bb0(%[[ARG4_0:.*]]: f32, %[[ARG5_0:.*]]: f32):
// CHECK:           %[[ADDF:.*]] = arith.addf %[[ARG4_0]], %[[ARG5_0]]
// CHECK:           linalg.yield %[[ADDF]]
// CHECK:         gml_st.set_yield %[[GENERIC_0]] into %[[ARG3]][%[[TILE_2]]]
// CHECK:       return %[[FOR]]
