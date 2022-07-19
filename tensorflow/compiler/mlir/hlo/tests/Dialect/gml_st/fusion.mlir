// RUN: mlir-hlo-opt %s --split-input-file --gml-fusion | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim_at_tile
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>, %[[TILE:.*]]: !gml_st.tile<3x4x?>
func.func @dynamic_broadcast_in_dim_at_tile(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>, %tile : !gml_st.tile<3x4x?>)
    -> tensor<3x4x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[C2:.*]] = arith.constant 2
  // CHECK-DAG:  %[[S0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG:  %[[S1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG:  %[[S2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [%[[S0]], %[[S1]], %[[S2]]]
  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK-DAG:  %[[D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK-DAG:  %[[ISPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
  // CHECK-DAG:  %[[CED_TILE:.*]] = gml_st.drop_dims %[[TILE]], [0, 2]
  // CHECK-DAG:  %[[IS_D0_EXPANDING:.*]] = arith.cmpi ne, %[[D0]], %[[S0]]
  // CHECK-DAG:  %[[CED_TILE_OFFSET0:.*]] = gml_st.offset %[[CED_TILE]][%[[C0]]]
  // CHECK-DAG:  %[[CED_TILE_SIZE0:.*]] = gml_st.size %[[CED_TILE]][%[[C0]]]
  // CHECK-DAG:  %[[ARG_TILE_OFFSET0:.*]] = arith.select %[[IS_D0_EXPANDING]], %[[C0]], %[[CED_TILE_OFFSET0]]
  // CHECK-DAG:  %[[ARG_TILE_SIZE0:.*]] = arith.select %[[IS_D0_EXPANDING]], %[[C1]], %[[CED_TILE_SIZE0]]
  // CHECK-DAG:  %[[IS_D1_EXPANDING:.*]] = arith.cmpi ne, %[[D1]], %[[S2]]
  // CHECK-DAG:  %[[CED_TILE_OFFSET1:.*]] = gml_st.offset %[[CED_TILE]][%[[C1]]]
  // CHECK-DAG:  %[[CED_TILE_SIZE1:.*]] = gml_st.size %[[CED_TILE]][%[[C1]]]
  // CHECK-DAG:  %[[ARG_TILE_OFFSET1:.*]] = arith.select %[[IS_D1_EXPANDING]], %[[C0]], %[[CED_TILE_OFFSET1]]
  // CHECK-DAG:  %[[ARG_TILE_SIZE1:.*]] = arith.select %[[IS_D1_EXPANDING]], %[[C1]], %[[CED_TILE_SIZE1]]
  // CHECK-DAG:  %[[ARG_TILE:.*]] = gml_st.tile %[[ISPACE]] [%[[ARG_TILE_OFFSET0]], %[[ARG_TILE_OFFSET1]]] [%[[ARG_TILE_SIZE0]], %[[ARG_TILE_SIZE1]]] [1, 1]
  // CHECK-DAG:  %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK-DAG:  %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE]]]
  // CHECK:      %[[BCAST_SUB:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME:     ins(%[[ARG_SUB]] : tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<3x4x?xf32>)
  // CHECK-SAME:     {broadcast_dimensions = [:i64 0, 2]}
  // CHECK:      return %[[BCAST_SUB]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = gml_st.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      { broadcast_dimensions = [:i64 0, 2] }
  %space = gml_st.space [%d0, %d1, %d2] : !gml_st.tile<?x?x?>
  %bcast_sub = gml_st.materialize %bcast[%tile]
      : tensor<?x?x?xf32>[!gml_st.tile<3x4x?>]
  func.return %bcast_sub : tensor<3x4x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_broadcast_in_dim_at_point
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>, %[[POINT:.*]]: !gml_st.point
func.func @dynamic_broadcast_in_dim_at_point(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>, %point : !gml_st.point) -> f32 {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[S0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[S2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK-DAG: %[[D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [%[[D0]], %[[D1]]]
  // CHECK-DAG: %[[CED_POINT:.*]] = gml_st.drop_dims %[[POINT]], [0, 2]
  // CHECK-DAG: %[[IS_D0_EXPANDING:.*]] = arith.cmpi ne, %[[D0]], %[[S0]]
  // CHECK-DAG: %[[CED_POINT_OFFSET0:.*]] = gml_st.offset %[[CED_POINT]][%[[C0]]]
  // CHECK-DAG: %[[ARG_POINT_OFFSET0:.*]] = arith.select %[[IS_D0_EXPANDING]], %[[C0]], %[[CED_POINT_OFFSET0]]
  // CHECK-DAG: %[[IS_D1_EXPANDING:.*]] = arith.cmpi ne, %[[D1]], %[[S2]]
  // CHECK-DAG: %[[CED_POINT_OFFSET1:.*]] = gml_st.offset %[[CED_POINT]][%[[C1]]]
  // CHECK-DAG: %[[ARG_POINT_OFFSET1:.*]] = arith.select %[[IS_D1_EXPANDING]], %[[C0]], %[[CED_POINT_OFFSET1]]
  // CHECK-DAG: %[[ARG_POINT:.*]] = gml_st.point %[[SPACE]] [%[[ARG_POINT_OFFSET0]], %[[ARG_POINT_OFFSET1]]]
  // CHECK-DAG: %[[BCAST_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_POINT]]]
  // CHECK:     return %[[BCAST_SUB]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = gml_st.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      { broadcast_dimensions = [:i64 0, 2] }
  %space = gml_st.space [%d0, %d1, %d2] : !gml_st.tile<?x?x?>
  %bcast_sub = gml_st.materialize %bcast[%point]
      : tensor<?x?x?xf32>[!gml_st.point]
  func.return %bcast_sub : f32
}

// -----

// CHECK: #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:      @add
// CHECK-SAME: %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>)
func.func @add(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK-DAG:  %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
  // CHECK-DAG:  %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]]
  // CHECK-DAG:  %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[RES:.*]] = linalg.generic
  // CHECK-SAME:     indexing_maps = [#[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]],
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[LHS_SCALAR:.*]]: f32, %[[RHS_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:    %[[RES_SCALAR:.*]] = arith.addf %[[LHS_SCALAR]], %[[RHS_SCALAR]]
  // CHECK:        linalg.yield %[[RES_SCALAR]]
  // CHECK:      return %[[RES]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  return %result : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_two_users
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>, %[[D0:.*]]: index, %[[D1:.*]]: index
func.func @add_two_users(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK:      %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
  // CHECK:      %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]]
  // CHECK:      %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[GENERIC0:.*]] = linalg.generic
  // CHECK-SAME:     ins(%[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK:      %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
  // CHECK:      %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]]
  // CHECK:      %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[GENERIC1:.*]] = linalg.generic
  // CHECK-SAME:     ins(%[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
  // CHECK:      %[[RES:.*]] = linalg.generic
  // CHECK-SAME:     ins(%[[GENERIC0]], %[[GENERIC1]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)
  // CHECK: return %[[RES]]
  %init0 = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg0 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init0 : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %user0 = gml_st.materialize %linalg0[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %user1 = gml_st.materialize %linalg0[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %init1 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %linalg1 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%user0, %user1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init1 : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>
  func.return %linalg1 : tensor<?x?xf32>
}

// -----

// CHECK: #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:      @cos
// CHECK-SAME: %[[ARG:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @cos(%arg: tensor<32x32xf32>, %tile: !gml_st.tile<?x?>)
    -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [32, 32]
  // CHECK-DAG:  %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
  // CHECK-DAG:  %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[RES:.*]] = linalg.generic
  // CHECK-SAME:     indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]],
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[ARG_SUB]] : tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[ARG_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:    %[[RES_SCALAR:.*]] = math.cos %[[ARG_SCALAR]]
  // CHECK:        linalg.yield %[[RES_SCALAR]]
  // CHECK:      return %[[RES]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg : tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%arg_scalar: f32, %_: f32):
    %cos = math.cos %arg_scalar : f32
    linalg.yield %cos : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%tile]
      : tensor<32x32xf32>[!gml_st.tile<?x?>]
  return %result : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_point
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[POINT:.*]]: !gml_st.point
func.func @add_point(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %point: !gml_st.point) -> f32 {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[POINT]]]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[POINT]]]
  // CHECK-DAG: %[[RES:.*]] = arith.addf %[[LHS_SUB]], %[[RHS_SUB]]
  // CHECK:     return %[[RES]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%point]
      : tensor<32x32xf32>[!gml_st.point]
  return %result : f32
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @cos_point
// CHECK-SAME:  %[[ARG:.*]]: tensor<32x32xf32>, %[[POINT:.*]]: !gml_st.point
func.func @cos_point(%arg: tensor<32x32xf32>, %point: !gml_st.point) -> f32 {
  // CHECK-DAG: %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]]
  // CHECK-DAG: %[[RES:.*]] = math.cos %[[ARG_SUB]]
  // CHECK:     return %[[RES]]
  %init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %linalg = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg : tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) {
  ^bb0(%arg_scalar: f32, %_: f32):
    %cos = math.cos %arg_scalar : f32
    linalg.yield %cos : f32
  } -> tensor<32x32xf32>
  %result = gml_st.materialize %linalg[%point]
      : tensor<32x32xf32>[!gml_st.point]
  return %result : f32
}

// -----

// CHECK: #[[ID_MAP:.*]] = affine_map<(d0) -> (d0)>
#id_map = affine_map<(d0) -> (d0)>

// CHECK:      @fuse_into_ploop
// CHECK-SAME: %[[LHS:.*]]: tensor<8xf32>, %[[RHS:.*]]: tensor<8xf32>
func.func @fuse_into_ploop(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>)
    -> tensor<8xf32> {
  // CHECK-DAG:  %[[C8:.*]] = arith.constant 8
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [8]
  // CHECK-DAG:  %[[SPACE:.*]] = gml_st.space [8]
  // CHECK:      %[[RESULT:.*]] = gml_st.parallel (%[[IV:.*]]) = (%[[C0]]) to (%[[C8]]) step (%[[C4]])
  // CHECK-DAG:    %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[IV]]] [4] [1]
  // CHECK-DAG:    %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
  // CHECK-DAG:    %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:        %[[TANH_SUB:.*]] = linalg.generic
  // CHECK-SAME:       indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]]
  // CHECK-SAME:       iterator_types = ["parallel"]
  // CHECK-SAME:       ins(%[[LHS_SUB]] : tensor<4xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB]] : tensor<4xf32>)
  // CHECK:        ^bb0(%[[LHS_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:      %[[TANH_SCALAR:.*]] = math.tanh %[[LHS_SCALAR]]
  // CHECK:          linalg.yield %[[TANH_SCALAR]]
  // CHECK-DAG:    %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]]
  // CHECK-DAG:    %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:        %[[COS_SUB:.*]] = linalg.generic
  // CHECK-SAME:       indexing_maps = [#[[ID_MAP]], #[[ID_MAP]]]
  // CHECK-SAME:       iterator_types = ["parallel"]
  // CHECK-SAME:       ins(%[[RHS_SUB]] : tensor<4xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB]] : tensor<4xf32>)
  // CHECK:        ^bb0(%[[RHS_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:      %[[COS_SCALAR:.*]] = math.cos %[[RHS_SCALAR]]
  // CHECK:          linalg.yield %[[COS_SCALAR]]
  // CHECK-DAG:    %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:        %[[RESULT_SUB:.*]] = linalg.generic
  // CHECK-SAME:       indexing_maps = [#[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
  // CHECK-SAME:       iterator_types = ["parallel"]
  // CHECK-SAME:       ins(%[[TANH_SUB]], %[[COS_SUB]] : tensor<4xf32>, tensor<4xf32>)
  // CHECK-SAME:       outs(%[[INIT_SUB]] : tensor<4xf32>)
  // CHECK:        ^bb0(%[[TANH_SCALAR:.*]]: f32, %[[COS_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:      %[[RESULT_SCALAR:.*]] = arith.addf %[[TANH_SCALAR]], %[[COS_SCALAR]]
  // CHECK:          linalg.yield %[[RESULT_SCALAR]]
  // CHECK:        gml_st.set_yield %[[RESULT_SUB]] into %[[INIT]][%[[TILE]]]
  // CHECK:      return %[[RESULT]]
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %init = linalg.init_tensor [8] : tensor<8xf32>
  %tanh = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel"]}
      ins(%lhs : tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%lhs_scalar: f32, %_: f32):
    %tanh_scalar = math.tanh %lhs_scalar : f32
    linalg.yield %tanh_scalar : f32
  } -> tensor<8xf32>
  %cos = linalg.generic {
      indexing_maps = [#id_map, #id_map],
      iterator_types = ["parallel"]}
      ins(%rhs : tensor<8xf32>)
      outs(%init : tensor<8xf32>) {
  ^bb0(%rhs_scalar: f32, %_: f32):
    %cos_scalar = math.cos %rhs_scalar : f32
    linalg.yield %cos_scalar : f32
  } -> tensor<8xf32>
  %space = gml_st.space [8] : !gml_st.tile<8>
  %result = gml_st.parallel (%iv) = (%c0) to (%c8) step (%c4) {
    %tile = gml_st.tile %space [%iv] [4] [1]
        : !gml_st.tile<8> to !gml_st.tile<4>
    %tanh_sub = gml_st.materialize %tanh[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]
    %cos_sub = gml_st.materialize %cos[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]
    %init_sub = gml_st.materialize %init[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]
    %result_sub = linalg.generic {
        indexing_maps = [#id_map, #id_map, #id_map],
        iterator_types = ["parallel"]}
        ins(%tanh_sub, %cos_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%init_sub : tensor<4xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %tanh0 = arith.addf %arg4, %arg5 : f32
      linalg.yield %tanh0 : f32
    } -> tensor<4xf32>
    gml_st.set_yield %result_sub into %init[%tile]
        : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  return %result : tensor<8xf32>
}

// -----

// CHECK: #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:      @fuse_cwise_linalg_generic
// CHECK-SAME: %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @fuse_cwise_linalg_generic(%lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>, %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %[[LHS]], %[[C0]]
  // CHECK-DAG:  %[[D1:.*]] = tensor.dim %[[LHS]], %[[C1]]
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]]]
  // CHECK-DAG:  %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
  // CHECK-DAG:  %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]]
  // CHECK-DAG:  %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
  // CHECK:      %[[RES:.*]] = linalg.generic
  // CHECK-SAME:     indexing_maps = [#[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
  // CHECK-SAME:     iterator_types = ["parallel", "parallel"]
  // CHECK-SAME:     ins(%[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[INIT_SUB]] : tensor<?x?xf32>)
  // CHECK:      ^bb0(%[[LHS_SCALAR:.*]]: f32, %[[RHS_SCALAR:.*]]: f32, %[[INIT_SCALAR:.*]]: f32):
  // CHECK-DAG:    %[[RES_SCALAR:.*]] = arith.addf %[[LHS_SCALAR]], %[[RHS_SCALAR]]
  // CHECK:        linalg.yield %[[RES_SCALAR]]
  // CHECK:      return %[[RES]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<?x?xf32>
  %4 = gml_st.materialize %3[%tile] : tensor<?x?xf32>[!gml_st.tile<?x?>]
  return %4 : tensor<?x?xf32>
}

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:      @fuse_cwise_linalg_generic_at_point
// CHECK-SAME: %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>, %[[POINT:.*]]: !gml_st.point
func.func @fuse_cwise_linalg_generic_at_point(%lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>, %point: !gml_st.point) -> f32 {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[POINT]]]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[POINT]]]
  // CHECK-DAG: %[[RES:.*]] = arith.addf %[[LHS_SUB]], %[[RHS_SUB]]
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"]}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<?x?xf32>
  %4 = gml_st.materialize %3[%point] : tensor<?x?xf32>[!gml_st.point]
  return %4 : f32
}

// -----

// CHECK:      @dim_reification_fission
// CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
func.func @dim_reification_fission(%arg0: tensor<?xf32>) -> index {
  // CHECK:      %[[C0:.*]] = arith.constant 0
  // CHECK:      %[[DIM:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK:      return %[[DIM]]
  %c0 = arith.constant 0 : index
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = tensor.extract %0[%c0] : tensor<1xindex>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_materialize
// CHECK-SAME:  %{{.*}}: tensor<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @dim_reification_materialize(%arg: tensor<?x?xf32>,
    %tile: !gml_st.tile<?x?>) -> index {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[RES:.*]] = gml_st.size %[[TILE]][%[[C0]]]
  // CHECK:     return %[[RES]]
  %c0 = arith.constant 0 : index
  %0 = gml_st.materialize %arg[%tile] : tensor<?x?xf32>[!gml_st.tile<?x?>]
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_generic
// CHECK-SAME:  %{{.*}}: tensor<?x?xf32>, %[[INIT:.*]]: tensor<?x?xf32>, %[[IDX:.*]]: index
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @dim_reification_generic(%arg: tensor<?x?xf32>,
    %init: tensor<?x?xf32>, %idx: index) -> index {
  // CHECK-DAG: %[[RES:.*]] = tensor.dim %[[INIT]], %[[IDX]] 
  // CHECK:     return %[[RES]]
  %0 = linalg.generic
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%arg : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %2 = math.log %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.dim %0, %idx : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_init_tensor
// CHECK-SAME:  %{{.*}}: index, %[[J:.*]]: index
func.func @dim_reification_init_tensor(%i: index, %j: index) -> index {
  // CHECK: return %[[J]]
  %c1 = arith.constant 1 : index
  %0 = linalg.init_tensor [%i, %j] : tensor<?x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_reification_dynamic_broadcast_in_dim
// CHECK-SAME:  %{{.*}}: tensor<?xf32>, %[[INIT:.*]]: tensor<?x?xf32>
func.func @dim_reification_dynamic_broadcast_in_dim(%arg: tensor<?xf32>,
    %init: tensor<?x?xf32>) -> index {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[RES:.*]] = tensor.dim %[[INIT]], %[[C1]]
  // CHECK:     return %[[RES]] : index
  %c1 = arith.constant 1 : index
  %0 = gml_st.dynamic_broadcast_in_dim
      ins(%arg : tensor<?xf32>) outs(%init : tensor<?x?xf32>)
      {broadcast_dimensions = [:i64 1]}
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}
