// RUN: mlir-hlo-opt %s --split-input-file --gml-fusion | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>,
    %shape : tensor<3xindex>) -> tensor<3x4x5xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2

  // Check init tensor.
  // CHECK-DAG: %[[RES_D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[RES_D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[RES_D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[INIT:.*]] = linalg.init_tensor [%[[RES_D0]], %[[RES_D1]], %[[RES_D2]]]

  // Check result space and tile.
  // CHECK-DAG: %[[RES_SPACE:.*]] = gml_st.space [%[[RES_D0]], %[[RES_D1]], %[[RES_D2]]] : !gml_st.tile<?x?x?>
  // CHECK-DAG: %[[RES_TILE:.*]] = gml_st.tile %[[RES_SPACE]] [0, 1, 2] [3, 4, 5] [1, 1, 1] : !gml_st.tile<?x?x?> to !gml_st.tile<3x4x5>

  // Check arg space.
  // CHECK-DAG: %[[ARG_D0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf32>
  // CHECK-DAG: %[[ARG_D1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf32>
  // CHECK-DAG: %[[ARG_SPACE:.*]] = gml_st.space [%[[ARG_D0]], %[[ARG_D1]]] : !gml_st.tile<?x?>

  // Check collapsing.
  // CHECK-DAG: %[[CED_RES_TILE:.*]] = gml_st.collapse_tile %[[RES_TILE]], [0, 2] : !gml_st.tile<3x4x5> -> !gml_st.tile<3x5>

  // Check first dim of the arg tile.
  // CHECK-DAG: %[[INIT_D0:.*]] = tensor.dim %[[INIT]], %[[C0]] : tensor<?x?x?xf32>
  // CHECK-DAG: %[[IS_EXPANDING_D0:.*]] = arith.cmpi ne, %[[ARG_D0]], %[[INIT_D0]] : index
  // CHECK-DAG: %[[CED_RES_OFFSET_D0:.*]] = gml_st.offset %[[CED_RES_TILE]][%[[C0]]] : !gml_st.tile<3x5>
  // CHECK-DAG: %[[CED_RES_SIZE_D0:.*]] = gml_st.size %[[CED_RES_TILE]][%[[C0]]] : !gml_st.tile<3x5>
  // CHECK-DAG: %[[ARG_OFFSET_D0:.*]] = arith.select %[[IS_EXPANDING_D0]], %[[C0]], %[[CED_RES_OFFSET_D0]] : index
  // CHECK-DAG: %[[ARG_SIZE_D0:.*]] = arith.select %[[IS_EXPANDING_D0]], %[[C1]], %[[CED_RES_SIZE_D0]] : index

  // Check second dim of the arg tile.
  // CHECK-DAG: %[[INIT_D2:.*]] = tensor.dim %[[INIT]], %[[C2]] : tensor<?x?x?xf32>
  // CHECK-DAG: %[[IS_EXPANDING_D1:.*]] = arith.cmpi ne, %[[ARG_D1]], %[[INIT_D2]] : index
  // CHECK-DAG: %[[CED_RES_OFFSET_D1:.*]] = gml_st.offset %[[CED_RES_TILE]][%[[C1]]] : !gml_st.tile<3x5>
  // CHECK-DAG: %[[CED_RES_SIZE_D1:.*]] = gml_st.size %[[CED_RES_TILE]][%[[C1]]] : !gml_st.tile<3x5>
  // CHECK-DAG: %[[ARG_OFFSET_D1:.*]] = arith.select %[[IS_EXPANDING_D1]], %[[C0]], %[[CED_RES_OFFSET_D1]] : index
  // CHECK-DAG: %[[ARG_SIZE_D1:.*]] = arith.select %[[IS_EXPANDING_D1]], %[[C1]], %[[CED_RES_SIZE_D1]] : index

  // Check arg tile.
  // CHECK-DAG: %[[ARG_TILE:.*]] = gml_st.tile %8 [%[[ARG_OFFSET_D0]], %[[ARG_OFFSET_D1]]] [%[[ARG_SIZE_D0]], %[[ARG_SIZE_D1]]] [1, 1] : !gml_st.tile<?x?> to !gml_st.tile<?x?>

  // Check tiled broadcast.
  // CHECK-DAG: %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[RES_TILE]]] : tensor<?x?x?xf32>[!gml_st.tile<3x4x5>]
  // CHECK-DAG: %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[ARG_TILE]]] : tensor<?x?xf32>[!gml_st.tile<?x?>]
  // CHECK-NEXT: %[[RES:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME ins(%[[ARG_SUB]] : tensor<?x?xf32>)
  // CHECK-SAME outs(%[[INIT_SUB]] : tensor<3x4x5xf32>)
  // CHECK-SAME {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}
  // CHECK: return %[[RES]] : tensor<3x4x5xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // `mhlo.dynamic_broadcast_in_dim` lowered to dst-style op.
  %d0 = tensor.extract %shape[%c0] : tensor<3xindex>
  %d1 = tensor.extract %shape[%c1] : tensor<3xindex>
  %d2 = tensor.extract %shape[%c2] : tensor<3xindex>
  %dst = linalg.init_tensor [%d0, %d1, %d2] : tensor<?x?x?xf32>
  %bcast = gml_st.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>) 
      outs(%dst: tensor<?x?x?xf32>) { broadcast_dimensions = dense<[0, 2]> : tensor<2xi64> }

  // Materialze a tile.
  %space = gml_st.space [%d0, %d1, %d2] : !gml_st.tile<?x?x?>
  %tile = gml_st.tile %space [0, 1, 2] [3, 4, 5] [1, 1, 1]
      : !gml_st.tile<?x?x?> to !gml_st.tile<3x4x5>
  %bcast_sub = gml_st.materialize %bcast[%tile]
      : tensor<?x?x?xf32>[!gml_st.tile<3x4x5>]

  func.return %bcast_sub : tensor<3x4x5xf32>
}

// -----

// CHECK-LABEL: @add
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @add(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[RES:.*]] = mhlo.add %[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>
  // CHECK:     return %[[RES]]
  %0 = mhlo.add %lhs, %rhs : tensor<32x32xf32>
  %1 = gml_st.materialize %0[%tile] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  func.return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @add_two_users
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @add_two_users(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %tile: !gml_st.tile<?x?>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[ADD1:.*]] = mhlo.add %[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[ADD2:.*]] = mhlo.add %[[LHS_SUB]], %[[RHS_SUB]] : tensor<?x?xf32>
  // CHECK-DAG: %[[RES:.*]] = mhlo.add %[[ADD1]], %[[ADD2]] : tensor<?x?xf32>
  // CHECK:     return %[[RES]]
  %0 = mhlo.add %lhs, %rhs : tensor<32x32xf32>
  %1 = gml_st.materialize %0[%tile] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %2 = gml_st.materialize %0[%tile] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  %3 = mhlo.add %1, %2 : tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @cos
// CHECK-SAME:  %[[ARG:.*]]: tensor<32x32xf32>, %[[TILE:.*]]: !gml_st.tile<?x?>
func.func @cos(%arg: tensor<32x32xf32>, %tile: !gml_st.tile<?x?>)
    -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  // CHECK-DAG: %[[RES:.*]] = mhlo.cosine %[[ARG_SUB]] : tensor<?x?xf32>
  // CHECK:     return %[[RES]]
  %0 = mhlo.cosine %arg : tensor<32x32xf32>
  %1 = gml_st.materialize %0[%tile] : tensor<32x32xf32>[!gml_st.tile<?x?>]
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @add_point
// CHECK-SAME:  %[[LHS:.*]]: tensor<32x32xf32>, %[[RHS:.*]]: tensor<32x32xf32>, %[[POINT:.*]]: !gml_st.point
func.func @add_point(%lhs: tensor<32x32xf32>, %rhs: tensor<32x32xf32>,
    %point: !gml_st.point) -> f32 {
  // CHECK-DAG: %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[POINT]]] : tensor<32x32xf32>[!gml_st.point]
  // CHECK-DAG: %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[POINT]]] : tensor<32x32xf32>[!gml_st.point]
  // CHECK-DAG: %[[RES:.*]] = arith.addf %[[LHS_SUB]], %[[RHS_SUB]]
  // CHECK:     return %[[RES]]
  %0 = mhlo.add %lhs, %rhs : tensor<32x32xf32>
  %1 = gml_st.materialize %0[%point] : tensor<32x32xf32>[!gml_st.point]
  func.return %1 : f32
}

// -----

// CHECK-LABEL: @cos_point
// CHECK-SAME:  %[[ARG:.*]]: tensor<32x32xf32>, %[[POINT:.*]]: !gml_st.point
func.func @cos_point(%arg: tensor<32x32xf32>, %point: !gml_st.point) -> f32 {
  // CHECK-DAG: %[[ARG_SUB:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]] : tensor<32x32xf32>[!gml_st.point]
  // CHECK-DAG: %[[RES:.*]] = math.cos %[[ARG_SUB]]
  // CHECK:     return %[[RES]]
  %0 = mhlo.cosine %arg : tensor<32x32xf32>
  %1 = gml_st.materialize %0[%point] : tensor<32x32xf32>[!gml_st.point]
  return %1 : f32
}

// -----

#cwise_trait = {
  indexing_maps = [
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>
  ],
  iterator_types = ["parallel"]
}

// CHECK-LABEL: @fuse_into_ploop
// CHECK-SAME:  %[[LHS:.*]]: tensor<8xf32>, %[[RHS:.*]]: tensor<8xf32>, %[[OUT:.*]]: tensor<8xf32>
func.func @fuse_into_ploop(%lhs : tensor<8xf32>, %rhs : tensor<8xf32>,
                           %out: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [8] : !gml_st.tile<8>
  // CHECK:     %[[RES:.*]] = gml_st.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[C8]]) step (%[[C4]]) {
  // CHECK-DAG:   %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]]] [4] [1] : !gml_st.tile<8> to !gml_st.tile<4>
  // CHECK-DAG:   %[[LHS_SUB:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]] : tensor<8xf32>[!gml_st.tile<4>]
  // CHECK-DAG:   %[[RHS_SUB:.*]] = gml_st.materialize %[[RHS]][%[[TILE]]] : tensor<8xf32>[!gml_st.tile<4>]
  // CHECK-DAG:   %[[OUT_SUB:.*]] = gml_st.materialize %[[OUT]][%[[TILE]]] : tensor<8xf32>[!gml_st.tile<4>]
  // CHECK-DAG:   %[[TANH_SUB:.*]] = mhlo.tanh %[[LHS_SUB]]
  // CHECK-DAG:   %[[COS_SUB:.*]] = mhlo.cosine %[[RHS_SUB]]
  // CHECK:       %[[RES_SUB:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[TANH_SUB]], %[[COS_SUB]] : tensor<4xf32>, tensor<4xf32>) outs(%[[OUT_SUB]] : tensor<4xf32>)
  // CHECK:       ^bb0(%[[TANH_SCALAR:.*]]: f32, %[[COS_SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK-DAG:     %[[RES_SCALAR:.*]] = arith.addf %[[TANH_SCALAR]], %[[COS_SCALAR]] : f32
  // CHECK:         linalg.yield %[[RES_SCALAR]]
  // CHECK:       gml_st.set_yield %[[RES_SUB]] into %[[OUT]][%[[TILE]]
  // CHECK-SAME:    : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  // CHECK:     return %[[RES]]

  %tanh = mhlo.tanh %lhs : tensor<8xf32>
  %cos = mhlo.cosine %rhs : tensor<8xf32>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %space = gml_st.space [8] : !gml_st.tile<8>
  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4) {
    %tile = gml_st.tile %space [%i] [4] [1] : !gml_st.tile<8> to !gml_st.tile<4>
    %tanh_sub = gml_st.materialize %tanh[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]
    %cos_sub = gml_st.materialize %cos[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]
    %out_sub = gml_st.materialize %out[%tile]
        : tensor<8xf32>[!gml_st.tile<4>]

    %result_sub = linalg.generic #cwise_trait
        ins(%tanh_sub, %cos_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    gml_st.set_yield %result_sub into %out[%tile] : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
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
