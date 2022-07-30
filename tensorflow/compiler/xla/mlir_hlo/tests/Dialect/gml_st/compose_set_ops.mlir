// RUN: mlir-hlo-opt %s --split-input-file --gml-compose-set-ops | \
// RUN: FileCheck %s

// CHECK-LABEL: @tile_of_tile
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[M:.*]]: index, %[[N:.*]]: index, %[[A:.*]]: index, %[[B:.*]]: index
func.func @tile_of_tile(%arg : tensor<?x?xf32>, %i : index, %j : index,
    %k : index, %m : index, %n : index, %a : index, %b : index)
    -> tensor<4x?xf32> {
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [1024, %[[M]]] : !gml_st.tile<1024x?>
  // CHECK-DAG: %[[AK:.*]] = arith.muli %[[A]], %[[K]]
  // CHECK-DAG: %[[J_PLUS_AK:.*]] = arith.addi %[[J]], %[[AK]]
  // CHECK-DAG: %[[AB:.*]] = arith.muli %[[A]], %[[B]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J_PLUS_AK]]] [4, %[[N]]] [2, %[[AB]]] : !gml_st.tile<1024x?> to !gml_st.tile<4x?>
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]] : tensor<?x?xf32>[!gml_st.tile<4x?>]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [1024, %m] : !gml_st.tile<1024x?>
  %tile = gml_st.tile %space [%i, %j] [4, 128] [2, %a]
      : !gml_st.tile<1024x?> to !gml_st.tile<4x128>
  %tile_of_tile = gml_st.tile %tile [0, %k] [4, %n] [1, %b]
      : !gml_st.tile<4x128> to !gml_st.tile<4x?>
  %result = gml_st.materialize %arg[%tile_of_tile]
      : tensor<?x?xf32>[!gml_st.tile<4x?>]
  func.return %result : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @tile_of_tile_of_tile_all_constant
// CHECK-SAME:  %[[ARG:.*]]: tensor<4096x2048xf32>
func.func @tile_of_tile_of_tile_all_constant(%arg : tensor<4096x2048xf32>)
    -> tensor<128x64xf32> {
  // CHECK: %[[SPACE:.*]] = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  // CHECK: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [18, 64] [128, 64] [4, 0] : !gml_st.tile<4096x2048> to !gml_st.tile<128x64>
  // CHECK: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]] : tensor<4096x2048xf32>[!gml_st.tile<128x64>]
  // CHECK:     return %[[RES]]
  %s = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  %t = gml_st.tile %s [0, 32] [2048, 256] [1, 2]
      : !gml_st.tile<4096x2048> to !gml_st.tile<2048x256>
  %tt = gml_st.tile %t [2, 16] [256, 128] [4, 0]
      : !gml_st.tile<2048x256> to !gml_st.tile<256x128>
  %ttt = gml_st.tile %tt [4, 8] [128, 64] [1, 1]
      : !gml_st.tile<256x128> to !gml_st.tile<128x64>
  %res = gml_st.materialize %arg[%ttt]
      : tensor<4096x2048xf32>[!gml_st.tile<128x64>]
  func.return %res : tensor<128x64xf32>
}

// -----

// CHECK-LABEL: @tile_chain_w_zeroes_and_ones
// CHECK-SAME:  %[[ARG:.*]]: tensor<8192x4096x2048xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[M:.*]]: index, %[[N:.*]]: index, %[[O:.*]]: index, %[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index
func.func @tile_chain_w_zeroes_and_ones(%arg : tensor<8192x4096x2048xf32>,
    %i : index, %j : index, %k : index, %m : index, %n : index, %o : index,
    %a : index, %b : index, %c : index) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [8192, 4096, 2048] : !gml_st.tile<8192x4096x2048>
  // CHECK-DAG: %[[TWO_K:.*]] = arith.muli %[[K]], %[[C2]]
  // CHECK-DAG: %[[SIXTEEN_PLUS_J:.*]] = arith.addi %[[J]], %[[C16]]
  // CHECK-DAG: %[[TWO_K_PLUS_32:.*]] = arith.addi %[[TWO_K]], %[[C32]]
  // CHECK-DAG: %[[C_TIMES_C2:.*]] = arith.muli %[[C]], %[[C2]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [0, %[[SIXTEEN_PLUS_J]], %[[TWO_K_PLUS_32]]] [%[[M]], %[[N]], %[[O]]] [0, %[[B]], %[[C_TIMES_C2]]] : !gml_st.tile<8192x4096x2048> to !gml_st.tile<?x?x?>
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]] : tensor<8192x4096x2048xf32>[!gml_st.tile<?x?x?>]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [8192, 4096, 2048] : !gml_st.tile<8192x4096x2048>
  %tile = gml_st.tile %space [0, 16, 32] [2048, 1024, 512] [0, 1, 2]
      : !gml_st.tile<8192x4096x2048> to !gml_st.tile<2048x1024x512>
  %tile_of_tile = gml_st.tile %tile [%i, %j, %k] [%m, %n, %o] [%a, %b, %c]
      : !gml_st.tile<2048x1024x512> to !gml_st.tile<?x?x?>
  %result = gml_st.materialize %arg[%tile_of_tile]
      : tensor<8192x4096x2048xf32>[!gml_st.tile<?x?x?>]
  func.return %result : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @tile_of_tile_arith_shortcuts_add
// CHECK-SAME:  %[[ARG:.*]]: tensor<32x32x32xf32>, %[[I:.*]]: index, %[[J:.*]]: index
func.func @tile_of_tile_arith_shortcuts_add(%arg : tensor<32x32x32xf32>,
    %i : index, %j : index) -> tensor<8x8x8xf32> {
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space
  // CHECK-DAG: %[[I_PLUS_J:.*]] = arith.addi %[[I]], %[[J]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[J]], %[[I]], %[[I_PLUS_J]]] [8, 8, 8] [1, 1, 1]
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [32, 32, 32] : !gml_st.tile<32x32x32>
  %tile = gml_st.tile %space [0, %i, %i] [16, 16, 16] [1, 1, 1]
      : !gml_st.tile<32x32x32> to !gml_st.tile<16x16x16>
  %tile_of_tile = gml_st.tile %tile [%j, 0, %j] [8, 8, 8] [1, 1, 1]
      : !gml_st.tile<16x16x16> to !gml_st.tile<8x8x8>
  %result = gml_st.materialize %arg[%tile_of_tile]
      : tensor<32x32x32xf32>[!gml_st.tile<8x8x8>]
  func.return %result : tensor<8x8x8xf32>
}

// -----

// CHECK-LABEL: @tile_of_tile_arith_shortcuts_mul
// CHECK-SAME:  %[[ARG:.*]]: tensor<32x32x32x32x32xf32>, %[[A:.*]]: index, %[[B:.*]]: index
func.func @tile_of_tile_arith_shortcuts_mul(%arg : tensor<32x32x32x32x32xf32>,
    %a : index, %b : index) -> tensor<8x8x8x8x8xf32> {
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space
  // CHECK-DAG: %[[AB:.*]] = arith.muli %[[A]], %[[B]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [0, %[[B]], 0, %[[A]], %[[AB]]]
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [32, 32, 32, 32, 32] : !gml_st.tile<32x32x32x32x32>
  %tile = gml_st.tile %space
      [0, 0, 0, 0, 0] [16, 16, 16, 16, 16] [0, 1, %a, %a, %a]
      : !gml_st.tile<32x32x32x32x32> to !gml_st.tile<16x16x16x16x16>
  %tile_of_tile = gml_st.tile %tile
      [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [%b, %b, 0, 1, %b]
      : !gml_st.tile<16x16x16x16x16> to !gml_st.tile<8x8x8x8x8>
  %result = gml_st.materialize %arg[%tile_of_tile]
      : tensor<32x32x32x32x32xf32>[!gml_st.tile<8x8x8x8x8>]
  func.return %result : tensor<8x8x8x8x8xf32>
}

// -----

// CHECK-LABEL: @point_of_tile
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[M:.*]]: index, %[[A:.*]]: index
func.func @point_of_tile(%arg : tensor<?x?xf32>, %i : index, %j : index,
    %k : index, %m : index, %a : index) -> f32 {
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [1024, %[[M]]] : !gml_st.tile<1024x?>
  // CHECK-DAG: %[[AK:.*]] = arith.muli %[[A]], %[[K]]
  // CHECK-DAG: %[[J_PLUS_AK:.*]] = arith.addi %[[J]], %[[AK]]
  // CHECK-DAG: %[[POINT:.*]] = gml_st.point %[[SPACE]] [%[[I]], %[[J_PLUS_AK]]] : !gml_st.tile<1024x?> to !gml_st.point
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]] : tensor<?x?xf32>[!gml_st.point]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [1024, %m] : !gml_st.tile<1024x?>
  %tile = gml_st.tile %space [%i, %j] [4, 128] [2, %a]
      : !gml_st.tile<1024x?> to !gml_st.tile<4x128>
  %point_of_tile = gml_st.point %tile [0, %k]
      : !gml_st.tile<4x128> to !gml_st.point
  %result = gml_st.materialize %arg[%point_of_tile]
      : tensor<?x?xf32>[!gml_st.point]
  func.return %result : f32
}

// -----

// CHECK-LABEL: @point_of_tile_of_tile_all_constant
// CHECK-SAME:  %[[ARG:.*]]: tensor<4096x2048xf32>
func.func @point_of_tile_of_tile_all_constant(%arg : tensor<4096x2048xf32>)
    -> f32 {
  // CHECK: %[[SPACE:.*]] = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  // CHECK: %[[POINT:.*]] = gml_st.point %[[SPACE]] [18, 64] : !gml_st.tile<4096x2048> to !gml_st.point
  // CHECK: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]] : tensor<4096x2048xf32>[!gml_st.point]
  // CHECK:     return %[[RES]]
  %s = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  %t = gml_st.tile %s [0, 32] [2048, 256] [1, 2]
      : !gml_st.tile<4096x2048> to !gml_st.tile<2048x256>
  %tt = gml_st.tile %t [2, 16] [256, 128] [4, 0]
      : !gml_st.tile<2048x256> to !gml_st.tile<256x128>
  %ptt = gml_st.point %tt [4, 8] : !gml_st.tile<256x128> to !gml_st.point
  %res = gml_st.materialize %arg[%ptt]
      : tensor<4096x2048xf32>[!gml_st.point]
  func.return %res : f32
}

// -----

// CHECK-LABEL: @point_chain_w_zeroes_and_ones
// CHECK-SAME:  %[[ARG:.*]]: tensor<8192x4096x2048xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index
func.func @point_chain_w_zeroes_and_ones(%arg : tensor<8192x4096x2048xf32>,
    %i : index, %j : index, %k : index) -> f32 {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [8192, 4096, 2048] : !gml_st.tile<8192x4096x2048>
  // CHECK-DAG: %[[TWO_K:.*]] = arith.muli %[[K]], %[[C2]]
  // CHECK-DAG: %[[SIXTEEN_PLUS_J:.*]] = arith.addi %[[J]], %[[C16]]
  // CHECK-DAG: %[[TWO_K_PLUS_32:.*]] = arith.addi %[[TWO_K]], %[[C32]]
  // CHECK-DAG: %[[POINT:.*]] = gml_st.point %[[SPACE]] [0, %[[SIXTEEN_PLUS_J]], %[[TWO_K_PLUS_32]]] : !gml_st.tile<8192x4096x2048> to !gml_st.point
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]] : tensor<8192x4096x2048xf32>[!gml_st.point]
  // CHECK:     return %[[RES]]
  %space = gml_st.space [8192, 4096, 2048] : !gml_st.tile<8192x4096x2048>
  %tile = gml_st.tile %space [0, 16, 32] [2048, 1024, 512] [0, 1, 2]
      : !gml_st.tile<8192x4096x2048> to !gml_st.tile<2048x1024x512>
  %point_of_tile = gml_st.point %tile [%i, %j, %k]
      : !gml_st.tile<2048x1024x512> to !gml_st.point
  %result = gml_st.materialize %arg[%point_of_tile]
      : tensor<8192x4096x2048xf32>[!gml_st.point]
  func.return %result : f32
}

// -----

// CHECK-LABEL: @point_of_transpose_dims_of_tile_all_constant
// CHECK-SAME:  %[[ARG:.*]]: tensor<2048x4096xf32>
func.func @point_of_transpose_dims_of_tile_all_constant(%arg : tensor<2048x4096xf32>)
    -> f32 {
  // CHECK: %[[SPACE:.*]] = gml_st.space [2048, 4096] : !gml_st.tile<2048x4096>
  // CHECK: %[[POINT:.*]] = gml_st.point %[[SPACE]] [40, 8] : !gml_st.tile<2048x4096> to !gml_st.point
  // CHECK: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[POINT]]] : tensor<2048x4096xf32>[!gml_st.point]
  // CHECK:     return %[[RES]]
  %s = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  %t = gml_st.tile %s [0, 32] [128, 256] [1, 2]
      : !gml_st.tile<4096x2048> to !gml_st.tile<128x256>
  %tt = gml_st.transpose_dims %t, [1, 0]
      : !gml_st.tile<128x256> to !gml_st.tile<256x128>
  %ptt = gml_st.point %tt [4, 8] : !gml_st.tile<256x128> to !gml_st.point
  %res = gml_st.materialize %arg[%ptt]
      : tensor<2048x4096xf32>[!gml_st.point]
  func.return %res : f32
}

// -----

// CHECK-LABEL: @transpose_dims_of_transpose_dims_of_tile
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x?x5xf32>, %[[SIZE:.*]]: index
func.func @transpose_dims_of_transpose_dims_of_tile(
    %arg : tensor<10x?x5xf32>, %size: index) -> tensor<4x?x5xf32> {
// CHECK: %[[SPACE:.*]] = gml_st.space [10, %[[SIZE]], 5] : !gml_st.tile<10x?x5>
// CHECK: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [3, 0, 0] [4, %[[SIZE]], 5] [2, %[[SIZE]], 1] : !gml_st.tile<10x?x5> to !gml_st.tile<4x?x5>
// CHECK: %[[RES:.*]] = gml_st.materialize %arg0[%[[TILE]]] : tensor<10x?x5xf32>[!gml_st.tile<4x?x5>]
// CHECK: return %[[RES]] : tensor<4x?x5xf32>
  %s = gml_st.space [%size, 5, 10] : !gml_st.tile<?x5x10>
  %t = gml_st.tile %s [0, 0, 3] [%size, 5, 4] [%size, 1, 2]
      : !gml_st.tile<?x5x10> to !gml_st.tile<?x5x4>
  %tt = gml_st.transpose_dims %t, [1, 0, 2]
      : !gml_st.tile<?x5x4> to !gml_st.tile<5x?x4>
  %tt2 = gml_st.transpose_dims %tt, [2, 1, 0]
      : !gml_st.tile<5x?x4> to !gml_st.tile<4x?x5>
  %res = gml_st.materialize %arg[%tt2]
      : tensor<10x?x5xf32>[!gml_st.tile<4x?x5>]
  func.return %res : tensor<4x?x5xf32>
}

// -----

// CHECK-LABEL: @transpose_dims_of_space
// CHECK-SAME:  %[[ARG:.*]]: tensor<5x10x?xf32>, %[[SIZE:.*]]: index
func.func @transpose_dims_of_space(
    %arg : tensor<5x10x?xf32>, %size: index) -> tensor<5x10x?xf32> {
// CHECK: %[[SPACE:.*]] = gml_st.space [5, 10, %[[SIZE]]] : !gml_st.tile<5x10x?>
// CHECK: %[[RES:.*]] = gml_st.materialize %arg0[%[[SPACE]]] : tensor<5x10x?xf32>[!gml_st.tile<5x10x?>]
// CHECK: return %[[RES]] : tensor<5x10x?xf32>
  %s = gml_st.space [%size, 5, 10] : !gml_st.tile<?x5x10>
  %tt = gml_st.transpose_dims %s, [1, 2, 0]
      : !gml_st.tile<?x5x10> to !gml_st.tile<5x10x?>
  %res = gml_st.materialize %arg[%tt]
      : tensor<5x10x?xf32>[!gml_st.tile<5x10x?>]
  func.return %res : tensor<5x10x?xf32>
}

// -----

// CHECK-LABEL: @drop_dims_of_space
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x10xf32>, %[[SIZE:.*]]: index
func.func @drop_dims_of_space(
    %arg : tensor<?x10xf32>, %size: index) -> tensor<?x10xf32> {
// CHECK: %[[SPACE:.*]] = gml_st.space [%[[SIZE]], 10] : !gml_st.tile<?x10>
// CHECK: %[[RES:.*]] = gml_st.materialize %arg0[%[[SPACE]]] : tensor<?x10xf32>[!gml_st.tile<?x10>]
// CHECK: return %[[RES]] : tensor<?x10xf32>
  %s = gml_st.space [%size, 5, 10] : !gml_st.tile<?x5x10>
  %tt = gml_st.drop_dims %s, [0, 2]
      : !gml_st.tile<?x5x10> to !gml_st.tile<?x10>
  %res = gml_st.materialize %arg[%tt]
      : tensor<?x10xf32>[!gml_st.tile<?x10>]
  func.return %res : tensor<?x10xf32>
}
