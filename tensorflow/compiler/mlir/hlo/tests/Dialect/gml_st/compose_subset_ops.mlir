// RUN: mlir-hlo-opt %s --split-input-file --gml-compose-subset-ops | \
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
  // CHECK-DAG: %[[SPACE:.*]] = gml_st.space [4096, 2048] : !gml_st.tile<4096x2048>
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [18, 64] [128, 64] [4, 0] : !gml_st.tile<4096x2048> to !gml_st.tile<128x64>
  // CHECK-DAG: %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]] : tensor<4096x2048xf32>[!gml_st.tile<128x64>]
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
  // CHECK-DAG: %[[SIXTEEN_J:.*]] = arith.addi %[[J]], %[[C16]]
  // CHECK-DAG: %[[TWO_K_PLUS_32:.*]] = arith.addi %[[TWO_K]], %[[C32]]
  // CHECK-DAG: %[[C_TIMES_C2:.*]] = arith.muli %[[C]], %[[C2]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [0, %[[SIXTEEN_J]], %[[TWO_K_PLUS_32]]] [%[[M]], %[[N]], %[[O]]] [0, %[[B]], %[[C_TIMES_C2]]] : !gml_st.tile<8192x4096x2048> to !gml_st.tile<?x?x?>
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
  // CHECK-DAG: %[[IJ:.*]] = arith.addi %[[I]], %[[J]]
  // CHECK-DAG: %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[J]], %[[I]], %[[IJ]]] [8, 8, 8] [1, 1, 1]
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
