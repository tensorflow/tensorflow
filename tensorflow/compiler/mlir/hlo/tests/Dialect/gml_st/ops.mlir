// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

// CHECK-LABEL: @types
func.func @types() {
  // CHECK: %[[ARG:.*]] = gml_st.space [64] : !gml_st.tile<64>
  %0 = gml_st.space [64] : !gml_st.tile<64>
  // CHECK: %[[ARG2:.*]] = gml_st.space [64, 32] : !gml_st.tile<64x32>
  %1 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  // CHECK: %{{.*}} = gml_st.point %[[ARG]] [42] : !gml_st.tile<64> to !gml_st.point
  %2 = gml_st.point %0 [42] : !gml_st.tile<64> to !gml_st.point
  // CHECK: %{{.*}} = gml_st.tile %[[ARG2]] [0, 0] [42, 16] [1, 1] : !gml_st.tile<64x32> to !gml_st.tile<42x16>
  %3 = gml_st.tile %1 [0, 0] [42, 16] [1, 1] : !gml_st.tile<64x32> to !gml_st.tile<42x16>
  // CHECK: %{{.*}} = gml_st.transpose_tile %[[ARG2]], [1, 0] : !gml_st.tile<64x32> to !gml_st.tile<32x64>
  %4 = gml_st.transpose_tile %1, [1, 0] : !gml_st.tile<64x32> to !gml_st.tile<32x64>
  func.return
}

// -----

// CHECK-LABEL: @dynamic_types
// CHECK-SAME: (%[[SIZE:.*]]: index)
func.func @dynamic_types(%size : index) {
  // CHECK: %[[ARG:.*]] = gml_st.space [%[[SIZE]]] : !gml_st.tile<?>
  %0 = gml_st.space [%size] : !gml_st.tile<?>
  // CHECK: %[[ARG2:.*]] = gml_st.space [%[[SIZE]], 16] : !gml_st.tile<?x16>
  %1 = gml_st.space [%size, 16] : !gml_st.tile<?x16>
  // CHECK: %{{.*}} = gml_st.point %[[ARG]] [42] : !gml_st.tile<?> to !gml_st.point
  %2 = gml_st.point %0 [42] : !gml_st.tile<?> to !gml_st.point
  // CHECK: %{{.*}} = gml_st.tile %[[ARG2]] [0, 0] [42, 8] [1, 1] : !gml_st.tile<?x16> to !gml_st.tile<42x8>
  %3 = gml_st.tile %1 [0, 0] [42, 8] [1, 1] : !gml_st.tile<?x16> to !gml_st.tile<42x8>
  func.return
}

// -----

// CHECK-LABEL: @materialize_static_memref
// CHECK-SAME: %[[MEMREF:.*]]: memref<64x32xf32>, %[[TILE:.*]]: !gml_st.tile<42x16>, %[[POINT:.*]]: !gml_st.point
func.func @materialize_static_memref(%memref: memref<64x32xf32>, %tile: !gml_st.tile<42x16>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]][%[[TILE]]] : memref<64x32xf32>[!gml_st.tile<42x16>]
  %0 = gml_st.materialize %memref[%tile] : memref<64x32xf32>[!gml_st.tile<42x16>]
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]][%[[POINT]]] : memref<64x32xf32>[!gml_st.point]
  %1 = gml_st.materialize %memref[%point] : memref<64x32xf32>[!gml_st.point]
  func.return
}

// -----

// CHECK-LABEL: @materialize_static_tensor
// CHECK-SAME: %[[TENSOR:.*]]: tensor<64x32xf32>, %[[TILE:.*]]: !gml_st.tile<42x16>, %[[POINT:.*]]: !gml_st.point
func.func @materialize_static_tensor(%tensor: tensor<64x32xf32>, %tile: !gml_st.tile<42x16>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]][%[[TILE]]] : tensor<64x32xf32>[!gml_st.tile<42x16>]
  %0 = gml_st.materialize %tensor[%tile] : tensor<64x32xf32>[!gml_st.tile<42x16>]
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]][%[[POINT]]] : tensor<64x32xf32>[!gml_st.point]
  %1 = gml_st.materialize %tensor[%point] : tensor<64x32xf32>[!gml_st.point]
  func.return
}

// -----

// CHECK-LABEL: @materialize_dynamic_memref
// CHECK-SAME: %[[MEMREF:.*]]: memref<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<42x16>, %[[POINT:.*]]: !gml_st.point
func.func @materialize_dynamic_memref(%memref: memref<?x?xf32>, %tile: !gml_st.tile<42x16>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]][%[[TILE]]] : memref<?x?xf32>[!gml_st.tile<42x16>]
  %0 = gml_st.materialize %memref[%tile] : memref<?x?xf32>[!gml_st.tile<42x16>]
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]][%[[POINT]]] : memref<?x?xf32>[!gml_st.point]
  %1 = gml_st.materialize %memref[%point] : memref<?x?xf32>[!gml_st.point]
  func.return
}

// -----

// CHECK-LABEL: @materialize_dynamic_tensor
// CHECK-SAME: %[[TENSOR:.*]]: tensor<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<42x16>, %[[POINT:.*]]: !gml_st.point
func.func @materialize_dynamic_tensor(%tensor: tensor<?x?xf32>, %tile: !gml_st.tile<42x16>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]][%[[TILE]]] : tensor<?x?xf32>[!gml_st.tile<42x16>]
  %0 = gml_st.materialize %tensor[%tile] : tensor<?x?xf32>[!gml_st.tile<42x16>]
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]][%[[POINT]]] : tensor<?x?xf32>[!gml_st.point]
  %1 = gml_st.materialize %tensor[%point] : tensor<?x?xf32>[!gml_st.point]
  func.return
}

// -----

#cwise_trait = {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
}

func.func @tiled_loop(%lhs: tensor<24x64xi8>, %rhs: tensor<24x64xi8>,
                 %out: tensor<24x64xi8>) -> tensor<24x64xi8> {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %c4 = arith.constant 4 : index
 %c24 = arith.constant 24 : index
 %c64 = arith.constant 64 : index
 %prod = gml_st.loop (%i) = (%c0) to (%c24) step (%c4)
      ins(%lhs_ = %lhs: tensor<24x64xi8>, %rhs_ = %rhs: tensor<24x64xi8>)
      outs(%out_ = %out: tensor<24x64xi8>) {
    %lhs_sub = tensor.extract_slice %lhs_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %rhs_sub = tensor.extract_slice %rhs_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>
    %out_sub = tensor.extract_slice %out_[%i, 0] [%c4, %c64] [1, 1]
        : tensor<24x64xi8> to tensor<?x?xi8>

    %sum = linalg.generic #cwise_trait
        ins(%lhs_sub, %rhs_sub : tensor<?x?xi8>, tensor<?x?xi8>)
        outs(%out_sub : tensor<?x?xi8>) {
      ^bb(%l: i8, %r: i8, %o: i8) :
        %s = arith.addi %l, %r : i8
        linalg.yield %s : i8
      } -> tensor<?x?xi8>

    %sum_sub = tensor.insert_slice %sum into %out_[%i, 0][%c4, %c64][1, 1]
      : tensor<?x?xi8> into tensor<24x64xi8>
    gml_st.yield %sum_sub : tensor<24x64xi8>
  }
  func.return %prod : tensor<24x64xi8>
}
// CHECK-LABEL: func @tiled_loop
// CHECK-NOT: iterators[

// -----

#reduction_trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1)>,
    affine_map<(d0, d1, d2) -> (d1)>
  ],
  iterator_types = ["reduction", "parallel", "reduction"]
}

func.func @tiled_loop_reduction(%input_3d: tensor<16x24x32xf32>,
                           %input_2d: tensor<16x32xf32>,
                           %input_1d: tensor<24xf32>,
                           %output: tensor<24xf32>) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %X = tensor.dim %input_3d, %c0 : tensor<16x24x32xf32>
  %Y = tensor.dim %input_3d, %c1 : tensor<16x24x32xf32>
  %Z = tensor.dim %input_3d, %c2 : tensor<16x24x32xf32>
  %result = gml_st.loop (%i, %j, %k)
      = (%c0, %c0, %c0) to (%X, %Y, %Z) step (%c2, %c4, %c8)
      ins(%i3d_ = %input_3d: tensor<16x24x32xf32>,
          %i2d_ = %input_2d: tensor<16x32xf32>,
          %i1d_ = %input_1d: tensor<24xf32>)
      outs(%o_ =  %output: tensor<24xf32>)
      iterators["reduction", "parallel", "reduction"]
      distribution["block_x", "block_y", "none"] {
    %sub_3d = tensor.extract_slice %i3d_[%i, %j, %k][2, 4, 8][1, 1, 1]
      : tensor<16x24x32xf32> to tensor<2x4x8xf32>
    %sub_2d = tensor.extract_slice %i2d_[%i, %k][2, 8][1, 1]
      : tensor<16x32xf32> to tensor<2x8xf32>
    %sub_1d = tensor.extract_slice %i1d_[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %sub_out = tensor.extract_slice %o_[%j] [4] [1]
      : tensor<24xf32> to tensor<4xf32>
    %acc = linalg.generic #reduction_trait
      ins(%sub_3d, %sub_2d, %sub_1d
        : tensor<2x4x8xf32>, tensor<2x8xf32>, tensor<4xf32>)
      outs(%sub_out : tensor<4xf32>)  {
    ^bb0(%i3d: f32, %i2d: f32, %i1d: f32, %o: f32):
      %0 = arith.addf %i3d, %i2d : f32
      %1 = arith.addf %0, %i1d : f32
      linalg.yield %1 : f32
    } -> tensor<4xf32>

    %sum_sub = tensor.insert_slice %acc into %o_[%j][4][1]
      : tensor<4xf32> into tensor<24xf32>
    gml_st.yield %sum_sub : tensor<24xf32>
  }
  func.return %result : tensor<24xf32>
}
// CHECK-LABEL: func @tiled_loop_reduction
// CHECK: iterators[


#map_1 = affine_map<(d0, d1, d2)[s0] -> (d0 * 768 + s0 + d1 * 32 + d2)>
#map_2 = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>
#map_3 = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @tiled_loop_on_buffers(%input_3d: memref<16x24x32xf32>,
                            %input_2d: memref<16x32xf32>,
                            %input_1d: memref<24xf32>,
                            %output: memref<24xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %X = memref.dim %input_3d, %c0 : memref<16x24x32xf32>
  %Y = memref.dim %input_3d, %c1 : memref<16x24x32xf32>
  %Z = memref.dim %input_3d, %c2 : memref<16x24x32xf32>
  gml_st.loop (%i, %j, %k) = (%c0, %c0, %c0)
      to (%X, %Y, %Z) step (%c2, %c4, %c8)
      ins(%i3d_ = %input_3d: memref<16x24x32xf32>,
          %i2d_ = %input_2d: memref<16x32xf32>,
          %i1d_ = %input_1d: memref<24xf32>)
      outs(%o_ =  %output: memref<24xf32>)
      iterators["reduction", "parallel", "reduction"] {
    %sub_3d = memref.subview %i3d_[%i, %j, %k][2, 4, 8][1, 1, 1]
      : memref<16x24x32xf32> to memref<2x4x8xf32, #map_1>
    %sub_2d = memref.subview %i2d_[%i, %k][2, 8][1, 1]
      : memref<16x32xf32> to memref<2x8xf32, #map_2>
    %sub_1d = memref.subview %i1d_[%j] [4] [1]
      : memref<24xf32> to memref<4xf32, #map_3>
    %sub_out = memref.subview %o_[%j] [4] [1]
      : memref<24xf32> to memref<4xf32, #map_3>
    linalg.generic #reduction_trait
      ins(%sub_3d, %sub_2d, %sub_1d
        : memref<2x4x8xf32, #map_1>,
          memref<2x8xf32, #map_2>,
          memref<4xf32, #map_3>)
      outs(%sub_out : memref<4xf32, #map_3>)  {
    ^bb0(%i3d: f32, %i2d: f32, %i1d: f32, %o: f32):
      %0 = arith.addf %i3d, %i2d : f32
      %1 = arith.addf %0, %i1d : f32
      linalg.yield %1 : f32
    }
    gml_st.yield
  }
  func.return
}
// CHECK-LABEL: func @tiled_loop_on_buffers
// CHECK: iterators[

// -----

#cwise_trait = {
  indexing_maps = [
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>
  ],
  iterator_types = ["parallel"]
}

func.func @parallel_loop(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                         %output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %space = gml_st.space [8] : !gml_st.tile<8>

  %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4) {
    %tile = gml_st.tile %space [%i] [4] [1]
      : !gml_st.tile<8> to !gml_st.tile<4>
    %lhs_sub = gml_st.materialize %lhs[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]
    %rhs_sub = gml_st.materialize %rhs[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]
    %out_sub = gml_st.materialize %output[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]

    %result_sub = linalg.generic #cwise_trait
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    gml_st.set_yield %result_sub into %output[%tile]
      : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @parallel_loop

func.func @for_loop(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                         %output: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %space = gml_st.space [8] : !gml_st.tile<8>

  %sum = gml_st.for (%i) = (%c0) to (%c8) step (%c4)
      outs(%out_ = %output : tensor<8xf32>) {
    %tile = gml_st.tile %space [%i] [4] [1]
      : !gml_st.tile<8> to !gml_st.tile<4>
    %lhs_sub = gml_st.materialize %lhs[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]
    %rhs_sub = gml_st.materialize %rhs[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]
    %out_sub = gml_st.materialize %out_[%tile]
      : tensor<8xf32>[!gml_st.tile<4>]

    %result_sub = linalg.generic #cwise_trait
        ins(%lhs_sub, %rhs_sub : tensor<4xf32>, tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb(%l: f32, %r: f32, %o: f32) :
        %s = arith.addf %l, %r : f32
        linalg.yield %s : f32
    } -> tensor<4xf32>

    gml_st.set_yield %result_sub into %out_[%tile]
      : tensor<4xf32> into tensor<8xf32>[!gml_st.tile<4>]
  } : tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @for_loop

func.func @dynamic_broadcast_in_dim(%arg: tensor<?x?xf32>,
                                    %dst: tensor<?x?x?xf32>) {
  %bcast = gml_st.dynamic_broadcast_in_dim
      ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>) {
        broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>
      }
  func.return
}
// CHECK-LABEL: func @dynamic_broadcast_in_dim

// -----

func.func @gather(%arg: tensor<100xf32>,
                                    %indices: tensor<42x1xi64>,
                                    %dst: tensor<42xf32>) -> tensor<42xf32> {
  %gather = gml_st.gather
      ins(%arg: tensor<100xf32>, %indices: tensor<42x1xi64>)
      outs(%dst: tensor<42xf32>)
  func.return %gather : tensor<42xf32>
}
// CHECK-LABEL: func @gather

// -----

func.func @scatter(%arg: tensor<3x3xf32>,
                   %indices: tensor<2x2xi64>,
                   %updates: tensor<3xf32>,
                   %dst: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %scatter = gml_st.scatter
      ins(%arg: tensor<3x3xf32>,
          %indices: tensor<2x2xi64>,
          %updates: tensor<3xf32>)
      outs(%dst: tensor<3x3xf32>)
  func.return %scatter : tensor<3x3xf32>
}
// CHECK-LABEL: func @scatter
