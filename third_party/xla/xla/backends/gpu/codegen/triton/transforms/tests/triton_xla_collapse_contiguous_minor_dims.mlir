// RUN: xla-opt %s --triton-xla-collapse-contiguous-minor-dims --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @positive_extract
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<129x1248x4436xf32, #xtile.layout<[2, 1, 0]>>
// CHECK-SAME:       [0, 0, 0] [1, 2, 8192] [1, 1, 1] : tensor<2x8192xf32>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<2x8192xf32> -> tensor<2x4096x2xf32>
// CHECK:        return %[[RESHAPE]] : tensor<2x4096x2xf32>
func.func @positive_extract(%arg0: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @positive_insert
// CHECK-SAME: (%[[SRC:.*]]: tensor<2x4096x2xf32>, %[[DST:.*]]: !tt.ptr<f32>) {
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[SRC]] : tensor<2x4096x2xf32> -> tensor<2x8192xf32>
// CHECK:        triton_xla.insert %[[RESHAPE]] into %[[DST]]
// CHECK-SAME:       as memref<129x1248x4436xf32, #xtile.layout<[2, 1, 0]>>
// CHECK-SAME:       [0, 0, 0] [1, 2, 8192] [1, 1, 1] : tensor<2x8192xf32>
// CHECK:        return
func.func @positive_insert(%src: tensor<2x4096x2xf32>, %dst: !tt.ptr<f32>) {
  triton_xla.insert %src into %dst
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @minor_offset_nonzero
// CHECK:        triton_xla.extract from %arg0 as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>> [0, 0, 0, 1] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
// CHECK-NOT:    tt.reshape
func.func @minor_offset_nonzero(%arg0: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 1] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @partial_minor_tile
// CHECK:        triton_xla.extract from %arg0 as memref<129x1248x2218x4xf32, #xtile.layout<[3, 2, 1, 0]>> [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
// CHECK-NOT:    tt.reshape
func.func @partial_minor_tile(%arg0: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x4xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @minor_tile_already_16b
// CHECK:        triton_xla.extract from %arg0 as memref<129x1248x2218x4xf32, #xtile.layout<[3, 2, 1, 0]>> [0, 0, 0, 0] [1, 2, 4096, 4] [1, 1, 1, 1] : tensor<2x4096x4xf32>
// CHECK-NOT:    tt.reshape
func.func @minor_tile_already_16b(%arg0: !tt.ptr<f32>) -> tensor<2x4096x4xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x4xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 4] [1, 1, 1, 1] : tensor<2x4096x4xf32>
  return %0 : tensor<2x4096x4xf32>
}

// -----

// CHECK-LABEL: func.func @stride_not_one
// CHECK:        triton_xla.extract from %arg0 as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>> [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 2] : tensor<2x4096x2xf32>
// CHECK-NOT:    tt.reshape
func.func @stride_not_one(%arg0: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 1, 2] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @dynamic_offset_m1
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>, %[[IDX:.*]]: index) -> tensor<2x4096x2xf32> {
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[MUL:.*]] = arith.muli %[[IDX]], %[[C2]] : index
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<129x1248x4436xf32, #xtile.layout<[2, 1, 0]>>
// CHECK-SAME:       [0, 0, %[[MUL]]] [1, 2, 8192] [1, 1, 1] : tensor<2x8192xf32>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<2x8192xf32> -> tensor<2x4096x2xf32>
// CHECK:        return %[[RESHAPE]] : tensor<2x4096x2xf32>
func.func @dynamic_offset_m1(%arg0: !tt.ptr<f32>, %idx: index) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, %idx, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @multi_step_collapse_all_unit
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<1xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<4xf32, #xtile.layout<[0]>>
// CHECK-SAME:       [0] [1] [1] : tensor<1xf32>
// CHECK:        return %[[EXTRACT]] : tensor<1xf32>
func.func @multi_step_collapse_all_unit(%arg0: !tt.ptr<f32>) -> tensor<1xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<4x1x1xf32, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @multi_step_collapse_i1
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<i1>) -> tensor<2x2x2x2xi1> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<64xi1, #xtile.layout<[0]>>
// CHECK-SAME:       [0] [16] [1] : tensor<16xi1>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<16xi1> -> tensor<2x2x2x2xi1>
// CHECK:        return %[[RESHAPE]] : tensor<2x2x2x2xi1>
func.func @multi_step_collapse_i1(%arg0: !tt.ptr<i1>) -> tensor<2x2x2x2xi1> {
  %0 = triton_xla.extract from %arg0
      as memref<8x2x2x2xi1, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [2, 2, 2, 2] [1, 1, 1, 1] : tensor<2x2x2x2xi1>
  return %0 : tensor<2x2x2x2xi1>
}

// -----

// CHECK-LABEL: func.func @padded_m1
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<4x4x2xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<7x6xf32, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [0, 0] [4, 8] [1, 1] : tensor<4x8xf32>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<4x8xf32> -> tensor<4x4x2xf32>
// CHECK:        return %[[RESHAPE]] : tensor<4x4x2xf32>
func.func @padded_m1(%arg0: !tt.ptr<f32>) -> tensor<4x4x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<7x3x2xf32, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [4, 4, 2] [1, 1, 1] : tensor<4x4x2xf32>
  return %0 : tensor<4x4x2xf32>
}

// -----

// CHECK-LABEL: func.func @stride_m1_not_one
// CHECK:        triton_xla.extract from %arg0 as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>> [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 2, 1] : tensor<2x4096x2xf32>
// CHECK-NOT:    tt.reshape
func.func @stride_m1_not_one(%arg0: !tt.ptr<f32>) -> tensor<2x4096x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, 0, 0] [1, 2, 4096, 2] [1, 1, 2, 1] : tensor<2x4096x2xf32>
  return %0 : tensor<2x4096x2xf32>
}

// -----

// CHECK-LABEL: func.func @non_default_layout
// CHECK:        triton_xla.extract from %arg0 as memref<4x2xf32, #xtile.layout<[0, 1]>> [0, 0] [4, 2] [1, 1] : tensor<4x2xf32>
// CHECK-NOT:    tt.reshape
func.func @non_default_layout(%arg0: !tt.ptr<f32>) -> tensor<4x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<4x2xf32, #xtile.layout<[0, 1]>>
      [0, 0] [4, 2] [1, 1] : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduced_tile
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<4x2xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<16xf32, #xtile.layout<[0]>>
// CHECK-SAME:       [0] [8] [1] : tensor<8xf32>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<8xf32> -> tensor<4x2xf32>
// CHECK:        return %[[RESHAPE]] : tensor<4x2xf32>
func.func @rank_reduced_tile(%arg0: !tt.ptr<f32>) -> tensor<4x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<8x1x2xf32, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [4, 1, 2] [1, 1, 1] : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduced_minor_dim
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<4xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<8xf32, #xtile.layout<[0]>>
// CHECK-SAME:       [0] [4] [1] : tensor<4xf32>
// CHECK-NOT:    tt.reshape
// CHECK:        return %[[EXTRACT]] : tensor<4xf32>
func.func @rank_reduced_minor_dim(%arg0: !tt.ptr<f32>) -> tensor<4xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<8x1xf32, #xtile.layout<[1, 0]>>
      [0, 0] [4, 1] [1, 1] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @insert_dynamic_offset
// CHECK-SAME: (%[[SRC:.*]]: tensor<2x4096x2xf32>, %[[DST:.*]]: !tt.ptr<f32>, %[[IDX:.*]]: index) {
// CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[MUL:.*]] = arith.muli %[[IDX]], %[[C2]] : index
// CHECK-DAG:    %[[RESHAPE:.*]] = tt.reshape %[[SRC]] : tensor<2x4096x2xf32> -> tensor<2x8192xf32>
// CHECK:        triton_xla.insert %[[RESHAPE]] into %[[DST]]
// CHECK-SAME:       as memref<129x1248x4436xf32, #xtile.layout<[2, 1, 0]>>
// CHECK-SAME:       [0, 0, %[[MUL]]] [1, 2, 8192] [1, 1, 1] : tensor<2x8192xf32>
// CHECK:        return
func.func @insert_dynamic_offset(%src: tensor<2x4096x2xf32>, %dst: !tt.ptr<f32>, %idx: index) {
  triton_xla.insert %src into %dst
      as memref<129x1248x2218x2xf32, #xtile.layout<[3, 2, 1, 0]>>
      [0, 0, %idx, 0] [1, 2, 4096, 2] [1, 1, 1, 1] : tensor<2x4096x2xf32>
  return
}

// -----

// CHECK-LABEL: func.func @bf16_collapse
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<bf16>) -> tensor<64x64x2xbf16> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<128x128xbf16, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [0, 0] [64, 128] [1, 1] : tensor<64x128xbf16>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<64x128xbf16> -> tensor<64x64x2xbf16>
// CHECK:        return %[[RESHAPE]] : tensor<64x64x2xbf16>
func.func @bf16_collapse(%arg0: !tt.ptr<bf16>) -> tensor<64x64x2xbf16> {
  %0 = triton_xla.extract from %arg0
      as memref<128x64x2xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [64, 64, 2] [1, 1, 1] : tensor<64x64x2xbf16>
  return %0 : tensor<64x64x2xbf16>
}

// -----

// CHECK-LABEL: func.func @overflow_merged_dim_unchanged
// CHECK:        triton_xla.extract from %arg0 as memref<1610612736x2xf32, #xtile.layout<[1, 0]>> [0, 0] [64, 2] [1, 1] : tensor<64x2xf32>
// CHECK-NOT:    tt.reshape
func.func @overflow_merged_dim_unchanged(%arg0: !tt.ptr<f32>) -> tensor<64x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<1610612736x2xf32, #xtile.layout<[1, 0]>>
      [0, 0] [64, 2] [1, 1] : tensor<64x2xf32>
  return %0 : tensor<64x2xf32>
}

// -----

// CHECK-LABEL: func.func @overflow_constant_offset_unchanged
// CHECK:        triton_xla.extract from %arg0 as memref<1610612736x2xf32, #xtile.layout<[1, 0]>> [1342177280, 0] [64, 2] [1, 1] : tensor<64x2xf32>
// CHECK-NOT:    tt.reshape
func.func @overflow_constant_offset_unchanged(%arg0: !tt.ptr<f32>) -> tensor<64x2xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<1610612736x2xf32, #xtile.layout<[1, 0]>>
      [1342177280, 0] [64, 2] [1, 1] : tensor<64x2xf32>
  return %0 : tensor<64x2xf32>
}

// -----

// CHECK:        #[[$MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 128), domain: d0 in [0, 15]">
// CHECK-LABEL:  func.func @apply_indexing_offset
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<f32>, %[[PID:.*]]: index) -> tensor<64x2xf32> {
// CHECK-NOT:   arith.muli
// CHECK:        %[[OFFSET:.*]] = xla.apply_indexing #[[$MAP]](%[[PID]])
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<2048xf32, #xtile.layout<[0]>>
// CHECK-SAME:       [%[[OFFSET]]] [128] [1] : tensor<128xf32>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<128xf32> -> tensor<64x2xf32>
// CHECK:        return %[[RESHAPE]] : tensor<64x2xf32>
#map = #xla.indexing_map<"(d0) -> (d0 * 64), domain: d0 in [0, 15]">
func.func @apply_indexing_offset(%arg0: !tt.ptr<f32>, %pid: index) -> tensor<64x2xf32> {
  %0 = xla.apply_indexing #map(%pid)
  %1 = triton_xla.extract from %arg0
      as memref<1024x2xf32, #xtile.layout<[1, 0]>>
      [%0, 0] [64, 2] [1, 1] : tensor<64x2xf32>
  return %1 : tensor<64x2xf32>
}

// -----

// CHECK-LABEL: func.func @subbyte_i4_extract
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<i4>) -> tensor<64x16xi4> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<16384xi4, #xtile.layout<[0]>>
// CHECK-SAME:       [0] [1024] [1] : tensor<1024xi4>
// CHECK:        %[[RESHAPE:.*]] = tt.reshape %[[EXTRACT]] : tensor<1024xi4> -> tensor<64x16xi4>
// CHECK:        return %[[RESHAPE]] : tensor<64x16xi4>
func.func @subbyte_i4_extract(%arg0: !tt.ptr<i4>) -> tensor<64x16xi4> {
  %0 = triton_xla.extract from %arg0
      as memref<1024x16xi4, #xtile.layout<[1, 0]>>
      [0, 0] [64, 16] [1, 1] : tensor<64x16xi4>
  return %0 : tensor<64x16xi4>
}

// -----

// CHECK-LABEL: func.func @negative_offset_unchanged
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<1x2xf32> {
// CHECK:        %[[NEG:.*]] = arith.constant -1 : index
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<4x2xf32, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [%[[NEG]], 0] [1, 2] [1, 1] : tensor<1x2xf32>
// CHECK:        return %[[EXTRACT]] : tensor<1x2xf32>
func.func @negative_offset_unchanged(%arg0: !tt.ptr<f32>) -> tensor<1x2xf32> {
  %c_neg = arith.constant -1 : index
  %0 = triton_xla.extract from %arg0
      as memref<4x2xf32, #xtile.layout<[1, 0]>>
      [%c_neg, 0] [1, 2] [1, 1] : tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func.func @apply_indexing_overflow_unchanged
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<f32>, %[[PID:.*]]: index) -> tensor<64x2xf32> {
// CHECK:        %[[OFFSET:.*]] = xla.apply_indexing #indexing_map{{.*}}(%[[PID]])
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<1000000000x2xf32, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [%[[OFFSET]], 0] [64, 2] [1, 1] : tensor<64x2xf32>
// CHECK:        return %[[EXTRACT]] : tensor<64x2xf32>
#map_overflow = #xla.indexing_map<"(d0) -> (d0 * 600000000), domain: d0 in [0, 5]">
func.func @apply_indexing_overflow_unchanged(%arg0: !tt.ptr<f32>, %pid: index) -> tensor<64x2xf32> {
  %0 = xla.apply_indexing #map_overflow(%pid)
  %1 = triton_xla.extract from %arg0
      as memref<1000000000x2xf32, #xtile.layout<[1, 0]>>
      [%0, 0] [64, 2] [1, 1] : tensor<64x2xf32>
  return %1 : tensor<64x2xf32>
}

// -----

// CHECK-LABEL: func.func @non_power_of_two_tile_unchanged
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<1x3xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<4x3xf32, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [0, 0] [1, 3] [1, 1] : tensor<1x3xf32>
// CHECK:        return %[[EXTRACT]] : tensor<1x3xf32>
func.func @non_power_of_two_tile_unchanged(%arg0: !tt.ptr<f32>) -> tensor<1x3xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<4x3xf32, #xtile.layout<[1, 0]>>
      [0, 0] [1, 3] [1, 1] : tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduced_collapsed_minor_unit_dims
// CHECK-SAME:  (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<4xf32> {
// CHECK:        %[[EXTRACT:.*]] = triton_xla.extract from %[[ARG0]]
// CHECK-SAME:       as memref<8x1xf32, #xtile.layout<[1, 0]>>
// CHECK-SAME:       [0, 0] [4, 1] [2, 1] : tensor<4xf32>
// CHECK-NOT:    tt.reshape
// CHECK:        return %[[EXTRACT]] : tensor<4xf32>
func.func @rank_reduced_collapsed_minor_unit_dims(%arg0: !tt.ptr<f32>) -> tensor<4xf32> {
  %0 = triton_xla.extract from %arg0
      as memref<8x1x1xf32, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [4, 1, 1] [2, 1, 1] : tensor<4xf32>
  return %0 : tensor<4xf32>
}
