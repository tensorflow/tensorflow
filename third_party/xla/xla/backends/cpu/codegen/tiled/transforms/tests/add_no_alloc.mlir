// RUN: fusion_compiler_opt %s --xtile-cpu-shlo-to-vector --xtile-cpu-elementwise-to-vector --xtile-cpu-bufferization | FileCheck %s

module @add_tranpose {
  xtile.entry_func @add_tranpose(
      %input: memref<4096x4096xf32>,
      %output: memref<4096x4096xf32>,
      %tile_id: index) attributes {xtile.tiling_info = #xtile.tiling_info<tile_count:262144, tiles_per_workgroup:32768>} {
    %offset_0 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid mod 512) * 8), domain: tid in [0, 262144]">(%tile_id)
    %offset_1 = xla.apply_indexing #xla.indexing_map<"(tid) -> ((tid floordiv 512) * 8), domain: tid in [0, 262144]">(%tile_id)
    %input_tile_0 = xtile.extract %input[%offset_0, %offset_1][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
    %input_tile_1 = xtile.extract %input[%offset_1, %offset_0][8, 8][1, 1] : memref<4096x4096xf32> -> tensor<8x8xf32>
    %transposed_tile = stablehlo.transpose %input_tile_0, dims = [1, 0] : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %added_tile = arith.addf %input_tile_1, %transposed_tile : tensor<8x8xf32>
    xtile.insert %added_tile into %output[%offset_1, %offset_0][8, 8][1, 1] : tensor<8x8xf32> -> memref<4096x4096xf32>
    xtile.return
  }
}

// CHECK-LABEL: @add_tranpose
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.alloca
