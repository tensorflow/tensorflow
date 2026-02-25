// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton \
// RUN: | FileCheck %s

// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton="allow_tma=1 num_stages=3" \
// RUN: | FileCheck %s --check-prefix=CHECK-TMA

func.func @lower_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<512x8x128xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 3, 0] [16, 1, 64] [1, 1, 1] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x16x256xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 5, 0] [16, 1, 64] [1, 1, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert(
// CHECK-SAME:      %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
// CHECK:         %[[LOAD:.*]] = tt.load
// CHECK:         tt.store {{.*}}, %[[LOAD]]
// CHECK:         tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert
// CHECK-TMA-SAME:      %arg0: !tt.tensordesc<tensor<16x1x64xbf16>>
// CHECK-TMA-SAME:      %arg1: !tt.tensordesc<tensor<16x1x64xbf16>>
// CHECK-TMA:         %[[LOAD:.*]] = tt.descriptor_load %arg0
// CHECK-TMA:         tt.descriptor_store %arg1[{{.*}}],
// CHECK-TMA:         tt.return

// -----

func.func @non_perfect_tile_shape(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
    as memref<300x300xbf16, #xtile.layout<[1, 0]>>
    [0, 0] [8, 8] [1, 1] : tensor<8x8xbf16>
  triton_xla.insert %extracted_tensor into %arg1
    as memref<300x300xbf16, #xtile.layout<[1, 0]>>
    [0, 0] [8, 8] [1, 1] : tensor<8x8xbf16>
  func.return
}

// CHECK-LABEL: tt.func @non_perfect_tile_shape
// CHECK:         %[[LOAD:.*]] = tt.load {{.*}}, %{{.*}}, %{{.*}} :
// CHECK:         tt.store {{.*}}, %[[LOAD]], %{{.*}} :

// -----

func.func @incompatible_tma_global_strides(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<234x234xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [128, 1] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<123x123xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [128, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_global_strides
// CHECK-TMA:         tt.load
// CHECK-TMA:         tt.store

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> (pid_0 * 32), domain: pid_0 in [0, 1]">
module {
  func.func @slice_with_tiling_that_needs_padding_has_boundary_checks(
          %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract from %arg0
        as memref<64xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    %4 = math.absf %extracted_tile : tensor<32xf32>
    %5 = arith.subf %cst, %4 : tensor<32xf32>
    triton_xla.insert %5 into %arg1 as memref<63xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    triton_xla.insert %4 into %arg2 as memref<63xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    func.return
  }
}

// CHECK-LABEL:   tt.func @slice_with_tiling_that_needs_padding_has_boundary_checks
// CHECK-COUNT-1: tt.load
// CHECK:         tt.store {{.*}}, %{{.*}}, %{{.*}}
// CHECK:         tt.store {{.*}}, %{{.*}}, %{{.*}}

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> (pid_0 * 32), domain: pid_0 in [0, 1]">
module {
  func.func @slice_with_extra_output_that_can_reuse_tile_due_to_padding(
            %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract from %arg0
        as memref<64xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    %4 = math.absf %extracted_tile : tensor<32xf32>
    %5 = arith.subf %cst, %4 : tensor<32xf32>
    triton_xla.insert %5 into %arg1 as memref<63xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    triton_xla.insert %4 into %arg2 as memref<64xf32, #xtile.layout<[0]>>
        [%3][32][1] : tensor<32xf32>
    func.return
  }
}

// CHECK-LABEL:   tt.func @slice_with_extra_output_that_can_reuse_tile_due_to_padding
// CHECK-COUNT-1: tt.load
// CHECK:         tt.store {{.*}}, %{{.*}}, %{{.*}}
// CHECK:         tt.store {{.*}}, %{{.*}} :

// -----

func.func @extract_with_non_unit_minor_dim_stride(%arg0: !tt.ptr<bf16>,
                          %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<1024x1024xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [2, 2] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-LABEL: tt.func @extract_with_non_unit_minor_dim_stride
// CHECK-TMA:   tt.load
// CHECK-TMA:   tt.descriptor_store

// -----

func.func @lower_extract_insert_1d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<128xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert_1d
// CHECK-SAME:      %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[LOAD:.*]] = tt.load
// CHECK:         tt.store {{.*}}, %[[LOAD]]
// CHECK:         tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert_1d
// CHECK-TMA-SAME:      %arg0: !tt.tensordesc<tensor<16xbf16>>
// CHECK-TMA-SAME:      %arg1: !tt.tensordesc<tensor<16xbf16>>
// CHECK-TMA:         %[[LOAD:.*]] = tt.descriptor_load %arg0
// CHECK-TMA:         tt.descriptor_store %arg1[{{.*}}], %[[LOAD]]
// CHECK-TMA:         tt.return

// -----

func.func @lower_extract_insert_5d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<16x16x16x16x16xbf16, #xtile.layout<[4, 3, 2, 1, 0]>>
      [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [1, 1, 1, 1, 1] : tensor<8x8x8x8x8xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<32x32x32x32x32xbf16, #xtile.layout<[4, 3, 2, 1, 0]>>
      [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [1, 1, 1, 1, 1] : tensor<8x8x8x8x8xbf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert_5d
// CHECK-SAME:      %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[LOAD:.*]] = tt.load
// CHECK:         tt.store {{.*}}, %[[LOAD]]
// CHECK:         tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert_5d
// CHECK-TMA-SAME:      %arg0: !tt.tensordesc<tensor<8x8x8x8x8xbf16>>
// CHECK-TMA-SAME:      %arg1: !tt.tensordesc<tensor<8x8x8x8x8xbf16>>
// CHECK-TMA:         %[[LOAD:.*]] = tt.descriptor_load %arg0
// CHECK-TMA:         tt.descriptor_store %arg1[{{.*}}], %[[LOAD]]
// CHECK-TMA:         tt.return

// -----

func.func @extract_insert_with_zero_stride(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<512x128xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [1, 64] [0, 1] : tensor<1x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [1, 64] [0, 1] : tensor<1x64xbf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @extract_insert_with_zero_stride
// CHECK-TMA-SAME:      %arg0: !tt.tensordesc<tensor<1x64xbf16>>
// CHECK-TMA-SAME:      %arg1: !tt.tensordesc<tensor<1x64xbf16>>

// -----

func.func @incompatible_tma_const_offset_not_divisible_by_16_bytes(
          %arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<512x128xbf16, #xtile.layout<[1, 0]>>
      [0, 15] [1, 64] [1, 1] : tensor<1x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [1, 64] [0, 1] : tensor<1x64xbf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_const_offset_not_divisible_by_16_bytes
// CHECK-TMA:         tt.load
// CHECK-TMA:         tt.descriptor_store

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 9) * 16 + (pid_0 floordiv 9) * 130), domain: pid_0 in [0, 575]">
#indexing_map1 = #xla.indexing_map<"(pid_0) -> (pid_0 floordiv 9), domain: pid_0 in [0, 575]">
#indexing_map2 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 9) * 16), domain: pid_0 in [0, 575]">
module {
  func.func @incompatible_tma_dynamic_offset_not_divisible_by_16_bytes(
            %arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract from %arg0
        as memref<16x16xbf16, #xtile.layout<[1, 0]>>
        [0, %3] [16, 16] [1, 1] : tensor<16x16xbf16>
    %4 = tt.reshape %extracted_tile : tensor<16x16xbf16> -> tensor<16x1x16xbf16>
    %5 = xla.apply_indexing #indexing_map1(%2)
    %6 = xla.apply_indexing #indexing_map2(%2)
    triton_xla.insert %4 into %arg1
        as memref<16x1x16xbf16, #xtile.layout<[2, 1, 0]>>
        [0, %5, %6] [16, 1, 16] [1, 1, 1] : tensor<16x1x16xbf16>
    func.return
  }
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_dynamic_offset_not_divisible_by_16_bytes
// CHECK-TMA:         tt.load
// CHECK-TMA:         tt.descriptor_store

// -----

func.func @parameter_into_broadcast_with_3_or_more_stages_does_not_use_tma(
          %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %extracted_tile = triton_xla.extract from %arg0 as
      memref<64xf32, #xtile.layout<[0]>> [0] [64] [1] : tensor<64xf32>
  %0 = tt.expand_dims %extracted_tile {axis = 1 : i32}
      : tensor<64xf32> -> tensor<64x1xf32>
  %1 = tt.broadcast %0 : tensor<64x1xf32> -> tensor<64x64xf32>
  %extracted_tile_0 = triton_xla.extract from %arg1 as
      memref<64x64xf32, #xtile.layout<[1, 0]>> [0, 0] [64, 64] [1, 1]
      : tensor<64x64xf32>
  %2 = tt.dot %1, %extracted_tile_0, %cst, inputPrecision = tf32
      : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
  triton_xla.insert %2 into %arg2 as
      memref<64x64xf32, #xtile.layout<[1, 0]>> [0, 0] [64, 64] [1, 1]
      : tensor<64x64xf32>
  return
}

// CHECK-TMA-LABEL: tt.func @parameter_into_broadcast_with_3_or_more_stages_does_not_use_tma
// CHECK-TMA-NOT:         tt.descriptor_load %arg0
// CHECK-TMA:             tt.descriptor_load %arg1

// -----

#indexing_map_unaligned = #xla.indexing_map<"(d0) -> (d0 * 2816), domain: d0 in [0, 2047]">
module {
  func.func @apply_mask_to_unaligned_offset_with_perfect_total_size(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map_unaligned(%2)
    // Total size 5767168 is divisible by 4096 (1408 * 4096)
    // But offset stride 2816 is not divisible by 4096.
    %extracted_tile = triton_xla.extract from %arg0
        as memref<5767168xf32, #xtile.layout<[0]>>
        [%3] [4096] [1] : tensor<4096xf32>
    triton_xla.insert %extracted_tile into %arg1
        as memref<5767168xf32, #xtile.layout<[0]>>
        [%3] [4096] [1] : tensor<4096xf32>
    func.return
  }
}

// CHECK-LABEL: tt.func @apply_mask_to_unaligned_offset_with_perfect_total_size
// CHECK: %[[MASK:.*]] = arith.cmpi slt
// CHECK: tt.load {{.*}}, %[[MASK]], {{.*}}
