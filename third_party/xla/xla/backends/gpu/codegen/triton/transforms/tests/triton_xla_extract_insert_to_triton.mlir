// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton="gpu_device_info='cuda_compute_capability {major: 6}' tma_enabled=0" \
// RUN: | FileCheck %s

// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton="gpu_device_info='cuda_compute_capability {major: 9}' tma_enabled=1" \
// RUN: | FileCheck %s --check-prefix=CHECK-TMA

func.func @lower_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [16, 64] [1, 1]
    {shape = array<i64:512, 128>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [16, 64] [1, 1] {shape = array<i64:256, 256>, layout = array<i64:1, 0>}
    : tensor<16x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %[[ARG_1:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[ADDPTR_0:.*]] = tt.addptr %[[ARG_0]]
// CHECK:         %[[PTR_0:.*]] = tt.make_tensor_ptr %[[ADDPTR_0]]
// CHECK:         %[[LOAD:.*]] = tt.load %[[PTR_0]]
// CHECK:         %[[ADDPTR_1:.*]] = tt.addptr %[[ARG_1]]
// CHECK:         %[[PTR_1:.*]] = tt.make_tensor_ptr %[[ADDPTR_1]]
// CHECK:         tt.store %[[PTR_1]], %[[LOAD]]
// CHECK:       tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert
// CHECK-TMA-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<tensor<16x64xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [512, 128], tile_shape = [16, 64], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2>},
// CHECK-TMA-SAME:  %[[ARG_1:.*]]: !tt.tensordesc<tensor<16x64xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256, 256], tile_shape = [16, 64], tile_strides = [1, 1], layout = [1, 0], element_byte_size = 2>}
// CHECK-TMA:    %[[LOAD:.*]] = tt.descriptor_load %[[ARG_0]]
// CHECK-TMA:    tt.descriptor_store %[[ARG_1]][{{.*}}], %[[LOAD]]
// CHECK-TMA:    tt.return

// -----

func.func @non_perfect_tile_shape(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [8, 8] [1, 1]
    {shape = array<i64:300, 300>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<8x8xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [8, 8] [1, 1] {shape = array<i64:300, 300>, layout = array<i64:1, 0>}
    : tensor<8x8xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @non_perfect_tile_shape
// CHECK:        tt.load {{.*}} {
// CHECK-SAME:     boundaryCheck = array<i32: 0, 1>, padding = 1 : i32
// CHECK:        tt.store {{.*}}, {{.*}} {
// CHECK-SAME:     boundaryCheck = array<i32: 0, 1>

// -----

func.func @incompatible_tma_global_strides(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [16, 64] [128, 1]
    {shape = array<i64:234, 234>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [16, 64] [128, 1] {shape = array<i64:123, 123>, layout = array<i64:1, 0>}
    : tensor<16x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_global_strides
// CHECK-TMA:         tt.make_tensor_ptr
// CHECK-TMA:         tt.load
// CHECK-TMA:         tt.make_tensor_ptr
// CHECK-TMA:         tt.store

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> (pid_0 * 32), domain: pid_0 in [0, 1]">
module {
  func.func @slice_with_tiling_that_needs_padding_has_boundary_checks(
          %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_castui %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract %arg0[%3][32][1]
      {shape = array<i64:64>, layout = array<i64:0>} : !tt.ptr<f32> to tensor<32xf32>
    %4 = math.absf %extracted_tile : tensor<32xf32>
    %5 = arith.subf %cst, %4 : tensor<32xf32>
    triton_xla.insert %5 into %arg1[%3][32][1]
      {shape = array<i64:63>, layout = array<i64:0>} : tensor<32xf32> into !tt.ptr<f32>
    triton_xla.insert %4 into %arg2[%3][32][1]
      {shape = array<i64:63>, layout = array<i64:0>} : tensor<32xf32> into !tt.ptr<f32>
    func.return
  }
}

// CHECK-LABEL:   tt.func @slice_with_tiling_that_needs_padding_has_boundary_checks
// CHECK-COUNT-1: tt.load
// CHECK:         tt.store
// CHECK-SAME:    boundaryCheck = array<i32: 0>
// CHECK:         tt.store
// CHECK-SAME:    boundaryCheck = array<i32: 0>

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> (pid_0 * 32), domain: pid_0 in [0, 1]">
module {
  func.func @slice_with_extra_output_that_can_reuse_tile_due_to_padding(
            %arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_castui %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract %arg0[%3][32][1]
      {shape = array<i64:64>, layout = array<i64:0>} : !tt.ptr<f32> to tensor<32xf32>
    %4 = math.absf %extracted_tile : tensor<32xf32>
    %5 = arith.subf %cst, %4 : tensor<32xf32>
    triton_xla.insert %5 into %arg1[%3][32][1]
      {shape = array<i64:63>, layout = array<i64:0>} : tensor<32xf32> into !tt.ptr<f32>
    triton_xla.insert %4 into %arg2[%3][32][1]
      {shape = array<i64:64>, layout = array<i64:0>} : tensor<32xf32> into !tt.ptr<f32>
    func.return
  }
}

// CHECK-LABEL:   tt.func @slice_with_extra_output_that_can_reuse_tile_due_to_padding
// CHECK-COUNT-1: tt.load
// CHECK:         tt.store
// CHECK-SAME:    boundaryCheck = array<i32: 0>
// CHECK:         tt.store
// CHECK-NOT:     boundaryCheck = array<i32: 0>

// -----

func.func @extract_with_non_unit_minor_dim_stride(%arg0: !tt.ptr<bf16>,
                          %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [16, 64] [2, 2]
    {shape = array<i64:1024, 1024>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [16, 64] [1, 1] {shape = array<i64:256, 256>, layout = array<i64:1, 0>}
    : tensor<16x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @extract_with_non_unit_minor_dim_stride
// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.load
// CHECK-TMA:   tt.descriptor_store

// -----

func.func @extract_with_non_static_strides(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %0 = tt.get_program_id x : i32
  %1 = arith.extsi %0 : i32 to i64
  %2 = arith.index_castui %1 : i64 to index
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [16, 64] [%2, 1]
    {shape = array<i64:1024, 1024>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [16, 64] [1, 1] {shape = array<i64:256, 256>, layout = array<i64:1, 0>}
    : tensor<16x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @extract_with_non_static_strides
// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.load
// CHECK-TMA:   tt.descriptor_store

// -----

func.func @lower_extract_insert_1d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0] [16] [1]
    {shape = array<i64:128>, layout = array<i64:0>} : !tt.ptr<bf16> to tensor<16xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0] [16] [1] {shape = array<i64:256>, layout = array<i64:0>}
    : tensor<16xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert_1d
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %[[ARG_1:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[PTR_0:.*]] = tt.make_tensor_ptr %[[ARG_0]]
// CHECK:         %[[LOAD:.*]] = tt.load %[[PTR_0]]
// CHECK:         %[[PTR_1:.*]] = tt.make_tensor_ptr %[[ARG_1]]
// CHECK:         tt.store %[[PTR_1]], %[[LOAD]]
// CHECK:       tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert_1d
// CHECK-TMA-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<tensor<16xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [128], tile_shape = [16], tile_strides = [1], layout = [0], element_byte_size = 2>},
// CHECK-TMA-SAME:  %[[ARG_1:.*]]: !tt.tensordesc<tensor<16xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256], tile_shape = [16], tile_strides = [1], layout = [0], element_byte_size = 2>}
// CHECK-TMA:    %[[LOAD:.*]] = tt.descriptor_load %[[ARG_0]]
// CHECK-TMA:    tt.descriptor_store %[[ARG_1]][{{.*}}], %[[LOAD]]
// CHECK-TMA:    tt.return

// -----

func.func @lower_extract_insert_5d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract
                      %arg0 [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [1, 1, 1, 1, 1]
                      {shape = array<i64:16, 16, 16, 16, 16>, layout = array<i64:4, 3, 2, 1, 0>}
                      : !tt.ptr<bf16> to tensor<8x8x8x8x8xbf16>
  triton_xla.insert %extracted_tensor into
                    %arg1 [0, 0, 0, 0, 0] [8, 8, 8, 8, 8] [1, 1, 1, 1, 1]
                    {shape = array<i64:32, 32, 32, 32, 32>, layout = array<i64:4, 3, 2, 1, 0>}
                    : tensor<8x8x8x8x8xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-LABEL: tt.func @lower_extract_insert_5d
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %[[ARG_1:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[ADDPTR_0:.*]] = tt.addptr %[[ARG_0]]
// CHECK:         %[[PTR_0:.*]] = tt.make_tensor_ptr %[[ADDPTR_0]]
// CHECK:         %[[LOAD:.*]] = tt.load %[[PTR_0]]
// CHECK:         %[[ADDPTR_1:.*]] = tt.addptr %[[ARG_1]]
// CHECK:         %[[PTR_1:.*]] = tt.make_tensor_ptr %[[ADDPTR_1]]
// CHECK:         tt.store %[[PTR_1]], %[[LOAD]]
// CHECK:       tt.return

// CHECK-TMA-LABEL: tt.func @lower_extract_insert_5d
// CHECK-TMA-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<tensor<8x8x8x8x8xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [16, 16, 16, 16, 16], tile_shape = [8, 8, 8, 8, 8], tile_strides = [1, 1, 1, 1, 1], layout = [4, 3, 2, 1, 0], element_byte_size = 2>},
// CHECK-TMA-SAME:  %[[ARG_1:.*]]: !tt.tensordesc<tensor<8x8x8x8x8xbf16>> {tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 32, 32, 32, 32], tile_shape = [8, 8, 8, 8, 8], tile_strides = [1, 1, 1, 1, 1], layout = [4, 3, 2, 1, 0], element_byte_size = 2>}
// CHECK-TMA:    %[[LOAD:.*]] = tt.descriptor_load %[[ARG_0]]
// CHECK-TMA:    tt.descriptor_store %[[ARG_1]][{{.*}}], %[[LOAD]]
// CHECK-TMA:    tt.return

// -----

func.func @extract_insert_with_zero_stride(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 0] [1, 64] [0, 1]
    {shape = array<i64:512, 128>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<1x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 0] [1, 64] [0, 1] {shape = array<i64:256, 256>, layout = array<i64:1, 0>}
    : tensor<1x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @extract_insert_with_zero_stride
// CHECK-TMA-SAME:  %[[ARG_0:.*]]: !tt.tensordesc{{.*}} tile_strides = [1, 1], {{.*}} %[[ARG_1:.*]]: !tt.tensordesc{{.*}} tile_strides = [1, 1]

// -----

func.func @incompatible_tma_const_offset_not_divisible_by_16_bytes(
          %arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract %arg0 [0, 15] [1, 64] [1, 1]
    {shape = array<i64:512, 128>, layout = array<i64:1, 0>} : !tt.ptr<bf16> to tensor<1x64xbf16>
  triton_xla.insert %extracted_tensor into
    %arg1 [0, 16] [1, 64] [1, 1] {shape = array<i64:256, 256>, layout = array<i64:1, 0>}
    : tensor<1x64xbf16> into !tt.ptr<bf16>
  func.return
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_const_offset_not_divisible_by_16_bytes
// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.load
// CHECK-TMA:   tt.descriptor_store

// -----

#indexing_map = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 9) * 16 + (pid_0 floordiv 9) * 130), domain: pid_0 in [0, 575]">
#indexing_map1 = #xla.indexing_map<"(pid_0) -> (pid_0 floordiv 9), domain: pid_0 in [0, 575]">
#indexing_map2 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 9) * 16), domain: pid_0 in [0, 575]">
module {
  func.func @incompatible_tma_dynamic_offset_not_divisible_by_16_bytes(
            %arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_castui %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %extracted_tile = triton_xla.extract %arg0[0, %3] [16, 16] [1, 1]
      {shape = array<i64:4, 8320>, layout = array<i64: 1, 0>} : !tt.ptr<bf16> to tensor<16x16xbf16>
    %4 = tt.reshape %extracted_tile : tensor<16x16xbf16> -> tensor<16x1x16xbf16>
    %5 = xla.apply_indexing #indexing_map1(%2)
    %6 = xla.apply_indexing #indexing_map2(%2)
    triton_xla.insert %4 into %arg1[0, %5, %6] [16, 1, 16] [1, 1, 1]
      {shape = array<i64: 4, 64, 130>, layout = array<i64: 2, 1, 0>}
      : tensor<16x1x16xbf16> into !tt.ptr<bf16>
    func.return
  }
}

// CHECK-TMA-LABEL: tt.func @incompatible_tma_dynamic_offset_not_divisible_by_16_bytes
// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.load
