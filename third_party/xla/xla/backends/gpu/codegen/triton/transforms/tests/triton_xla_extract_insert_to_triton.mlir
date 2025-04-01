// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton="gpu_device_info='cuda_compute_capability {major: 6}' tma_enabled=0" \
// RUN: | FileCheck %s

// RUN: xla-opt %s -split-input-file \
// RUN: -triton-xla-extract-insert-to-triton="gpu_device_info='cuda_compute_capability {major: 9}' tma_enabled=1" \
// RUN: | FileCheck %s --check-prefix=CHECK-TMA

func.func @lower_tile_extract_insert(%arg0: tensor<512x128xbf16>,
          %arg1: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %c = arith.constant 1 : i32
  %c_0 = arith.constant 0 : i32
  %c_1 = arith.constant 1 : i64
  %c_16 = arith.constant 16 : i64
  %c_64 = arith.constant 64 : i64
  %c_128 = arith.constant 128 : i64
  %tiled_tensor_in = triton_xla.tile %arg0 [%c_0, %c_0] [%c_16, %c_64] [%c_128, %c_1]
    : !triton_xla.tiled_tensor<16x64|512x128xbf16>
  %tiled_tensor_out = triton_xla.tile %arg1 [%c_0, %c_0] [%c_16, %c_64] [%c_128, %c_1]
    : !triton_xla.tiled_tensor<16x64|256x256xbf16>
  %extracted_tensor = triton_xla.extract %tiled_tensor_in [%c, %c]
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  %updated_tensor = triton_xla.insert %extracted_tensor into
    %tiled_tensor_out [%c, %c]
    : tensor<16x64xbf16> into tensor<256x256xbf16>
  func.return %updated_tensor : tensor<256x256xbf16>
}

// CHECK-LABEL: tt.func @lower_tile_extract_insert
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %[[ARG_1:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32}
// CHECK:         %[[PTR_0:.*]] = tt.make_tensor_ptr %[[ARG_0]]
// CHECK:         %[[PTR_1:.*]] = tt.make_tensor_ptr %[[ARG_1]]
// CHECK:         %[[ADV_0:.*]] = tt.advance %[[PTR_0]]
// CHECK:         %[[LOAD:.*]] = tt.load %[[ADV_0]]
// CHECK:         %[[ADV_1:.*]] = tt.advance %[[PTR_1]]
// CHECK:         tt.store %[[ADV_1]], %[[LOAD]]
// CHECK:       tt.return

// CHECK-TMA-LABEL:tt.func @lower_tile_extract_insert
// CHECK-TMA-SAME:  %[[ARG_0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [512, 128], block_shape = [16, 64], element_byte_size = 2>},
// CHECK-TMA-SAME:  %[[ARG_1:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.nv_tma_desc = 1 : i32, tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [256, 256], block_shape = [16, 64], element_byte_size = 2>}
// CHECK-TMA:    %[[DESC_0:.*]] = tt.reinterpret_tensor_descriptor %[[ARG_0]]
// CHECK-TMA:    %[[DESC_1:.*]] = tt.reinterpret_tensor_descriptor %[[ARG_1]]
// CHECK-TMA:    %[[LOAD:.*]] = tt.experimental_descriptor_load %[[DESC_0]]
// CHECK-TMA:    tt.experimental_descriptor_store %[[DESC_1]][{{.*}}], %[[LOAD]]
// CHECK-TMA:    tt.return

// -----

func.func @non_perfect_tile_shape(
                %arg0: tensor<300x300xbf16>, %arg1: tensor<300x300xbf16>)
                -> tensor<300x300xbf16> {
  %c_0 = arith.constant 0 : i32
  %c_1 = arith.constant 1 : i64
  %c_8 = arith.constant 8 : i64
  %tiled_tensor_in = triton_xla.tile %arg0 [%c_0, %c_0] [%c_8, %c_8] [%c_1, %c_1]
    : !triton_xla.tiled_tensor<8x8|300x300xbf16>
  %tiled_tensor_out = triton_xla.tile %arg1 [%c_0, %c_0] [%c_8, %c_8] [%c_1, %c_1]
    : !triton_xla.tiled_tensor<8x8|300x300xbf16>
  %extracted_tensor = triton_xla.extract %tiled_tensor_in [%c_0, %c_0]
    : tensor<300x300xbf16> to tensor<8x8xbf16>
  %updated_tensor = triton_xla.insert %extracted_tensor into
    %tiled_tensor_out [%c_0, %c_0] : tensor<8x8xbf16> into tensor<300x300xbf16>
  func.return %updated_tensor : tensor<300x300xbf16>
}

// CHECK-LABEL: tt.func @non_perfect_tile_shape
// CHECK:        tt.load {{.*}} {
// CHECK-SAME:     boundaryCheck = array<i32: 0, 1>, padding = 1 : i32
// CHECK:        tt.store {{.*}}, {{.*}} {
// CHECK-SAME:     boundaryCheck = array<i32: 0, 1>

// -----

func.func @incompatible_tma_shapes(%arg0: tensor<1000x1000xbf16>,
          %arg1: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
  %c_0 = arith.constant 1 : i32
  %c_1 = arith.constant 1 : i64
  %c_512 = arith.constant 512 : i64
  %c_256 = arith.constant 256 : i64
  %c_128 = arith.constant 128 : i64
  %tiled_tensor_in = triton_xla.tile %arg0 [%c_0, %c_0] [%c_512, %c_256] [%c_128, %c_1]
    : !triton_xla.tiled_tensor<512x256|1000x1000xbf16>
  %tiled_tensor_out = triton_xla.tile %arg1 [%c_0, %c_0] [%c_512, %c_256] [%c_128, %c_1]
    : !triton_xla.tiled_tensor<512x256|1024x1024xbf16>
  %extracted_tensor = triton_xla.extract %tiled_tensor_in [%c_0, %c_0]
    : tensor<1000x1000xbf16> to tensor<512x256xbf16>
  %updated_tensor = triton_xla.insert %extracted_tensor into
    %tiled_tensor_out [%c_0, %c_0]
    : tensor<512x256xbf16> into tensor<1024x1024xbf16>
  func.return %updated_tensor : tensor<1024x1024xbf16>
}

// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.make_tensor_ptr
// CHECK-TMA:   tt.advance
// CHECK-TMA:   tt.load
// CHECK-TMA:   tt.advance
// CHECK-TMA:   tt.store
