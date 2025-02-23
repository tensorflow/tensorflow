// RUN: xla-opt %s -split-input-file -triton-xla-extract-insert-to-triton | FileCheck %s
tt.func @lower_tile_extract_insert(%arg0: tensor<512x128xbf16>,
          %arg1: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %cst = arith.constant 1 : i32
  %tiled_tensor_in = triton_xla.tile %arg0 [0, 0] [16, 64] [128, 1]
    : !triton_xla.tiled_tensor<16x64|512x128xbf16>
  %tiled_tensor_out = triton_xla.tile %arg1 [0, 0] [16, 64] [128, 1]
    : !triton_xla.tiled_tensor<16x64|256x256xbf16>
  %extracted_tensor = triton_xla.extract %tiled_tensor_in [%cst, %cst]
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  %updated_tensor = triton_xla.insert %extracted_tensor into %tiled_tensor_out [%cst, %cst]
  : tensor<16x64xbf16> into tensor<256x256xbf16>
  tt.return %updated_tensor : tensor<256x256xbf16>
}

// CHECK-LABEL: func @lower_tile_extract_insert
// CHECK-SAME:  %[[ARG0:.*]]: !tt.ptr<bf16>, %[[ARG1:.*]]: !tt.ptr<bf16>
// CHECK:         %[[PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]]
// CHECK:         %[[PTR_1:.*]] = tt.make_tensor_ptr %[[ARG1]]
// CHECK:         %[[ADV_0:.*]] = tt.advance %[[PTR_0]]
// CHECK:         %[[LOAD:.*]] = tt.load %[[ADV_0]]
// CHECK:         %[[ADV_1:.*]] = tt.advance %[[PTR_1]]
// CHECK:         tt.store %[[ADV_1]], %[[LOAD]]
// CHECK:       tt.return