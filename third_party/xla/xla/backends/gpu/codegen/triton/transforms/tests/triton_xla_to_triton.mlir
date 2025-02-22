// RUN: xla-opt %s -split-input-file -triton-xla-to-triton | FileCheck %s

// CHECK-LABEL: func @lower_tile_extract_insert
// CHECK-SAME: !tt.ptr<bf16>
// CHECK-SAME: !tt.ptr<bf16>
tt.func @lower_tile_extract_insert(%arg0: tensor<512x128xbf16>,
          %arg1: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %cst = arith.constant 1 : i32
  %tiled_tensor_in = triton_xla.tile %arg0 [0, 0] [16, 64] [128, 1]
    : !triton_xla.tiled_tensor<16x64|512x128xbf16>
  // CHECK: tt.make_tensor_ptr
  %tiled_tensor_out = triton_xla.tile %arg1 [0, 0] [16, 64] [128, 1]
    : !triton_xla.tiled_tensor<16x64|256x256xbf16>
  // CHECK: tt.make_tensor_ptr
  %extracted_tensor = triton_xla.extract %tiled_tensor_in [%cst, %cst]
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  // CHECK: tt.advance
  // CHECK: tt.load
  %updated_tensor = triton_xla.insert %extracted_tensor into %tiled_tensor_out [%cst, %cst]
  : tensor<16x64xbf16> into tensor<256x256xbf16>
  // CHECK: tt.advance
  // CHECK: tt.store
  tt.return %updated_tensor : tensor<256x256xbf16>
  // CHECK: tt.return
}