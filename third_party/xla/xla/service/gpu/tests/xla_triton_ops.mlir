// RUN: xla-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @xla_tiled_tensor_type
tt.func @xla_tiled_tensor_type(%arg0: !triton_xla.tiled_tensor<16x64xbf16>)
    -> !triton_xla.tiled_tensor<16x64xbf16> {
  // CHECK: !triton_xla.tiled_tensor<16x64xbf16>
  tt.return %arg0 : !triton_xla.tiled_tensor<16x64xbf16>
}

// -----

// CHECK-LABEL: xla_triton_tile
tt.func @xla_triton_tile(%arg0: tensor<120x320xbf16>)
    -> !triton_xla.tiled_tensor<16x64xbf16> {
  // CHECK: triton_xla.tile
  %tiled_tensor = triton_xla.tile %arg0 [0, 0] [1, 1] [16, 64]
    : tensor<120x320xbf16> -> !triton_xla.tiled_tensor<16x64xbf16>
  tt.return %tiled_tensor : !triton_xla.tiled_tensor<16x64xbf16>
}

// -----

// CHECK-LABEL: xla_triton_extract
tt.func @xla_triton_extract(%arg0: !triton_xla.tiled_tensor<16x64xbf16>)
    -> tensor<16x64xbf16> {
  // CHECK: triton_xla.extract
  %cst = arith.constant 0 : index
  %extracted_tensor = triton_xla.extract %arg0 [%cst, %cst]
    : !triton_xla.tiled_tensor<16x64xbf16> -> tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}

// -----

// CHECK-LABEL: xla_triton_insert
tt.func @xla_triton_insert(%arg0: tensor<16x64xbf16>,
                %arg1: !triton_xla.tiled_tensor<16x64xbf16>)
                -> tensor<16x64xbf16> {
  // CHECK: triton_xla.insert
  %cst = arith.constant 0 : index
  %inserted_tensor = triton_xla.insert %arg0 into %arg1 [%cst, %cst]
  : tensor<16x64xbf16> into !triton_xla.tiled_tensor<16x64xbf16>
    -> tensor<16x64xbf16>
  tt.return %inserted_tensor : tensor<16x64xbf16>
}