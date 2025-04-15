// RUN: xla-opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: xla-opt %s --split-input-file | xla-opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: xla-opt %s --split-input-file --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s


tt.func @xla_triton_tile(%arg0: tensor<512x128xbf16>)
    -> !triton_xla.tiled_tensor<16x64|512x128xbf16> {
  %cst_0 = arith.constant 0 : index
  %cst_1 = arith.constant 1 : index
  %cst_128 = arith.constant 128 : index
  %tiled_tensor = triton_xla.tile %arg0 [%cst_0, %cst_0] [%cst_128, %cst_1]
    {layout = array<i64:1, 0>} : !triton_xla.tiled_tensor<16x64|512x128xbf16>
  tt.return %tiled_tensor : !triton_xla.tiled_tensor<16x64|512x128xbf16>
}
// CHECK-LABEL: xla_triton_tile
//       CHECK:   triton_xla.tile

// -----

tt.func @xla_triton_extract(%arg0: !triton_xla.tiled_tensor<16x64|512x128xbf16>)
    -> tensor<16x64xbf16> {
  %cst = arith.constant 0 : index
  %extracted_tensor = triton_xla.extract %arg0 [%cst, %cst]
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}
// CHECK-LABEL: xla_triton_extract
//       CHECK:   triton_xla.extract

// -----

tt.func @xla_triton_insert(%src: tensor<16x64xbf16>,
    %dst: !triton_xla.tiled_tensor<16x64|512x128xbf16>) -> tensor<512x128xbf16> {
  %cst = arith.constant 0 : index
  %updated_tensor = triton_xla.insert %src into %dst [%cst, %cst]
  : tensor<16x64xbf16> into tensor<512x128xbf16>
  tt.return %updated_tensor : tensor<512x128xbf16>
}
// CHECK-LABEL: xla_triton_insert
//       CHECK:   triton_xla.insert

