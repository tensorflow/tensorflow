// RUN: xla-opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: xla-opt %s --split-input-file | xla-opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: xla-opt %s --split-input-file --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s

tt.func @xla_triton_extract(%arg0: tensor<512x128xbf16>, %i : index)
    -> tensor<16x64xbf16> {
  %extracted_tensor = triton_xla.extract %arg0 [0, %i] [16, 64] [128, 1]
    {layout = array<i64:1, 0>} : tensor<512x128xbf16> to tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}
// CHECK-LABEL: xla_triton_extract
//       CHECK:   triton_xla.extract

// -----

tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: tensor<512x128xbf16>,
    %j: index) -> tensor<512x128xbf16> {
  %updated_tensor = triton_xla.insert %src into %dst [0, 0][16, 64][%j, 1]
    {layout = array<i64:1, 0>} : tensor<16x64xbf16> into tensor<512x128xbf16>
  tt.return %updated_tensor : tensor<512x128xbf16>
}
// CHECK-LABEL: xla_triton_insert
//       CHECK:   triton_xla.insert
