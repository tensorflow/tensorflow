// RUN: xla-opt %s | FileCheck %s

// Verify the printed output can be parsed.
// RUN: xla-opt %s | xla-opt --split-input-file | FileCheck %s

// Verify the generic form can be parsed.
// RUN: xla-opt %s --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @xla_triton_extract
tt.func @xla_triton_extract(%src: !tt.ptr<bf16>, %i : index) -> tensor<16x64xbf16> {
  // CHECK: triton_xla.extract
  %extracted_tensor = triton_xla.extract from %src
    as memref<512x1x128xbf16, #xtile.layout<[2, 1, 0]>>
    [0, 0, %i] [16, 1, 64] [128, 1, 1] : tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}

// CHECK-LABEL: @xla_triton_insert
tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: !tt.ptr<bf16>, %j: index) {
  // CHECK: triton_xla.insert
  triton_xla.insert %src into %dst
    as memref<512x128xbf16, #xtile.layout<[0, 1]>>
    [%j, 0][16, 64][1, 1] : tensor<16x64xbf16>
  tt.return
}
