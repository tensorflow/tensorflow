// RUN: xla-opt %s | FileCheck %s

// Verify the printed output can be parsed.
// RUN: xla-opt %s | xla-opt --split-input-file | FileCheck %s

// Verify the generic form can be parsed.
// RUN: xla-opt %s --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @xla_triton_extract
tt.func @xla_triton_extract(%arg0: !tt.ptr<bf16>, %i : index) -> tensor<16x64xbf16> {
  // CHECK: triton_xla.extract
  %extracted_tensor = triton_xla.extract %arg0 [0, %i] [16, 64] [128, 1]
    {shape = array<i64:512, 128>, layout = array<i64:1, 0>}
    : !tt.ptr<bf16> to tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}

// CHECK-LABEL: @xla_triton_insert
tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: !tt.ptr<bf16>, %j: index) {
  // CHECK: triton_xla.insert
  triton_xla.insert %src into %dst [0, 0][16, 64][%j, 1]
    {shape = array<i64:512, 128>, layout = array<i64:1, 0>}
    : tensor<16x64xbf16> into !tt.ptr<bf16>
  tt.return
}
