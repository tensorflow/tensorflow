// RUN: xla-opt %s --triton-xla-pipeline='target=gfx1250' \
// RUN:   | FileCheck %s --check-prefix=CHECK-TDM
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOTDM

// Verifies that the full Triton XLA + AMD lowering pipeline emits TDM
// intrinsics on gfx1250 and pointer-arithmetic buffer ops on non-TDM arches.

func.func @lower_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @lower_extract_insert
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @lower_extract_insert
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load
