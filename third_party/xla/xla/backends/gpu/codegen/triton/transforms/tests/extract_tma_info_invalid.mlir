// RUN: xla-opt %s -split-input-file \
// RUN: -extract-tma-info \
// RUN: --verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
tt.func @extract_tma_info_no_tma_descriptor(
// expected-error @+1 {{Argument of type tt.tensordesc must have attribute tt.tma_descriptor}}
  %arg0: !tt.tensordesc<tensor<16x32xf32, #shared>>
  {tt.nv_tma_desc = 1 : i32}) {
  tt.return
}
