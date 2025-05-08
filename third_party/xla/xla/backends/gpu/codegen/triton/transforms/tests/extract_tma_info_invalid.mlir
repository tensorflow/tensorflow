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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
// expected-error @+1 {{Unable to determine swizzle mode from tt.tensordesc's layout}}
tt.func @extract_tma_info_invalid_tma_layout(%arg0: !tt.tensordesc<tensor<16x32xf32, #blocked>>
  {tt.nv_tma_desc = 1 : i32,
   tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 256],
                                                  block_shape = [16, 32],
                                                  element_byte_size = 4>}) {
  tt.return
}