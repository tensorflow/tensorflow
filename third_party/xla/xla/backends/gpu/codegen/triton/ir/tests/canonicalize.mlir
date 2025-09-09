// RUN: xla-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: xla_triton_extract_insert
tt.func @xla_triton_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: index) {
  %c0 = arith.constant 0 : index
  // CHECK:       triton_xla.extract
  // CHECK-SAME:    [%arg1, 0] [16, 64] [128, 1]
  // CHECK-SAME:    {noinline = false}
  %tile = triton_xla.extract from %arg0
      as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
      [%arg1, %c0] [16, 64] [128, 1] {noinline = false} : tensor<16x64xbf16>
  // CHECK:       triton_xla.insert
  // CHECK-SAME:    [0, %arg1] [16, 64] [1, 1]
  // CHECK-SAME:    {noinline = false}
  triton_xla.insert %tile into %arg0
      as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
      [%c0, %arg1][16, 64][1, 1] {noinline = false} : tensor<16x64xbf16>
  tt.return
}
