// RUN: xla-opt %s --split-input-file --canonicalize | FileCheck %s

tt.func @xla_triton_extract(%arg0: tensor<512x128xbf16>, %i : index)
    -> tensor<16x64xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %extracted_tensor = triton_xla.extract %arg0 [%c0, %i] [16, 64] [%c128, %c1]
    {layout = array<i64:1, 0>, noinline = false}
    : tensor<512x128xbf16> to tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}
// CHECK-LABEL: xla_triton_extract

// CHECK:       triton_xla.extract
// CHECK-SAME:    [0, %{{.*}}] [16, 64] [128, 1]
// CHECK-SAME:    {layout = array<i64: 1, 0>, noinline = false}
// CHECK-SAME:    : tensor<512x128xbf16> to tensor<16x64xbf16>

// -----

tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: tensor<512x128xbf16>,
    %j: index) -> tensor<512x128xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %updated_tensor = triton_xla.insert %src into %dst [%c0, %c0][16, 64][%j, %c1]
    {layout = array<i64:1, 0>, noinline = false}
    : tensor<16x64xbf16> into tensor<512x128xbf16>
  tt.return %updated_tensor : tensor<512x128xbf16>
}
// CHECK-LABEL: xla_triton_insert
// CHECK:       triton_xla.insert
// CHECK-SAME:    [0, 0] [16, 64] [%{{.*}}, 1]
// CHECK-SAME:    {layout = array<i64: 1, 0>, noinline = false}
// CHECK-SAME:    : tensor<16x64xbf16> into tensor<512x128xbf16>
