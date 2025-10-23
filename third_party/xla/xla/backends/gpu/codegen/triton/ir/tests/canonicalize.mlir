// RUN: xla-opt %s --canonicalize | FileCheck %s

<<<<<<< HEAD
tt.func @xla_triton_extract(%arg0: !tt.ptr<bf16>, %i : index)
    -> tensor<16x64xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %extracted_tensor = triton_xla.extract from %arg0 as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
      [%c0, %i] [16, 64] [%c128, %c1] {noinline = false} : tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
=======
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
>>>>>>> upstream/master
}

<<<<<<< HEAD
// CHECK:       triton_xla.extract
// CHECK-SAME:    [0, %{{.*}}] [16, 64] [128, 1]
// CHECK-SAME:    {noinline = false}

// -----

tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: !tt.ptr<bf16>,
    %j: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  triton_xla.insert %src into %dst as memref<512x128xbf16, #triton_xla.layout<[1, 0]>>
    [%c0, %c0][16, 64][%j, %c1] {noinline = false} : tensor<16x64xbf16>
  tt.return
}
// CHECK-LABEL: xla_triton_insert
// CHECK:       triton_xla.insert
// CHECK-SAME:    [0, 0] [16, 64] [%{{.*}}, 1]
// CHECK-SAME:    {noinline = false}
=======
// CHECK-LABEL: @fold_ptr_memref_ptr(
// CHECK-SAME: %[[SRC:.*]]: !tt.ptr<f32>
func.func @fold_ptr_memref_ptr(%src: !tt.ptr<f32>) -> !tt.ptr<f32> {
  // CHECK: return %[[SRC]] : !tt.ptr<f32>
  %src_ptr = triton_xla.ptr_to_memref %src from !tt.ptr<f32> to memref<256xf32>
  %dst = triton_xla.memref_to_ptr %src_ptr from memref<256xf32> to !tt.ptr<f32>
  func.return %dst : !tt.ptr<f32>
}
>>>>>>> upstream/master
