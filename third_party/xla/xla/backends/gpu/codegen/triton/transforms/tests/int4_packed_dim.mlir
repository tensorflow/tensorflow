// RUN: xla-opt --split-input-file --int4-to-packed-int4-rewrite --canonicalize %s | FileCheck %s

// CHECK-LABEL: @triton_xla_extract_2d
func.func @triton_xla_extract_2d(%arg0: !tt.ptr<i4>) -> (tensor<16x16xi8>) {
  // CHECK: %[[EXTRACT:.*]] = triton_xla.extract from %arg0
  // CHECK-SAME: as memref<128x8x64xi8, #xtile.layout<[2, 1, 0]>>
  // CHECK-SAME: [0, 0, 0] [16, 1, 8] [1, 1, 1] : tensor<16x8xi8>
  %c0 = arith.constant 0 : index
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<128x8x128xi4, #xtile.layout<[2, 1, 0]>>
      [0, 0, %c0] [16, 1, 16] [1, 1, 1] : tensor<16x16xi4>
  %ext = arith.extsi %extracted_tensor : tensor<16x16xi4> to tensor<16x16xi8>
  // CHECK: %[[SHLI:.*]] = arith.shli %[[EXTRACT]]
  // CHECK: %[[SHRI_LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK: %[[SHRI_HI:.*]] = arith.shrsi %[[EXTRACT]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[SHRI_LO]], %[[SHRI_HI]]
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[JOIN]]
  // CHECK: return %[[RESHAPE]] : tensor<16x16xi8>
  func.return %ext : tensor<16x16xi8>
}
