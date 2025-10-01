// RUN: xla-opt --convert-unsupported-types %s | FileCheck %s

module {
  // CHECK:   func.func @triton_fn(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<f8E4M3FN>, %arg3: !tt.ptr<i8>, %arg4: !tt.ptr<f32>) {
  func.func @triton_fn(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E8M0FNU>, %arg2: !tt.ptr<f8E4M3FN>, %arg3: !tt.ptr<f8E8M0FNU>, %arg4: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %extracted_tile = triton_xla.extract from %arg0 as memref<64x512xf8E4M3FN, #triton_xla.layout<[1, 0]>> [0, 0] [16, 32] [1, 1] : tensor<16x32xf8E4M3FN>
    // CHECK: %[[arg_0:.*]] = triton_xla.extract from %arg0 as memref<64x512xf8E4M3FN, #triton_xla.layout<[1, 0]>> [0, 0] [16, 32] [1, 1] : tensor<16x32xf8E4M3FN>
    %extracted_tile_0 = triton_xla.extract from %arg1 as memref<64x16xf8E8M0FNU, #triton_xla.layout<[1, 0]>> [0, 0] [16, 1] [1, 1] : tensor<16x1xf8E8M0FNU>
    // CHECK: %[[arg_1:.*]] = triton_xla.extract from %arg1 as memref<64x16xi8, #triton_xla.layout<[1, 0]>> [0, 0] [16, 1] [1, 1] : tensor<16x1xi8>
    %extracted_tile_1 = triton_xla.extract from %arg2 as memref<512x64xf8E4M3FN, #triton_xla.layout<[1, 0]>> [0, 0] [32, 16] [1, 1] : tensor<32x16xf8E4M3FN>
    // CHECK: %[[arg_2:.*]] = triton_xla.extract from %arg2 as memref<512x64xf8E4M3FN, #triton_xla.layout<[1, 0]>> [0, 0] [32, 16] [1, 1] : tensor<32x16xf8E4M3FN>
    %extracted_tile_2 = triton_xla.extract from %arg3 as memref<16x64xf8E8M0FNU, #triton_xla.layout<[1, 0]>> [0, 0] [1, 16] [1, 1] : tensor<1x16xf8E8M0FNU>
    // CHECK: %[[arg_3:.*]] = triton_xla.extract from %arg3 as memref<16x64xi8, #triton_xla.layout<[1, 0]>> [0, 0] [1, 16] [1, 1] : tensor<1x16xi8>
    %16 = arith.bitcast %extracted_tile_0 : tensor<16x1xf8E8M0FNU> to tensor<16x1xi8>
    %17 = arith.bitcast %extracted_tile_2 : tensor<1x16xf8E8M0FNU> to tensor<1x16xi8>
    %18 = tt.dot_scaled %extracted_tile scale %16, %extracted_tile_1 scale %17, %cst lhs = e4m3 rhs = e4m3 {fastMath = true} : tensor<16x32xf8E4M3FN>, tensor<16x1xi8> * tensor<32x16xf8E4M3FN>, tensor<1x16xi8> -> tensor<16x16xf32>
    return
  }
}