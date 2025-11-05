// RUN: xla-opt %s -split-input-file \
// RUN: -tensor-lower-to-triton \
// RUN: | FileCheck %s

//TODO(basioli): Consider fusing this and stablehlo_to_triton_lowering.mlir into xtile_to_triton_lowering.mlir

// CHECK: func @lower_bitcast(%[[ARG:.*]]: tensor<2x4x8xf32>) -> tensor<2x4x8xi32>
func.func @lower_bitcast(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xi32> {
  // CHECK: %[[RES:.*]] = tt.bitcast %[[ARG]] : tensor<2x4x8xf32> -> tensor<2x4x8xi32>
  %0 = tensor.bitcast %arg0 : tensor<2x4x8xf32> to tensor<2x4x8xi32>
  // CHECK: return %[[RES]] : tensor<2x4x8xi32>
  return %0 : tensor<2x4x8xi32>
}
