// RUN: mlir-opt %s -quantizer-remove-instrumentation -split-input-file | FileCheck %s

// -----
// CHECK-LABEL: remove_ops
func @remove_ops(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quant.stats"(%arg0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quant.coupled_ref"(%0) { coupledKey = "foobar" } :
      (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %2 = "quant.stats_ref"(%1) { statsKey = "foobar" } :
      (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  // CHECK: return %arg0 : tensor<8x4x3xf32>
  return %2 : tensor<8x4x3xf32>
}
