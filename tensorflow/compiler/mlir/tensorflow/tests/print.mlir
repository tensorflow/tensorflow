// RUN: tf-opt %s --tf-print | FileCheck %s

module {
// Smoke test. We don't expect any modifications of the MLIR.

// CHECK-LABEL: foo
// CHECK: return
func.func @foo(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

}
