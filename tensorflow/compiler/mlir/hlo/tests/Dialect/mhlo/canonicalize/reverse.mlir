// RUN: mlir-hlo-opt %s -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @noop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2xf32>)
func.func @noop(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.reverse"(%arg0) {dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[ARG0]]
  func.return %0 : tensor<1x2xf32>
}
