// RUN: mlir-hlo-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @noop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32>)
// CHECK: return %[[ARG0]]
func @noop(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[]> : tensor<0xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}
