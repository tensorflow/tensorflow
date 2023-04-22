// RUN: tf-opt -xla-prepare-for-export %s | FileCheck %s

// CHECK-LABEL: func @splat_constants
func @splat_constants() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
  return %cst : tensor<1x64x224x224xf32>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[CST]])
  // CHECK-SAME: (tensor<f32>) -> tensor<1x64x224x224xf32>
}
