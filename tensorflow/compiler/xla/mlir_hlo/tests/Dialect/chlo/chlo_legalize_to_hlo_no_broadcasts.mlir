// RUN: mlir-hlo-opt -chlo-legalize-to-hlo="legalize-broadcasts=false" %s | FileCheck %s

// CHECK-LABEL: atan_static
// CHECK-SAME: %[[ARG:.*]]: tensor<2x3x4xf32>
func.func @atan_static(%arg0: tensor<2x3x4xf32>) -> tuple<tensor<2x3x4xf32>> {
  // CHECK: %[[CST:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x3x4xf32>
  // CHECK: mhlo.atan2 %[[ARG]], %[[CST]] : tensor<2x3x4xf32>
  %0 = chlo.atan %arg0 : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
  %1 = "mhlo.tuple"(%0) : (tensor<2x3x4xf32>) -> tuple<tensor<2x3x4xf32>>
  func.return %1 : tuple<tensor<2x3x4xf32>>
}
