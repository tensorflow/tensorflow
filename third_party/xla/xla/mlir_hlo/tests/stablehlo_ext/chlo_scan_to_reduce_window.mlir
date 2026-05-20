// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-scan-to-reduce-window="target-version=1.13.0" | FileCheck %s --check-prefix=CHECK-13
// RUN: mlir-hlo-opt %s --stablehlo-ext-chlo-scan-to-reduce-window="target-version=1.14.0" | FileCheck %s --check-prefix=CHECK-14

// CHECK-13-LABEL: @scan
// CHECK-14-LABEL: @scan
func.func @scan(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<f32>) {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-13: stablehlo.reduce_window
  // CHECK-14: chlo.scan
  %0:2 = chlo.scan(%arg0) inits(%cst) dimension=0 {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<4xf32>, tensor<f32>) -> (tensor<4xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<f32>
}
