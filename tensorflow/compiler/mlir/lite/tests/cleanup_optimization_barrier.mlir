// RUN: litert-opt %s --tfl-cleanup-optimization-barrier --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @cleanup_barrier(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK:           %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:           %1 = tfl.add(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:           return %1 : tensor<2x2xf32>

func.func @cleanup_barrier(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = arith.constant dense<5.000000e+00> : tensor<f32>
    %0 = tfl.add(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    %1 = stablehlo.optimization_barrier %0 : tensor<2x2xf32>
    %2 = tfl.add(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
}
