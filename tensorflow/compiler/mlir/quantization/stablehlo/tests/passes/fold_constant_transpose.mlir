// RUN: stablehlo-quant-opt %s -stablehlo-fold-constant-transpose \
// RUN:   -split-input-file | FileCheck %s

// CHECK-LABEL: transpose_simple_1d
func.func @transpose_simple_1d() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[0.000000e+0, 1.000000e+0]> : tensor<2xf32>
  %1 = stablehlo.transpose %0, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
// CHECK-NOT: transpose
// CHECK: return %[[CONST_0]] : tensor<2xf32>

// -----

// CHECK-LABEL: transpose_simple_2d
func.func @transpose_simple_2d() -> tensor<3x2xf32> {
  %0 = stablehlo.constant dense<[[0.000000e+0, 1.000000e+0, 2.000000e+0], [3.000000e+0, 4.000000e+0, 5.000000e+0]]> : tensor<2x3xf32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant dense<{{\[\[}}0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
// CHECK-NOT: transpose
// CHECK: return %[[CONST_0]] : tensor<3x2xf32>

// -----

// CHECK-LABEL: transpose_simple_4d
func.func @transpose_simple_4d() -> tensor<5x2x3x4xf32> {
  %0 = stablehlo.constant dense<1.000000e+0> : tensor<2x3x4x5xf32>
  %1 = stablehlo.transpose %0, dims = [3, 0, 1, 2] : (tensor<2x3x4x5xf32>) -> tensor<5x2x3x4xf32>
  return %1 : tensor<5x2x3x4xf32>
}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<5x2x3x4xf32>
// CHECK-NOT: transpose
// CHECK: return %[[CONST_0]] : tensor<5x2x3x4xf32>

// -----

// Tests that int constants are not folded.

// CHECK-LABEL: transpose_int
func.func @transpose_int() -> tensor<3x2xi32> {
  %0 = stablehlo.constant dense<0> : tensor<2x3xi32>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi32>) -> tensor<3x2xi32>
  return %1 : tensor<3x2xi32>
}
// CHECK: transpose

// -----

// Tests that transposing an argument cannot be folded.

// CHECK-LABEL: transpose_arg
func.func @transpose_arg(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// CHECK: transpose
