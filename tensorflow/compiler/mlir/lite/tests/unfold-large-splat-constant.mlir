// RUN: tf-opt %s -tfl-unfold-large-splat-constant | FileCheck %s

// CHECK-LABEL: @unfold_large_constant_splat
func @unfold_large_constant_splat() -> (tensor<10x10xf32>, tensor<1000x1000xf32>) {
  %0 = arith.constant dense<0.00000e+00> : tensor<10x10xf32>
  %1 = arith.constant dense<1.00000e+00> : tensor<1000x1000xf32>
  return %0, %1 : tensor<10x10xf32>, tensor<1000x1000xf32>

  // CHECK-DAG: %cst = arith.constant dense<0.000000e+00> : tensor<10x10xf32>
  // CHECK-DAG: %cst_0 = arith.constant dense<1000> : tensor<2xi64>
  // CHECK-DAG: %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %0 = "tfl.fill"(%cst_0, %cst_1) : (tensor<2xi64>, tensor<f32>) -> tensor<1000x1000xf32>
  // CHECK: return %cst, %0 : tensor<10x10xf32>, tensor<1000x1000xf32>
}
