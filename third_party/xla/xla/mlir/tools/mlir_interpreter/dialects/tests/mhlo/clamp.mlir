// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @clamp() -> tensor<2x2xi32> {
  %lb = mhlo.constant dense<[[1, 7], [1, 7]]> : tensor<2x2xi32>
  %arg = mhlo.constant dense<[[4, 5], [6, 9]]> : tensor<2x2xi32>
  %ub = mhlo.constant dense<[[5, 9], [3, 6]]> : tensor<2x2xi32>
  %clamp = "mhlo.clamp"(%lb, %arg, %ub)
      : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %clamp : tensor<2x2xi32>
}

// CHECK-LABEL:         @clamp
// CHECK-NEXT:          Results
// CHECK-NEXT{LITERAL}: [[4, 7], [3, 6]]

func.func @clamp_f32() -> tensor<2x2xf32> {
  %lb = mhlo.constant dense<[[1.1, 7.1], [1.1, 7.1]]> : tensor<2x2xf32>
  %arg = mhlo.constant dense<[[4.1, 5.1], [6.1, 9.1]]> : tensor<2x2xf32>
  %ub = mhlo.constant dense<[[5.1, 9.1], [3.1, 6.1]]> : tensor<2x2xf32>
  %clamp = "mhlo.clamp"(%lb, %arg, %ub)
      : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %clamp : tensor<2x2xf32>
}

// CHECK-LABEL:         @clamp
// CHECK-NEXT:          Results
// CHECK-NEXT{LITERAL}: [[4.100000e+00, 7.100000e+00], [3.100000e+00, 6.100000e+00]]
