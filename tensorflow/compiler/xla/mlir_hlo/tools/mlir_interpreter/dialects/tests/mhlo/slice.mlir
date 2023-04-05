// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @slice() -> tensor<1x2xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %0 = "mhlo.slice"(%cst) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// CHECK-LABEL: @slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[5, 7]]
