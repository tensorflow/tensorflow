// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dynamic_slice() -> tensor<1x2xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %s0 = mhlo.constant dense<1> : tensor<i32>
  %s1 = mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.dynamic_slice"(%cst, %s0, %s1) {
    slice_sizes = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i32>, tensor<i32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// CHECK-LABEL: @dynamic_slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[5, 6]]

func.func @clamp_starts() -> tensor<2x3xi32> {

  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %s0 = mhlo.constant dense<-10> : tensor<i32>
  %s1 = mhlo.constant dense<10> : tensor<i32>
  %0 = "mhlo.dynamic_slice"(%cst, %s0, %s1) {
    slice_sizes = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i32>, tensor<i32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @clamp_starts
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[2, 3, 4], [6, 7, 8]]
