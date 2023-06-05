// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dynamic_update_slice() -> tensor<3x4xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %v = mhlo.constant dense<[[13, 14]]> : tensor<1x2xi32>
  %s0 = mhlo.constant dense<1> : tensor<i32>
  %s1 = mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.dynamic_update_slice"(%cst, %v, %s0, %s1)
    : (tensor<3x4xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>)
      -> tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
}

// CHECK-LABEL: @dynamic_update_slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 3, 4], [13, 14, 7, 8], [9, 10, 11, 12]]

func.func @clamp_starts() -> tensor<3x4xi32> {
  %cst = mhlo.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]>
    : tensor<3x4xi32>
  %v = mhlo.constant dense<[[13, 14]]> : tensor<1x2xi32>
  %s0 = mhlo.constant dense<-10> : tensor<i32>
  %s1 = mhlo.constant dense<10> : tensor<i32>
  %0 = "mhlo.dynamic_update_slice"(%cst, %v, %s0, %s1) {
    slice_sizes = dense<[2, 3]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<1x2xi32>, tensor<i32>, tensor<i32>)
      -> tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
}

// CHECK-LABEL: @clamp_starts
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 2, 13, 14], [5, 6, 7, 8], [9, 10, 11, 12]]
