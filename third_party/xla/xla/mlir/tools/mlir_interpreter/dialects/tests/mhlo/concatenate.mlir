// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @concat_1d()  -> tensor<3xi32> {
  %a = mhlo.constant dense<[1]> : tensor<1xi32>
  %b = mhlo.constant dense<[2, 3]> : tensor<2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 0 : i64 }
     : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @concat_1d
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 2, 3]

func.func @concat_dim0() -> tensor<4x2xi32> {
  %a = mhlo.constant dense<1> : tensor<2x2xi32>
  %b = mhlo.constant dense<2> : tensor<2x2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 0 : i64 }
     : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  func.return %0 : tensor<4x2xi32>
}

// CHECK-LABEL: @concat_dim0
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1], [2, 2], [2, 2]]

func.func @concat_dim1() -> tensor<2x4xi32> {
  %a = mhlo.constant dense<1> : tensor<2x2xi32>
  %b = mhlo.constant dense<2> : tensor<2x2xi32>
  %0 = "mhlo.concatenate"(%a, %b) { dimension = 1 : i64 }
     : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: @concat_dim1
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1, 2, 2], [1, 1, 2, 2]]
