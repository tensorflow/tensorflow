// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @case() -> tensor<i32> {
  %c1 = mhlo.constant dense<1> : tensor<i32>
  %c2 = mhlo.constant dense<2> : tensor<i32>
  %c3 = mhlo.constant dense<3> : tensor<i32>
  %ret = "mhlo.case"(%c1) ({
    "mhlo.return"(%c2) : (tensor<i32>) -> ()
  }, {
    "mhlo.return"(%c3) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return %ret : tensor<i32>
}

// CHECK-LABEL: @case
// CHECK-NEXT: Results
// CHECK-NEXT: <i32>: 3
