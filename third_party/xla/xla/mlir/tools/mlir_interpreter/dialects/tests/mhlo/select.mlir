// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @reshape() -> tensor<2xi32> {
  %cst = mhlo.constant dense<[true, false]> : tensor<2xi1>
  %a = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %b = mhlo.constant dense<[3, 4]> : tensor<2xi32>
  %ret = "mhlo.select"(%cst, %a, %b) :
    (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @reshape
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 4]
