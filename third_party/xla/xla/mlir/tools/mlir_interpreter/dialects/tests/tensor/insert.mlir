// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @insert() -> tensor<1x2x3xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : i32
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.insert %c7 into %cst[%c0, %c1, %c1] : tensor<1x2x3xi32>
  return %ret : tensor<1x2x3xi32>
}

// CHECK-LABEL: @insert
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1, 2, 3], [4, 7, 6]]]
