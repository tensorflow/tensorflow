// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extract() -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.extract %cst[%c0, %c1, %c1] : tensor<1x2x3xi32>
  return %ret : i32
}

// CHECK-LABEL: @extract
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 5
