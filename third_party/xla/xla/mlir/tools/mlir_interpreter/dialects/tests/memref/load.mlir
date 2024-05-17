// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @load() -> i32 {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xi32>
  %ret = memref.load %cst[%c1, %c1] : memref<2x2xi32>
  return %ret : i32
}

// CHECK-LABEL: @load
// CHECK-NEXT: Results
// CHECK-NEXT: 4
