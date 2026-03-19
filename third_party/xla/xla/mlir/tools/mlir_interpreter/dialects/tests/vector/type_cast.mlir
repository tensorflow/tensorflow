// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @type_cast() -> memref<vector<2x2xi32>> {
  %alloc = arith.constant dense<[[1, 2], [3, 4]]> : memref<2x2xi32>
  %cast = vector.type_cast %alloc: memref<2x2xi32> to memref<vector<2x2xi32>>
  return %cast : memref<vector<2x2xi32>>
}

// CHECK-LABEL: @type_cast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<vector<2x2xi32>>: [[1, 2], [3, 4]]
