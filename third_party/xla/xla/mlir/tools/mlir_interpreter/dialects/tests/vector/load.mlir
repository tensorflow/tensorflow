// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @load_vector_memref() -> vector<2x2xi32> {
  %m = memref.alloc() : memref<4x4xvector<2x2xi32>>
  %c3 = arith.constant 3 : index
  %r = vector.load %m[%c3, %c3] : memref<4x4xvector<2x2xi32>>, vector<2x2xi32>
  return %r : vector<2x2xi32>
}

// CHECK-LABEL: @load_vector_memref
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[0, 0], [0, 0]]

func.func @load_scalar_memref() -> vector<2x2xi32> {
  %m = arith.constant dense<[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]> : memref<4x4xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %r = vector.load %m[%c1, %c2] : memref<4x4xi32>, vector<2x2xi32>
  return %r : vector<2x2xi32>
}

// CHECK-LABEL: @load_scalar_memref
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[7, 8], [11, 12]]
