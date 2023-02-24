// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @compressstore() -> memref<3x4xi32> {
  %alloc = memref.alloc() : memref<3x4xi32>
  %c = arith.constant dense<[1,2,3]> : vector<3xi32>
  %m = arith.constant dense<[true,false,true]> : vector<3xi1>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  vector.compressstore %alloc[%c1, %c2], %m, %c
    : memref<3x4xi32>, vector<3xi1>, vector<3xi32>
  return %alloc : memref<3x4xi32>
}

// CHECK-LABEL: @compressstore
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 0, 1, 3], [0, 0, 0, 0]]
