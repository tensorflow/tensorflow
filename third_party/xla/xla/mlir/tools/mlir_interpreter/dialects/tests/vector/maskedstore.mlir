// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @maskedstore() -> memref<2x5xi32> {
  %value = arith.constant dense<[1,2,3,4]> : vector<4xi32>
  %mask = arith.constant dense<[true, false, true, false]> : vector<4xi1>
  %memref = arith.constant dense<[[10,11,12,13,14],
                                  [15,16,17,18,19]]> : memref<2x5xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  vector.maskedstore %memref[%c1, %c2], %mask, %value
    : memref<2x5xi32>, vector<4xi1>, vector<4xi32>

  return %memref : memref<2x5xi32>
}

// CHECK-LABEL: @maskedstore
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[10, 11, 12, 13, 14], [15, 16, 1, 18, 3]]
