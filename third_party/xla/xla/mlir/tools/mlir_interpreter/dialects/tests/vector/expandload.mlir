// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @expandload() -> (vector<4xi32>, vector<4xi32>) {
  %passthrough = arith.constant dense<[1,2,3,4]> : vector<4xi32>
  %mask = arith.constant dense<[true, false, true, false]> : vector<4xi1>
  %memref = arith.constant dense<[[10,11,12,13,14],
                                  [15,16,17,18,19]]> : memref<2x5xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ret = vector.expandload %memref[%c1, %c2], %mask, %passthrough
    : memref<2x5xi32>, vector<4xi1>, vector<4xi32> into vector<4xi32>

  return %passthrough, %ret : vector<4xi32>, vector<4xi32>
}

// CHECK-LABEL: @expandload
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi32>: [1, 2, 3, 4]
// CHECK-NEXT: vector<4xi32>: [17, 2, 18, 4]
