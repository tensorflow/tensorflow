// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

// Adapted from mlir/test/Integration/Dialect/Vector/CPU/test-gather.mlir

func.func private @gather8(%base: memref<10xi32>, %indices: vector<8xi32>,
              %mask: vector<8xi1>, %pass_thru: vector<8xi32>) -> vector<8xi32> {
  %c0 = arith.constant 0: index
  %g = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
  return %g : vector<8xi32>
}

func.func @gather() ->
    (vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>) {
  %A = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : memref<10xi32>
  %idx = arith.constant dense<[0, 6, 1, 3, 5, 4, 9, 2]> : vector<8xi32>
  %pass = arith.constant dense<-7> : vector<8xi32>
  %none = vector.constant_mask [0] : vector<8xi1>
  %all = vector.constant_mask [8] : vector<8xi1>
  %some = vector.constant_mask [4] : vector<8xi1>
  %true = arith.constant true
  %more = vector.insert %true, %some[7] : i1 into vector<8xi1>

  %g1 = call @gather8(%A, %idx, %all, %pass)
    : (memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>)
    -> (vector<8xi32>)
  %g2 = call @gather8(%A, %idx, %none, %pass)
    : (memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>)
    -> (vector<8xi32>)
  %g3 = call @gather8(%A, %idx, %some, %pass)
    : (memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>)
    -> (vector<8xi32>)
  %g4 = call @gather8(%A, %idx, %more, %pass)
    : (memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>)
    -> (vector<8xi32>)
  %g5 = call @gather8(%A, %idx, %all, %pass)
    : (memref<10xi32>, vector<8xi32>, vector<8xi1>, vector<8xi32>)
    -> (vector<8xi32>)

  return %g1, %g2, %g3, %g4, %g5
    : vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>
}

// CHECK-LABEL: @gather
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 6, 1, 3, 5, 4, 9, 2]
// CHECK-NEXT: [-7, -7, -7, -7, -7, -7, -7, -7]
// CHECK-NEXT: [0, 6, 1, 3, -7, -7, -7, -7]
// CHECK-NEXT: [0, 6, 1, 3, -7, -7, -7, 2]
// CHECK-NEXT: [0, 6, 1, 3, 5, 4, 9, 2]
