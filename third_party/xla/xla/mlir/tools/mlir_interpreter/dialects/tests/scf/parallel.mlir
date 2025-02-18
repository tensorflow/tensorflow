// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @parallel() -> memref<4x4xi32> {
  %ret = memref.alloc() : memref<4x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : i32
  scf.parallel (%i, %j) = (%c0, %c1) to (%c4, %c4) step (%c1, %c2) {
    memref.store %c9, %ret[%i, %j] : memref<4x4xi32>
  }
  return %ret : memref<4x4xi32>
}

// CHECK-LABEL: @parallel
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 9, 0, 9], [0, 9, 0, 9], [0, 9, 0, 9], [0, 9, 0, 9]]

func.func @reduce_2() -> (index, index) {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %ret:2 = scf.parallel (%i) = (%c1) to (%c6) step (%c1)
             init (%c1, %c1) -> (index, index) {
    scf.reduce (%i, %i : index, index) {
      ^bb0(%lhs: index, %rhs: index):
        %ret = arith.muli %lhs, %rhs : index
        scf.reduce.return %ret : index
    }, {
      ^bb0(%lhs: index, %rhs: index):
        %ret = arith.addi %lhs, %rhs : index
        scf.reduce.return %ret : index
    }
  }
  return %ret#0, %ret#1 : index, index
}

// CHECK-LABEL: @reduce_2
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: i64: 120
// CHECK-NEXT{LITERAL}: i64: 16

