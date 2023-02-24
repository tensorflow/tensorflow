// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @while_empty() -> memref<i64> {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %alloc = memref.alloc() : memref<i64>
  memref.store %c0, %alloc[] : memref<i64>
  scf.while: () -> () {
    %value = memref.load %alloc[] : memref<i64>
    %cond = arith.cmpi slt, %value, %c4 : i64
    scf.condition(%cond)
  } do {
    %value = memref.load %alloc[] : memref<i64>
    %add = arith.addi %value, %c1 : i64
    memref.store %add, %alloc[] : memref<i64>
    scf.yield
  }
  return %alloc : memref<i64>
}

// CHECK-LABEL: @while_empty
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: TensorOrMemref<i64>: 4

func.func @while_var() -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %alloc = memref.alloc() : memref<i64>
  memref.store %c0, %alloc[] : memref<i64>
  %ret = scf.while(%arg0 = %c0): (i64) -> (i64) {
    %cond = arith.cmpi slt, %arg0, %c4 : i64
    scf.condition(%cond) %arg0 : i64
  } do {
  ^bb0(%arg1: i64):
    %add = arith.addi %arg1, %c1 : i64
    scf.yield %add : i64
  }
  return %ret : i64
}

// CHECK-LABEL: @while_var
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: i64: 4
