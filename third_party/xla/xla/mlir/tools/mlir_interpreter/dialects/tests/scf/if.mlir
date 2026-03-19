// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @true() -> i64 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %true = arith.constant true
  %ret = scf.if %true -> i64 {
    scf.yield %c0 : i64
  } else {
    scf.yield %c1 : i64
  }
  return %ret : i64
}

// CHECK-LABEL: @true
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 0

func.func @false() -> i64 {
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %false = arith.constant false
  %ret = scf.if %false -> i64 {
    scf.yield %c2 : i64
  } else {
    scf.yield %c3 : i64
  }
  return %ret : i64
}

// CHECK-LABEL: @false
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 3

func.func @side_effect() -> memref<i64> {
  %alloc = memref.alloc() : memref<i64>
  %true = arith.constant true
  %c124 = arith.constant 124 : i64
  %c125 = arith.constant 125 : i64
  scf.if %true {
    memref.store %c124, %alloc[] : memref<i64>
    scf.yield
  } else {
    memref.store %c125, %alloc[] : memref<i64>
    scf.yield
  }
  return %alloc : memref<i64>
}

// CHECK-LABEL: @side_effect
// CHECK-NEXT: Results
// CHECK-NEXT: <i64>: 124

func.func @side_effect_not_executed() -> memref<i64> {
  %alloc = memref.alloc() : memref<i64>
  %false = arith.constant false
  %c126 = arith.constant 126 : i64
  memref.store %c126, %alloc[] : memref<i64>
  %c127 = arith.constant 127 : i64
  scf.if %false {
    memref.store %c127, %alloc[] : memref<i64>
    scf.yield
  }
  return %alloc : memref<i64>
}

// CHECK-LABEL: @side_effect_not_executed
// CHECK-NEXT: Results
// CHECK-NEXT: <i64>: 126
