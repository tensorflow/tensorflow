// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @i32() -> (i32, i32) {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32
  %r1 = arith.minsi %c-1, %c1 : i32
  %r2 = arith.maxsi %c-1, %c1 : i32
  return %r1, %r2 : i32, i32
}

// CHECK-LABEL: @i32
// CHECK{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1

func.func @i64() -> (i64, i64) {
  %c-1 = arith.constant -1 : i64
  %c1 = arith.constant 1000000000000 : i64
  %r1 = arith.minsi %c-1, %c1 : i64
  %r2 = arith.maxsi %c-1, %c1 : i64
  return %r1, %r2 : i64, i64
}

// CHECK-LABEL: @i64
// CHECK{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1000000000000
