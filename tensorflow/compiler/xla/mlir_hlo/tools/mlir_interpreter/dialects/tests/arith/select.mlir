// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @select() -> (i32, i32) {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %r1 = arith.select %true, %c-1, %c1 : i32
  %r2 = arith.select %false, %c-1, %c1 : i32
  return %r1, %r2 : i32, i32
}

// CHECK-LABEL: @select
// CHECK{LITERAL}: -1
// CHECK-NEXT{LITERAL}: 1
