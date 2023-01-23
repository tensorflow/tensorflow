// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @no_op_cast() -> i32 {
  %cst = arith.constant 42 : i32
  %cast = builtin.unrealized_conversion_cast %cst : i32 to i32
  return %cast : i32
}

// CHECK-LABEL: @no_op_cast
// CHECK{LITERAL}: 42
