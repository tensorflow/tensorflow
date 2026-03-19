// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @no_op_cast() -> i32 {
  %cst = arith.constant 42 : i32
  %cast = builtin.unrealized_conversion_cast %cst : i32 to i32
  return %cast : i32
}

// CHECK-LABEL: @no_op_cast
// CHECK-NEXT: Results
// CHECK{LITERAL}: 42

func.func @cast_to_dynamic() -> tensor<?xi32> {
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %cast = builtin.unrealized_conversion_cast %cst : tensor<3xi32> to tensor<?xi32>
  return %cast : tensor<?xi32>
}

// CHECK-LABEL: @cast_to_dynamic
// CHECK-NEXT: Results
// CHECK{LITERAL}: [0, 1, 2]
