// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @splat() -> vector<2x4xi32> {
  %c42 = arith.constant 42 : i32
  %splat = vector.splat %c42 : vector<2x4xi32>
  return %splat : vector<2x4xi32>
}

// CHECK-LABEL: @splat
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x4xi32>: [[42, 42, 42, 42], [42, 42, 42, 42]]
