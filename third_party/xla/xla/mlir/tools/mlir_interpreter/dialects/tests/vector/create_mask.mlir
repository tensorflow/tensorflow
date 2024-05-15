// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @create_mask() -> vector<4x3xi1> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %1 = vector.create_mask %c3, %c2 : vector<4x3xi1>
  return %1 : vector<4x3xi1>
}

// CHECK-LABEL: @create_mask
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<4x3xi1>:
// CHECK-SAME{LITERAL}: [[true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [false, false, false]]
