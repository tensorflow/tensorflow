// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @constant_mask() -> vector<4x3xi1> {
  %1 = vector.constant_mask [3, 2] : vector<4x3xi1>
  return %1 : vector<4x3xi1>
}

// CHECK-LABEL: @constant_mask
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<4x3xi1>:
// CHECK-SAME{LITERAL}: [[true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [false, false, false]]
