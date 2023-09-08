// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @flattranspose_2x3() -> vector<6xi32> {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xi32>
  %ret = vector.flat_transpose %c { columns = 2: i32, rows = 3: i32 }
    : vector<6xi32> -> vector<6xi32>
  return %ret : vector<6xi32>
}

// CHECK-LABEL: @flattranspose_2x3
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 3, 1, 4, 2, 5]

func.func @flattranspose_3x2() -> vector<6xi32> {
  %c = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xi32>
  %ret = vector.flat_transpose %c { columns = 3: i32, rows = 2: i32 }
    : vector<6xi32> -> vector<6xi32>
  return %ret : vector<6xi32>
}

// CHECK-LABEL: @flattranspose_3x2
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 2, 4, 1, 3, 5]
