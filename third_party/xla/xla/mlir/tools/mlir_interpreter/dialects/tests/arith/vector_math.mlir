// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @addi() -> vector<2xi32> {
  %c1 = arith.constant dense<[1, 2]> : vector<2xi32>
  %c2 = arith.constant dense<[3, 4]> : vector<2xi32>
  %ret = arith.addi %c1, %c2 : vector<2xi32>
  return %ret : vector<2xi32>
}

// CHECK-LABEL: @addi
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xi32>: [4, 6]
