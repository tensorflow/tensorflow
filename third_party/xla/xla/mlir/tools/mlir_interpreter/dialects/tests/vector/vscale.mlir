// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @vscale() -> (vector<[4]xi32>, index) {
  %c = arith.constant dense<0> : vector<[4]xi32>
  %vscale = vector.vscale
  return %c, %vscale : vector<[4]xi32>, index
}

// CHECK-LABEL: @vscale
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 0, 0, 0]
// CHECK-NEXT: 1