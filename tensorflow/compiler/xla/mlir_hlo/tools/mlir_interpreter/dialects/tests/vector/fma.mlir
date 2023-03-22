// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @fma() -> vector<2xf32> {
  %a = arith.constant dense<[1.0,2.0]> : vector<2xf32>
  %b = arith.constant dense<[3.0,4.0]> : vector<2xf32>
  %c = arith.constant dense<[5.0,6.0]> : vector<2xf32>
  %r = vector.fma %a, %b, %c : vector<2xf32>
  return %r : vector<2xf32>
}

// CHECK-LABEL: @fma
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xf32>: [8.000000e+00, 1.400000e+01]
