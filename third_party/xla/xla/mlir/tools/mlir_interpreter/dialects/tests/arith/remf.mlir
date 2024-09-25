// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @remf() -> f32 {
  %a = arith.constant 3.5 : f32
  %b = arith.constant 2.25 : f32
  %ret = arith.remf %a, %b : f32
  return %ret : f32
}

// CHECK-LABEL: @remf
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 1.250000e+00
