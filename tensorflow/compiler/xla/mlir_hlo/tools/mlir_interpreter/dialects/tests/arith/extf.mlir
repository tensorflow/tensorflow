// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extf() -> f64 {
  %c1 = arith.constant 1.0 : f32
  %ext = arith.extf %c1 : f32 to f64
  return %ext : f64
}

// CHECK-LABEL: @extf
// CHECK-NEXT: Results
// CHECK-NEXT: f64: 1.000000e+00
