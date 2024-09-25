// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @negf32() -> f32 {
  %c = arith.constant -1.5 : f32
  %ret = arith.negf %c : f32
  return %ret : f32
}

// CHECK-LABEL: @negf32
// CHECK-NEXT: Results
// CHECK-NEXT: f32: 1.500000e+00

func.func @negf64() -> f64 {
  %c = arith.constant 3.5 : f64
  %ret = arith.negf %c : f64
  return %ret : f64
}

// CHECK-LABEL: @negf64
// CHECK-NEXT: Results
// CHECK-NEXT: f64: -3.500000e+00
