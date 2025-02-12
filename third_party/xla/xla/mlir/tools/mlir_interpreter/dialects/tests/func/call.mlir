// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @cbrtf_caller() -> f32 {
  %c-27 = arith.constant -27.0 : f32
  %ret = func.call @cbrtf(%c-27) : (f32) -> f32
  func.return %ret : f32
}

// CHECK-LABEL: @cbrtf_caller
// CHECK-NEXT: Results
// CHECK-NEXT: -3.000000e+00

func.func @cbrt_caller() -> f64 {
  %c-8 = arith.constant -8.0 : f64
  %ret = func.call @cbrt(%c-8) : (f64) -> f64
  func.return %ret : f64
}

// CHECK-LABEL: @cbrt_caller
// CHECK-NEXT: Results
// CHECK-NEXT: -2.000000e+00

func.func @atan2f_caller() -> f32 {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %ret = func.call @atan2f(%c1, %c2) : (f32, f32) -> f32
  func.return %ret : f32
}

// CHECK-LABEL: @atan2f_caller
// CHECK-NEXT: Results
// CHECK-NEXT: 4.636476e-01

func.func @atan2_caller() -> f64 {
  %c2 = arith.constant 2.0 : f64
  %c3 = arith.constant 3.0 : f64
  %ret = func.call @atan2(%c2, %c3) : (f64, f64) -> f64
  func.return %ret : f64
}

// CHECK-LABEL: @atan2_caller
// CHECK-NEXT: Results
// CHECK-NEXT: 5.880026e-01

func.func private @cbrtf(%a: f32) -> f32
func.func private @cbrt(%a: f64) -> f64
func.func private @atan2f(%a: f32, %b: f32) -> f32
func.func private @atan2(%a: f64, %b: f64) -> f64
