// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @copysign() -> (f32, f32) {
  %a = arith.constant 1.5 : f32
  %b = arith.constant -10.0 : f32
  %c = math.copysign %a, %b : f32
  %d = math.copysign %a, %a : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @copysign
// CHECK-NEXT: Results
// CHECK-NEXT: -1.500000e+00
// CHECK-NEXT: 1.500000e+00

func.func @absf() -> (f32, f32) {
  %a = arith.constant 1.0 : f32
  %b = arith.constant -1.0 : f32
  %c = math.absf %a : f32
  %d = math.absf %b : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @absf
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: 1.000000e+00

func.func @cos() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = math.cos %a : f32
  return %b : f32
}

// CHECK-LABEL: @cos
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00


func.func @exp() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = math.exp %a : f32
  return %b : f32
}

// CHECK-LABEL: @exp
// CHECK-NEXT: Results
// CHECK-NEXT: 2.718282e+00

func.func @sin() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = math.sin %a : f32
  return %b : f32
}

// CHECK-LABEL: @sin
// CHECK-NEXT: Results
// CHECK-NEXT: 0.000000e+00

func.func @floor() -> (f32, f32) {
  %a = arith.constant 1.5 : f32
  %b = arith.constant -10.5 : f32
  %c = math.floor %a : f32
  %d = math.floor %b : f32
  return %c, %d : f32, f32
}

// CHECK-LABEL: @floor
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: -1.100000e+01

func.func @log() -> f32 {
  %a = arith.constant 1.0 : f32
  %ret = math.log %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log
// CHECK-NEXT: Results
// CHECK-NEXT: 0.000000e+00

func.func @log1p() -> f32 {
  %a = arith.constant 1.0e-10 : f32
  %ret = math.log1p %a : f32
  return %ret : f32
}

// CHECK-LABEL: @log1p
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-10

func.func @powf() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %c = math.powf %a, %b : f32
  return %c : f32
}

// CHECK-LABEL: @powf
// CHECK-NEXT: Results
// CHECK-NEXT: 8.000000e+00
