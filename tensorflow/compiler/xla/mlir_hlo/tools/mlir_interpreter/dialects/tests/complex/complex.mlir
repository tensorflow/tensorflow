// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @abs() -> f32 {
  %c = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.abs %c : complex<f32>
  return %ret : f32
}

// CHECK-LABEL: @abs
// CHECK-NEXT: Results
// CHECK-NEXT: 2.236068e+00

func.func @add() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %b = complex.constant [10.0 : f32, 200.0 : f32] : complex<f32>
  %ret = complex.add %a, %b : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @add
// CHECK-NEXT: Results
// CHECK-NEXT: 1.100000e+01+2.020000e+02i

func.func @cos() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.cos %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @cos
// CHECK-NEXT: Results
// CHECK-NEXT: 2.032723e+00-3.051898e+00i

func.func @create() -> complex<f32> {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %ret = complex.create %c1, %c2 : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @create
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00+2.000000e+00i

func.func @div() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %b = complex.constant [3.0 : f32, 4.0 : f32] : complex<f32>
  %ret = complex.div %a, %b : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @div
// CHECK-NEXT: Results
// CHECK-NEXT: 4.400000e-01+8.000000e-02i

func.func @exp() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.exp %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @exp
// CHECK-NEXT: Results
// CHECK-NEXT: -1.131204e+00+2.471727e+00i

func.func @expm1() -> complex<f32> {
  %a = complex.constant [1.0e-06 : f32, 1.0e-06 : f32] : complex<f32>
  %ret = complex.expm1 %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @expm1
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-06+1.000001e-06i

func.func @im() -> f32 {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.im %a : complex<f32>
  return %ret : f32
}

// CHECK-LABEL: @im
// CHECK-NEXT: Results
// CHECK-NEXT: 2.000000e+00

func.func @log() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.log %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @log
// CHECK-NEXT: Results
// CHECK-NEXT: 8.047190e-01+1.107149e+00i

func.func @log1p() -> complex<f32> {
  %a = complex.constant [1.0e-07 : f32, 1.0e-20 : f32] : complex<f32>
  %ret = complex.log1p %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @log1p
// CHECK-NEXT: Results
// CHECK-NEXT: 1.192093e-07+9.999999e-21i

func.func @mul() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %b = complex.constant [3.0 : f32, 4.0 : f32] : complex<f32>
  %ret = complex.mul %a, %b : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @mul
// CHECK-NEXT: Results
// CHECK-NEXT: -5.000000e+00+1.000000e+01i

func.func @neg() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.neg %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @neg
// CHECK-NEXT: Results
// CHECK-NEXT: -1.000000e+00-2.000000e+00i

func.func @pow() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %b = complex.constant [3.0 : f32, 4.0 : f32] : complex<f32>
  %ret = complex.pow %a, %b : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @pow
// CHECK-NEXT: Results
// CHECK-NEXT: 1.290096e-01+3.392413e-02i

func.func @re() -> f32 {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.re %a : complex<f32>
  return %ret : f32
}

// CHECK-LABEL: @re
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @rsqrt() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.rsqrt %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @rsqrt
// CHECK-NEXT: Results
// CHECK-NEXT: 5.688645e-01-3.515776e-01i

func.func @sin() -> complex<f64> {
  %a = complex.constant [1.0 : f64, 2.0 : f64] : complex<f64>
  %ret = complex.sin %a : complex<f64>
  return %ret : complex<f64>
}

// CHECK-LABEL: @sin
// CHECK-NEXT: Results
// CHECK-NEXT: 3.165779e+00+1.959601e+00i

func.func @sqrt() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.sqrt %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @sqrt
// CHECK-NEXT: Results
// CHECK-NEXT: 1.272020e+00+7.861515e-01i

func.func @tanh() -> complex<f32> {
  %a = complex.constant [1.0 : f32, 2.0 : f32] : complex<f32>
  %ret = complex.tanh %a : complex<f32>
  return %ret : complex<f32>
}

// CHECK-LABEL: @tanh
// CHECK-NEXT: Results
// CHECK-NEXT: 1.166736e+00-2.434582e-01i
