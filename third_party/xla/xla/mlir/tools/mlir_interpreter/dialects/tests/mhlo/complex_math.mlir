// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @cos() -> tensor<complex<f64>> {
  %c = mhlo.constant dense<(1.0, 1.0)> : tensor<complex<f64>>
  %cos = mhlo.cosine %c : tensor<complex<f64>>
  return %cos : tensor<complex<f64>>
}

// CHECK-LABEL: @cos
// CHECK-NEXT: Results
// CHECK-NEXT: 8.337300e-01-9.888977e-01i

func.func @expm1() -> tensor<complex<f32>> {
  // import numpy as np  -- not jax.numpy
  // np.expm1(np.array(1e-6 + 1e-6j, dtype=np.complex64))
  // Don't run this with jax.numpy, it returns 9.536743e-07+0.j.
  %c = mhlo.constant dense<(1.0e-06, 1.0e-06)> : tensor<complex<f32>>
  %expm1 = mhlo.exponential_minus_one %c : tensor<complex<f32>>
  return %expm1 : tensor<complex<f32>>
}

// CHECK-LABEL: @expm1
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-06+1.000001e-06i

func.func @expm1d() -> tensor<complex<f64>> {
  %c = mhlo.constant dense<(1.0e-50, 1.0e-50)> : tensor<complex<f64>>
  %expm1 = mhlo.exponential_minus_one %c : tensor<complex<f64>>
  return %expm1 : tensor<complex<f64>>
}

// CHECK-LABEL: @expm1d
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-50+1.000000e-50i

func.func @imag() -> tensor<f64> {
  %c = mhlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %imag = mhlo.imag %c : (tensor<complex<f64>>) -> tensor<f64>
  return %imag : tensor<f64>
}

// CHECK-LABEL: @imag
// CHECK-NEXT: Results
// CHECK-NEXT: 2.000000e+00

func.func @log1pf() -> tensor<complex<f32>> {
  %c = mhlo.constant dense<(1.0e-07, 1.0e-20)> : tensor<complex<f32>>
  %cos = mhlo.log_plus_one %c : tensor<complex<f32>>
  return %cos : tensor<complex<f32>>
}

// CHECK-LABEL: @log1p
// CHECK-NEXT: Results
// The accuracy of this is rather poor, but it matches numpy.
// CHECK-NEXT: 1.192093e-07+9.999999e-21i

func.func @log1pd() -> tensor<complex<f64>> {
  %c = mhlo.constant dense<(1.0e-07, 1.0e-20)> : tensor<complex<f64>>
  %cos = mhlo.log_plus_one %c : tensor<complex<f64>>
  return %cos : tensor<complex<f64>>
}

// CHECK-LABEL: @log1pd
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e-07+9.999999e-21i

func.func @logistic() -> tensor<5xcomplex<f32>> {
  %c = mhlo.constant dense<[(-1.0e5, 0.0), (-1.0, 0.0),
      (0.0, 0.0), (1.0, 0.0), (1.0e5, 0.0)]> : tensor<5xcomplex<f32>>
  %ret = mhlo.logistic %c : tensor<5xcomplex<f32>>
  return %ret : tensor<5xcomplex<f32>>
}

// CHECK-LABEL: @logistic
// CHECK-NEXT: Results
// CHECK-NEXT: [0.000000e+00+0.000000e+00i,
// CHECK-SAME:  2.689414e-01+0.000000e+00i,
// CHECK-SAME:  5.000000e-01+0.000000e+00i,
// CHECK-SAME:  7.310586e-01+0.000000e+00i,
// CHECK-SAME:  1.000000e+00+0.000000e+00i]

func.func @real() -> tensor<f64> {
  %c = mhlo.constant dense<(1.0, 2.0)> : tensor<complex<f64>>
  %real = mhlo.real %c : (tensor<complex<f64>>) -> tensor<f64>
  return %real : tensor<f64>
}

// CHECK-LABEL: @real
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @sin() -> tensor<complex<f64>> {
  %c = mhlo.constant dense<(1.0, 1.0)> : tensor<complex<f64>>
  %cos = mhlo.sine %c : tensor<complex<f64>>
  return %cos : tensor<complex<f64>>
}

// CHECK-LABEL: @sin
// CHECK-NEXT: Results
// CHECK-NEXT: 1.298458e+00+6.349639e-01i
