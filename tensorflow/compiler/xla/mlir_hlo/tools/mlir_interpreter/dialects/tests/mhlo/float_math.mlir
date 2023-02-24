// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @atan2() -> tensor<1xf32> {
  // Why would you ever do this?
  %c10 = mhlo.constant dense<10.0> : tensor<1xf32>
  %c1 = mhlo.constant dense<1.0> : tensor<1xf32>
  %ret = mhlo.atan2 %c10, %c1 : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @atan2
// CHECK-NEXT: Results
// CHECK-NEXT: [1.471128e+00]

func.func @cbrt() -> tensor<1xf32> {
  %c-27 = mhlo.constant dense<-27.0> : tensor<1xf32>
  %ret = mhlo.cbrt %c-27 : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @cbrt
// CHECK-NEXT: Results
// CHECK-NEXT: [-3.000000e+00]

func.func @ceil() -> tensor<1xf32> {
  %c = mhlo.constant dense<0.123> : tensor<1xf32>
  %ret = mhlo.ceil %c : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @ceil
// CHECK-NEXT: Results
// CHECK-NEXT: [1.000000e+00]

func.func @complex() -> tensor<complex<f32>> {
  %c1 = mhlo.constant dense<1.0> : tensor<f32>
  %c2 = mhlo.constant dense<2.0> : tensor<f32>
  %ret = mhlo.complex %c1, %c2 : tensor<complex<f32>>
  return %ret : tensor<complex<f32>>
}

// CHECK-LABEL: @complex
// CHECK-NEXT: Results
// CHECK-NEXT: <complex<f32>>: 1.000000e+00+2.000000e+00i

func.func @exp() -> tensor<1xf32> {
  %c = mhlo.constant dense<0.0> : tensor<1xf32>
  %ret = mhlo.exponential %c : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @exp
// CHECK-NEXT: Results
// CHECK-NEXT: [1.000000e+00]

func.func @floor() -> tensor<1xf32> {
  %c = mhlo.constant dense<3.123> : tensor<1xf32>
  %ret = mhlo.floor %c : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @floor
// CHECK-NEXT: Results
// CHECK-NEXT: [3.000000e+00]

func.func @is_finite() -> tensor<3xi1> {
  %c = mhlo.constant dense<[0x7FC00000, 2.0, 0x7F800000]> : tensor<3xf32>
  %is_finite = mhlo.is_finite %c : (tensor<3xf32>) -> tensor<3xi1>
  return %is_finite : tensor<3xi1>
}

// CHECK-LABEL: @is_finite
// CHECK-NEXT: Results
// CHECK-NEXT: [false, true, false]

func.func @log() -> tensor<1xf32> {
  %c = mhlo.constant dense<1.0> : tensor<1xf32>
  %ret = mhlo.log %c : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @log
// CHECK-NEXT: Results
// CHECK-NEXT: [0.000000e+00]

func.func @logistic() -> tensor<5xf32> {
  %c = mhlo.constant dense<[-1.0e5, -1.0, 0.0, 1.0, 1.0e5]> : tensor<5xf32>
  %ret = mhlo.logistic %c : tensor<5xf32>
  return %ret : tensor<5xf32>
}

// CHECK-LABEL: @logistic
// CHECK-NEXT: Results
// CHECK-NEXT: [0.000000e+00, 2.689414e-01, 5.000000e-01, 7.310586e-01, 1.000000e+00]

func.func @maximum() -> tensor<f32> {
  %c1 = mhlo.constant dense<1.0> : tensor<f32>
  %c2 = mhlo.constant dense<2.0> : tensor<f32>
  %ret = mhlo.maximum %c1, %c2 : tensor<f32>
  return %ret : tensor<f32>
}

// CHECK-LABEL: @maximum
// CHECK-NEXT: Results
// CHECK-NEXT: 2.000000e+00

func.func @minimum() -> tensor<f32> {
  %c1 = mhlo.constant dense<1.0> : tensor<f32>
  %c2 = mhlo.constant dense<2.0> : tensor<f32>
  %ret = mhlo.minimum %c1, %c2 : tensor<f32>
  return %ret : tensor<f32>
}

// CHECK-LABEL: @minimum
// CHECK-NEXT: Results
// CHECK-NEXT: 1.000000e+00

func.func @pow() -> tensor<f32> {
  %c2 = mhlo.constant dense<2.0> : tensor<f32>
  %c5 = mhlo.constant dense<5.0> : tensor<f32>
  %pow = mhlo.power %c2, %c5 : tensor<f32>
  return %pow : tensor<f32>
}

// CHECK-LABEL: @pow
// CHECK-NEXT: Results
// CHECK-NEXT: 3.200000e+01

func.func @rem() -> tensor<8xf32> {
  %0 = mhlo.constant dense<[-2.5, 2.25, -10.0, 6.0, 3.0, 3.0, -1.0, -8.0]> : tensor<8xf32>
  %1 = mhlo.constant dense<[10.0, 1.0, 10.0, -6.0, 2.0, -2.0, 7.0, -4.0]> : tensor<8xf32>
  %2 = mhlo.remainder %0, %1 : tensor<8xf32>
  func.return %2 : tensor<8xf32>
}

// CHECK-LABEL: @rem
// CHECK-NEXT: Results
// CHECK-NEXT: [-2.500000e+00, 2.500000e-01, -0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, -1.000000e+00, -0.000000e+00]

func.func @rem_inf() -> (tensor<1xf32>, tensor<1xf32>) {
  %0 = mhlo.constant dense<[1.0]> : tensor<1xf32>
  %1 = mhlo.constant dense<[0x7F800000]> : tensor<1xf32>
  %2 = mhlo.remainder %0, %1 : tensor<1xf32>
  func.return %2, %1 : tensor<1xf32>, tensor<1xf32>
}

// CHECK-LABEL: @rem_inf
// CHECK-NEXT: Results
// CHECK-NEXT: [1.000000e+00]
// CHECK-NEXT: [INF]

func.func @round_nearest_afz() -> tensor<4xf32> {
  %c = mhlo.constant dense<[-1.5, -0.5, 0.5, 1.5]> : tensor<4xf32>
  %ret = mhlo.round_nearest_afz %c : tensor<4xf32>
  return %ret : tensor<4xf32>
}

// CHECK-LABEL: @round_nearest_afz
// CHECK-NEXT: Results
// CHECK-NEXT: [-2.000000e+00, -1.000000e+00, 1.000000e+00, 2.000000e+00]

func.func @round_nearest_even() -> tensor<5xf32> {
  %c = mhlo.constant dense<[-1.5, -0.5, 0.5, 0.6, 1.5]> : tensor<5xf32>
  %ret = mhlo.round_nearest_even %c : tensor<5xf32>
  return %ret : tensor<5xf32>
}

// CHECK-LABEL: @round_nearest_even
// CHECK-NEXT: Results
// CHECK-NEXT: [-2.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00]

func.func @rsqrt() -> tensor<1xf32> {
  %c4 = mhlo.constant dense<4.0> : tensor<1xf32>
  %ret = mhlo.rsqrt %c4 : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @rsqrt
// CHECK-NEXT: Results
// CHECK-NEXT: [5.000000e-01]

func.func @sign() -> tensor<3xf32> {
  %c = mhlo.constant dense<[-1.0, 2.0, 0x7F800000]> : tensor<3xf32>
  %ret = mhlo.sign %c : tensor<3xf32>
  return %ret : tensor<3xf32>
}

// CHECK-LABEL: @sign
// CHECK-NEXT: Results
// CHECK-NEXT: [-1.000000e+00, 1.000000e+00, 1.000000e+00]

func.func @tanh() -> tensor<1xf32> {
  %c = mhlo.constant dense<[1.0]> : tensor<1xf32>
  %ret = mhlo.tanh %c : tensor<1xf32>
  return %ret : tensor<1xf32>
}

// CHECK-LABEL: @tanh
// CHECK-NEXT: Results
// CHECK-NEXT: [7.615942e-01]
