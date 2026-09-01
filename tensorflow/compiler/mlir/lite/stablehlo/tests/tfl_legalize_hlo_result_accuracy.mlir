// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file -verify-diagnostics

// Ops that request a non-default result accuracy must not be silently lowered
// to TFL kernels, which make no accuracy guarantees beyond the default.

// expected-error@+1 {{stablehlo to TFLite legalization failed.}}
module {
func.func @exp_highest(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.exponential'}}
  %0 = "stablehlo.exponential"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// -----

// expected-error@+1 {{stablehlo to TFLite legalization failed.}}
module {
func.func @expm1_tolerance(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.exponential_minus_one'}}
  %0 = "stablehlo.exponential_minus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e-05, rtol = 0.000000e+00, ulps = 1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// -----

// expected-error@+1 {{stablehlo to TFLite legalization failed.}}
module {
func.func @log1p_highest(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.log_plus_one'}}
  %0 = "stablehlo.log_plus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}

// -----

// expected-error@+1 {{stablehlo to TFLite legalization failed.}}
module {
func.func @cbrt_highest(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.cbrt'}}
  %0 = "stablehlo.cbrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00, rtol = 0.000000e+00, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}
