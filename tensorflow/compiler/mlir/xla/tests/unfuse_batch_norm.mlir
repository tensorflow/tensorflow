// RUN: tf-opt -split-input-file -test-xla-unfuse-batch-norm -verify-diagnostics %s | FileCheck --enable-var-scope --dump-input=fail %s

// CHECK-LABEL: @batchNormInference_2D_inner_features
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func @batchNormInference_2D_inner_features(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS:.+]] = xla_hlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[EPS]]) : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = xla_hlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = "xla_hlo.sqrt"(%[[VARIANCE_EPS]]) : (tensor<256xf32>) -> tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[MEAN]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = xla_hlo.sub %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = xla_hlo.mul %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = xla_hlo.div %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = xla_hlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  return %0 : tensor<4x256xf32>
}

// -----
// CHECK-LABEL: @batchNormInference_4D_middle_features
// Just validate that one of the broadcasts happens correctly and rely on
// the verifier to enforce the rest.
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
// CHECK-DAG: %[[SCALE_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
func @batchNormInference_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>) {
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<3x4x256x6xf32>
  return %0 : tensor<3x4x256x6xf32>
}

// -----
// CHECK-LABEL: @batchNormInference_f64
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = xla_hlo.constant dense<1.000000e+00> : tensor<f64>
func @batchNormInference_f64(
    %x: tensor<4x256xf64>, %scale: tensor<256xf64>, %offset: tensor<256xf64>,
    %mean: tensor<256xf64>, %variance: tensor<256xf64>)
    -> (tensor<4x256xf64>) {
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>, tensor<256xf64>,
        tensor<256xf64>) -> tensor<4x256xf64>
  return %0 : tensor<4x256xf64>
}

// -----
// CHECK-LABEL: @batchNormInference_f16
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = xla_hlo.constant dense<1.000000e+00> : tensor<f16>
func @batchNormInference_f16(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  return %0 : tensor<4x256xf16>
}

// -----
// Validate that epsilon is properly promoted to f64
func @batchNormInference_f16_overflow(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  // expected-warning @+2 {{Could not convert batch_norm epsilon to target fp type: opStatus = 24}}
  // expected-error @+1 {{failed to legalize operation 'xla_hlo.batch_norm_inference' that was explicitly marked illegal}}
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.00000001 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  return %0 : tensor<4x256xf16>
}
