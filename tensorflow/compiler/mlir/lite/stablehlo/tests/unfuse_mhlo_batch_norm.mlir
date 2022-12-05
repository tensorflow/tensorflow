// RUN: tf-mhlo-tfl-opt %s -unfuse-mhlo-batch-norm-pass -cse -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @batchNormInference_2D_inner_features
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batchNormInference_2D_inner_features(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS_RSQRT:.+]] = mhlo.rsqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER:.+]] = mhlo.multiply %[[VARIANCE_EPS_RSQRT]], %[[SCALE]] : tensor<256xf32>
  // CHECK-DAG: %[[MUL_MEAN:.+]] = mhlo.multiply %[[MULTIPLIER]], %[[MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[RHS:.+]] = mhlo.subtract %[[OFFSET]], %[[MUL_MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[MULTIPLIER]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[RHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[RHS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = mhlo.multiply %[[X]], %[[MULTIPLIER_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_NORMED]], %[[RHS_BCAST]] : tensor<4x256xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<4x256xf32>
}

// CHECK-LABEL: @batchNormInference_4D_middle_features
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batchNormInference_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>) {
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS_RSQRT:.+]] = mhlo.rsqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER:.+]] = mhlo.multiply %[[VARIANCE_EPS_RSQRT]], %[[SCALE]] : tensor<256xf32>
  // CHECK-DAG: %[[MUL_MEAN:.+]] = mhlo.multiply %[[MULTIPLIER]], %[[MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[RHS:.+]] = mhlo.subtract %[[OFFSET]], %[[MUL_MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[MULTIPLIER]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[RHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[RHS]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<3x4x256x6xf32>
  func.return %0 : tensor<3x4x256x6xf32>
}

// CHECK-LABEL: @batchNormInference_dynamic_shape
// Validate that dynamic shapes are handled properly.
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @batchNormInference_dynamic_shape(
    %x: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>,
    %mean: tensor<?xf32>, %variance: tensor<?xf32>)
    -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.000000e-03> : tensor<f32>
  // CHECK-DAG: %[[VAR_SHAPE:.+]] = shape.shape_of %[[VARIANCE]] : tensor<?xf32> -> tensor<1xindex>
  // CHECK-DAG: %[[EPS_BCAST:.+]] =  "mhlo.dynamic_broadcast_in_dim"(%[[EPS]], %[[VAR_SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[R_STDDEV:.+]] = mhlo.rsqrt %[[VARIANCE_EPS]] : tensor<?xf32>
  // CHECK-DAG: %[[MULTIPLIER:.+]] = mhlo.multiply %[[R_STDDEV]], %[[SCALE]] : tensor<?xf32>
  // CHECK-DAG: %[[MUL_MEAN:.+]] = mhlo.multiply %[[MULTIPLIER]], %[[MEAN]] : tensor<?xf32>
  // CHECK-DAG: %[[RHS:.+]] = mhlo.subtract %[[OFFSET]], %[[MUL_MEAN]] : tensor<?xf32>
  // CHECK-DAG: %[[X_SHAPE:.+]] = shape.shape_of %[[X]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
  // CHECK-DAG: %[[MULTIPLIER_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[MULTIPLIER]], %[[X_SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[RHS]], %[[X_SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = mhlo.multiply %[[X]], %[[MULTIPLIER_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_NORMED]], %[[RHS_BCAST]] : tensor<?x?x?x?xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.001 : f32, feature_index = 1 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
        tensor<?xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @batchNormInference_f64
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.000000e+00> : tensor<256xf64>
func.func @batchNormInference_f64(
    %x: tensor<4x256xf64>, %scale: tensor<256xf64>, %offset: tensor<256xf64>,
    %mean: tensor<256xf64>, %variance: tensor<256xf64>)
    -> (tensor<4x256xf64>) {
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>, tensor<256xf64>,
        tensor<256xf64>) -> tensor<4x256xf64>
  func.return %0 : tensor<4x256xf64>
}

// CHECK-LABEL: @batchNormInference_f16
// Validate that epsilon is properly down to f16
// CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.000000e+00> : tensor<256xf16>
func.func @batchNormInference_f16(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  func.return %0 : tensor<4x256xf16>
}

// Validate that epsilon is overflow
func.func @batchNormInference_f16_overflow(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>,
    %mean: tensor<256xf16>, %variance: tensor<256xf16>)
    -> (tensor<4x256xf16>) {
  // expected-warning @+1 {{Could not convert batch_norm epsilon to target fp type: opStatus = 24}}
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.00000001 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>, tensor<256xf16>,
        tensor<256xf16>) -> tensor<4x256xf16>
  func.return %0 : tensor<4x256xf16>
}
