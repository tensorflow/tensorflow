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
  // CHECK-DAG: %[[EPS_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[EPS]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = xla_hlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = "xla_hlo.sqrt"(%[[VARIANCE_EPS]]) : (tensor<256xf32>) -> tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "xla_hlo.broadcast_in_dim"(%[[MEAN]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = xla_hlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = xla_hlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = xla_hlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
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

// -----
// CHECK-LABEL: @batchNormInference_dynamic_shape
// Validate that dynamic shapes are handled properly.
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func @batchNormInference_dynamic_shape(
    %x: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>,
    %mean: tensor<?xf32>, %variance: tensor<?xf32>)
    -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[EPS:.+]] = xla_hlo.constant dense<1.000000e-03> : tensor<f32>
  // CHECK-DAG: %[[DIM:.+]] = dim %[[VARIANCE]], 0 : tensor<?xf32>
  // CHECK-DAG: %[[INDEX_CAST:.+]] = index_cast %[[DIM]] : index to i32
  // CHECK-DAG: %[[TO_DIM_TENSOR:.+]] = "xla_hlo.scalars_to_dimension_tensor"(%[[INDEX_CAST]]) : (i32) -> tensor<1xi32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] =  "xla_hlo.dynamic_broadcast_in_dim"(%[[EPS]], %[[TO_DIM_TENSOR]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = xla_hlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = "xla_hlo.sqrt"(%[[VARIANCE_EPS]]) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK-DAG: %[[INPUT_DIM_0:.+]] = dim %[[X]], 0 : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[INPUT_INDEX_CAST_0:.+]] = index_cast %[[INPUT_DIM_0]] : index to i32
  // CHECK-DAG: %[[INPUT_DIM_1:.+]] = dim %[[X]], 1 : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[INPUT_INDEX_CAST_1:.+]] = index_cast %[[INPUT_DIM_1]] : index to i32
  // CHECK-DAG: %[[INPUT_DIM_2:.+]] = dim %[[X]], 2 : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[INPUT_INDEX_CAST_2:.+]] = index_cast %[[INPUT_DIM_2]] : index to i32
  // CHECK-DAG: %[[INPUT_DIM_3:.+]] = dim %[[X]], 3 : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[INPUT_INDEX_CAST_3:.+]] = index_cast %[[INPUT_DIM_3]] : index to i32
  // CHECK-DAG: %[[TO_INPUT_DIM_TENSOR:.+]] = "xla_hlo.scalars_to_dimension_tensor"(%[[INPUT_INDEX_CAST_0]], %[[INPUT_INDEX_CAST_1]], %[[INPUT_INDEX_CAST_2]], %[[INPUT_INDEX_CAST_3]]) : (i32, i32, i32, i32) -> tensor<4xi32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[STDDEV]], %[[TO_INPUT_DIM_TENSOR]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[SCALE]], %[[TO_INPUT_DIM_TENSOR]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[OFFSET]], %[[TO_INPUT_DIM_TENSOR]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[MEAN]], %[[TO_INPUT_DIM_TENSOR]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = xla_hlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = xla_hlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = xla_hlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = xla_hlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<?x?x?x?xf32>
  %0 = "xla_hlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.001 : f32, feature_index = 1 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
        tensor<?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
