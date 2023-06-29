// RUN: mlir-hlo-opt -split-input-file -mhlo-test-unfuse-batch-norm -cse -verify-diagnostics %s | FILECHECK_OPTS="" FileCheck --enable-var-scope %s

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
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.rsqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER:.+]] = mhlo.multiply %[[STDDEV]], %[[SCALE]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER_MEAN:.+]] = mhlo.multiply %[[MULTIPLIER]], %[[MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[RHS:.+]] = mhlo.subtract %[[OFFSET]], %[[MULTIPLIER_MEAN]] : tensor<256xf32>
  // CHECK-DAG: %[[MULTIPLIER_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[MULTIPLIER]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[RHS_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[RHS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_MULTIPLIER:.+]] = mhlo.multiply %[[X]], %[[MULTIPLIER_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_MULTIPLIER]], %[[RHS_BCAST]] : tensor<4x256xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<4x256xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_2D_inner_features
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
func.func @batchNormTraining_2D_inner_features(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>) -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[NUM_REDUCE:.+]] = mhlo.constant dense<4.000000e+00> : tensor<256xf32>
  // CHECK-DAG: %[[SumX:.+]] = mhlo.reduce(%[[X]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0] : (tensor<4x256xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[X2:.+]] = mhlo.multiply %[[X]], %[[X]] : tensor<4x256xf32>
  // CHECK-DAG: %[[SumX2:.+]] = mhlo.reduce(%[[X2]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0] : (tensor<4x256xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[EX:.+]] = mhlo.divide %[[SumX]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[EX2:.+]] = mhlo.divide %[[SumX2]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[E2X:.+]] = mhlo.multiply %[[EX]], %[[EX]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE:.+]] = mhlo.subtract %[[EX2]], %[[E2X]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[EX]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = mhlo.subtract %[[X]], %[[EX_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = mhlo.divide %[[X_CENTER]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED:.+]] = mhlo.multiply %[[X_NORMED]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED_OFFSET:.+]] = mhlo.add %[[X_NORMED_SCALED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  // CHECK-DAG: return %[[X_NORMED_SCALED_OFFSET]], %[[EX]], %[[VARIANCE]] : tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>
}

// -----
// CHECK-LABEL: @batchNormInference_4D_middle_features
// CHECK-SAME: %[[X:[^:]+]]
// CHECK-SAME: %[[SCALE:[^:]+]]
func.func @batchNormInference_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>) {
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<3x4x256x6xf32>
  func.return %0 : tensor<3x4x256x6xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_4D_middle_features
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
func.func @batchNormTraining_4D_middle_features(
    %x: tensor<3x4x256x6xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>)
    -> (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[NUM_REDUCE:.+]] = mhlo.constant dense<7.200000e+01> : tensor<256xf32>
  // CHECK-DAG: %[[SumX:.+]] = mhlo.reduce(%[[X]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0, 1, 3] : (tensor<3x4x256x6xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[X2:.+]] = mhlo.multiply %[[X]], %[[X]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[SumX2:.+]] = mhlo.reduce(%[[X2]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0, 1, 3] : (tensor<3x4x256x6xf32>, tensor<f32>) -> tensor<256xf32>
  // CHECK-DAG: %[[EX:.+]] = mhlo.divide %[[SumX]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[EX2:.+]] = mhlo.divide %[[SumX2]], %[[NUM_REDUCE]] : tensor<256xf32>
  // CHECK-DAG: %[[E2X:.+]] = mhlo.multiply %[[EX]], %[[EX]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE:.+]] = mhlo.subtract %[[EX2]], %[[E2X]] : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[EX]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_CENTER:.+]] = mhlo.subtract %[[X]], %[[EX_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[STDDEV]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED:.+]] = mhlo.divide %[[X_CENTER]], %[[STDDEV_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[SCALE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED:.+]] = mhlo.multiply %[[X_NORMED]], %[[SCALE_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[OFFSET]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<3x4x256x6xf32>
  // CHECK-DAG: %[[X_NORMED_SCALED_OFFSET:.+]] = mhlo.add %[[X_NORMED_SCALED]], %[[OFFSET_BCAST]] : tensor<3x4x256x6xf32>
  // CHECK-DAG: return %[[X_NORMED_SCALED_OFFSET]], %[[EX]], %[[VARIANCE]] : tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<3x4x256x6xf32>, tensor<256xf32>, tensor<256xf32>
}

// -----
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

// -----
// Validate that epsilon is properly promoted to f64
// CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.000000e+00> : tensor<256xf64>
func.func @batchNormTraining_f64(
    %x: tensor<4x256xf64>, %scale: tensor<256xf64>, %offset: tensor<256xf64>)
    -> (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>) {
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>) -> (tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf64>, tensor<256xf64>, tensor<256xf64>
}

// -----
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

// -----
// Validate that epsilon is properly down to f16
// CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.000000e+00> : tensor<256xf16>
func.func @batchNormTraining_f16(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>)
    -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) {
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.0 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>)
  func.return %0#0, %0#1, %0#2 : tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>
}

// -----
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

// -----
// Validate that epsilon is overflow
func.func @batchNormTraining_f16_overflow(
    %x: tensor<4x256xf16>, %scale: tensor<256xf16>, %offset: tensor<256xf16>)
    -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) {
  // expected-warning @+1 {{Could not convert batch_norm epsilon to target fp type: opStatus = 24}}
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 0.00000001 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>) -> (tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>)
  func.return %0#0, %0#1, %0#2 :tensor<4x256xf16>, tensor<256xf16>, tensor<256xf16>
}

// -----
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
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.rsqrt %[[VARIANCE_EPS]] : tensor<?xf32>
  // CHECK-DAG: %[[MULTIPLIER:.+]] = mhlo.multiply %[[STDDEV]], %[[SCALE]] : tensor<?xf32>
  // CHECK-DAG: %[[MULTIPLIER_MEAN:.+]] = mhlo.multiply %[[MULTIPLIER]], %[[MEAN]] : tensor<?xf32>
  // CHECK-DAG: %[[RHS:.+]] = mhlo.subtract %[[OFFSET]], %[[MULTIPLIER_MEAN]] : tensor<?xf32>
  // CHECK-DAG: %[[X_SHAPE:.+]] = shape.shape_of %[[X]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
  // CHECK-DAG: %[[MULTIPLIER_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[MULTIPLIER]], %[[X_SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RHS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[RHS]], %[[X_SHAPE]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_MULTIPLIER:.+]] = mhlo.multiply %[[X]], %[[MULTIPLIER_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_MULTIPLIER]], %[[RHS_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: return %[[RESULT]]
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 0.001 : f32, feature_index = 1 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
        tensor<?xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----
// CHECK-LABEL: @batchNormTraining_dynamic_shape
// Validate that dynamic shapes are handled properly.
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
func.func @batchNormTraining_dynamic_shape(
    %x: tensor<?x?x?x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>)
    -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>) {
  // CHECK-DAG: %[[ZERO:.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[EPS:.+]] = mhlo.constant dense<1.001000e-05> : tensor<f32>
  // CHECK-DAG: %[[SCALE_SHAPE:.+]] = shape.shape_of %[[SCALE]] : tensor<?xf32> -> tensor<1xindex>
  // CHECK-DAG: %[[EPS_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[EPS]], %[[SCALE_SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[X_SHAPE:.+]] = shape.shape_of %[[X]] : tensor<?x?x?x?xf32> -> tensor<4xindex>
  // CHECK-DAG: %[[X_SIZE:.+]] = shape.num_elements %[[X_SHAPE]] : tensor<4xindex> -> index
  // CHECK-DAG: %[[SCALE_SIZE:.+]] = shape.num_elements %[[SCALE_SHAPE]] : tensor<1xindex> -> index
  // CHECK-DAG: %[[REDUCE_SIZE:.+]] = shape.div %[[X_SIZE]], %[[SCALE_SIZE]] : index, index -> index
  // CHECK-DAG: %[[INDEX_CAST:.+]] = arith.index_cast %[[REDUCE_SIZE]] : index to i64
  // CHECK-DAG: %[[REDUCE_SIZE_TENSOR:.+]] = tensor.from_elements %[[INDEX_CAST]] : tensor<1xi64>
  // CHECK-DAG: %[[REDUCE_SIZE_TENSOR_FP:.+]] = mhlo.convert %[[REDUCE_SIZE_TENSOR]] : (tensor<1xi64>) -> tensor<1xf32>
  // CHECK-DAG: %[[REDUCE_SIZE_RESHAPE:.+]] = mhlo.reshape %[[REDUCE_SIZE_TENSOR_FP]] : (tensor<1xf32>) -> tensor<f32>
  // CHECK-DAG: %[[REDUCE_SIZE_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[REDUCE_SIZE_RESHAPE]], %[[SCALE_SHAPE]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-DAG: %[[X_SUM:.+]] = mhlo.reduce(%[[X]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0, 1, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK-DAG: %[[X2:.+]] = mhlo.multiply %[[X]], %[[X]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X2_SUM:.+]] = mhlo.reduce(%[[X2]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0, 1, 3] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK-DAG: %[[EX:.+]] = mhlo.divide %[[X_SUM]], %[[REDUCE_SIZE_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[EX2:.+]] = mhlo.divide %[[X2_SUM]], %[[REDUCE_SIZE_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[EX_2:.+]] = mhlo.multiply %[[EX]], %[[EX]] : tensor<?xf32>
  // CHECK-DAG: %[[VARX:.+]] = mhlo.subtract %[[EX2]], %[[EX_2]] : tensor<?xf32>
  // CHECK-DAG: %[[VARX_EPS:.+]] = mhlo.add %[[VARX]], %[[EPS_BCAST]] : tensor<?xf32>
  // CHECK-DAG: %[[STDX:.+]] = mhlo.sqrt %[[VARX_EPS]] : tensor<?xf32>
  // CHECK-DAG: %[[EX_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[EX]], %[[X_SHAPE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SUB_EX:.+]] = mhlo.subtract %[[X]], %[[EX_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[STDX_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[STDX]], %[[X_SHAPE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_CENTOR:.+]] = mhlo.divide %[[X_SUB_EX]], %[[STDX_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[SCALE]], %[[X_SHAPE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[X_SCALED:.+]] = mhlo.multiply %[[X_CENTOR]], %[[SCALE_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[OFFSET]], %[[X_SHAPE]]) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK-DAG: %[[RESULT:.+]] = mhlo.add %[[X_SCALED]], %[[OFFSET_BCAST]] : tensor<?x?x?x?xf32>
  // CHECK-DAG: return %[[RESULT]], %[[EX]], %[[VARX]] : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>
  %0:3 = "mhlo.batch_norm_training"(%x, %scale, %offset)
      {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>
}
