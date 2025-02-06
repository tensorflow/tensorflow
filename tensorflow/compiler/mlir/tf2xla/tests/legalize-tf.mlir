// RUN: tf-opt "-xla-legalize-tf=legalize-chlo=false" -split-input-file %s | FILECHECK_OPTS="" FileCheck %s
// RUN: tf-opt "-xla-legalize-tf=legalize-chlo=true" -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix CHLO
// This test runs twice:
//   1. Through FILECHECK_OPTS="" FileCheck with chlo legalization disabled since verifying
//      that the chlo ops emit produces more useful tests.
//   2. With chlo legalization enabled, verifying diagnostics to pick up any
//      issues with the full lowering (can catch some broadcasting corner
//      cases which emit with a warning).

//===----------------------------------------------------------------------===//
// BatchNorm op legalizations.
//===----------------------------------------------------------------------===//

// -----

// fusedBatchNormV2 is almost identical to fusedBatchNormV3 (and uses the same
// code), so only do a couple of basic checks.

// CHECK-LABEL: fusedBatchNormV2_noTraining
func.func @fusedBatchNormV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV2_training
func.func @fusedBatchNormV2_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.constant
  // CHECK: chlo.broadcast_multiply %[[VAR]], {{.*}} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining
func.func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining_mixedPrecision
// CHECK-SAME:  ([[X:%.*]]: tensor<8x8x8x8xbf16>, [[SCALE:%.*]]: tensor<8xf32>, [[OFFSET:%.*]]: tensor<8xf32>, [[MEAN:%.*]]: tensor<8xf32>, [[VARIANCE:%.*]]: tensor<8xf32>)
func.func @fusedBatchNormV3_noTraining_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) {
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert [[X]] : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK: [[Y:%.*]] = "mhlo.batch_norm_inference"([[CONVERT_X]], [[SCALE]], [[OFFSET]], [[MEAN]], [[VARIANCE]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)
  // CHECK: [[Y_CONVERT:%.*]] = mhlo.convert [[Y]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK: [[DUMMY:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<0xf32>
  // CHECK: [[DUMMY_CAST:%.*]] = tensor.cast [[DUMMY]] : tensor<0xf32> to tensor<*xf32>
  // CHECK: return [[Y_CONVERT]], [[MEAN]], [[VARIANCE]], [[MEAN]], [[VARIANCE]], [[DUMMY_CAST]]
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training
func.func @fusedBatchNormV3_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.constant
  // CHECK: chlo.broadcast_multiply %[[VAR]], {{.*}} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: func @fusedBatchNormV3_training_batchVariance
func.func @fusedBatchNormV3_training_batchVariance(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: return %[[VAR]]
  func.return %0#4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_exponentialAvgFactor
func.func @fusedBatchNormV3_training_exponentialAvgFactor(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 0.8 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: %[[FACTOR:.*]] = mhlo.constant dense<1.00195694>
  // CHECK: %[[CORRECTED_VAR:.*]] = chlo.broadcast_multiply %[[VAR]], %[[FACTOR]]

  // CHECK-DAG: %[[ALPHA:.*]] = mhlo.constant dense<0.199999988>
  // CHECK-DAG: %[[BETA:.*]] = mhlo.constant dense<8.000000e-01>

  // CHECK: %[[ALPHA_MUL_OLD_MEAN:.*]] = chlo.broadcast_multiply %[[ALPHA]], %arg3
  // CHECK: %[[BETA_MUL_BATCH_MEAN:.*]] = chlo.broadcast_multiply %[[BETA]], %[[MEAN]]
  // CHECK: %[[NEW_BATCH_MEAN:.*]] = chlo.broadcast_add %[[ALPHA_MUL_OLD_MEAN]], %[[BETA_MUL_BATCH_MEAN]]

  // CHECK: %[[ALPHA_MUL_OLD_VAR:.*]] = chlo.broadcast_multiply %[[ALPHA]], %arg4
  // CHECK: %[[BETA_MUL_CORRECTED_VAR:.*]] = chlo.broadcast_multiply %[[BETA]], %[[CORRECTED_VAR]]
  // CHECK: %[[NEW_BATCH_VAR:.*]] = chlo.broadcast_add %[[ALPHA_MUL_OLD_VAR]], %[[BETA_MUL_CORRECTED_VAR]]

  // CHECK: return %[[NEW_BATCH_MEAN]], %[[NEW_BATCH_VAR]], %[[MEAN]], %[[VAR]]
  func.return %0#1, %0#2, %0#3, %0#4 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_mixedPrecision
func.func @fusedBatchNormV3_training_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK: mhlo.convert %arg0 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.convert {{.*}} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_NCHW
func.func @fusedBatchNormV3_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_NDHWC
func.func @fusedBatchNormV3_NDHWC(%arg0: tensor<8x8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8x8xf32>) {
  // CHECK: feature_index = 4 : i64
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NDHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining_dynamic_supported
func.func @fusedBatchNormV3_noTraining_dynamic_supported(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>) -> (tensor<?x?x?x?xf32>) {
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_dynamic_unsupported1
func.func @fusedBatchNormV3_training_dynamic_unsupported1(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>) -> (tensor<?x?x?x?xf32>) {
  // CHECK: tf.FusedBatchNormV3
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_dynamic_unsupported2
func.func @fusedBatchNormV3_training_dynamic_unsupported2(%arg0: tensor<?x6x?x?xf32>, %arg1: tensor<6xf32>, %arg2: tensor<6xf32>, %arg3: tensor<6xf32>, %arg4: tensor<6xf32>) -> (tensor<?x6x?x?xf32>) {
  // CHECK: tf.FusedBatchNormV3
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<?x6x?x?xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> (tensor<?x6x?x?xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>)
  func.return %0#0 : tensor<?x6x?x?xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGrad_noTraining
func.func @fusedBatchNormGrad_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = mhlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = array<i64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = mhlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = mhlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = mhlo.convert %[[grad]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = mhlo.reduce(%[[cgrad]] init: %[[init2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = mhlo.convert %[[red2]] : tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[mul3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGrad_Training
func.func @fusedBatchNormGrad_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_noTraining
func.func @fusedBatchNormGradV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = mhlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = array<i64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = mhlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = mhlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = mhlo.convert %[[grad]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = mhlo.reduce(%[[cgrad]] init: %[[init2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = mhlo.convert %[[red2]] : tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[mul3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_Training
func.func @fusedBatchNormGradV2_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_noTraining_mixed_precision
func.func @fusedBatchNormGradV2_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = mhlo.convert {{.*}} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_Training_mixed_precision
func.func @fusedBatchNormGradV2_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining
func.func @fusedBatchNormGradV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = mhlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = array<i64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = mhlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = mhlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = mhlo.convert %[[grad]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = mhlo.reduce(%[[cgrad]] init: %[[init2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = mhlo.convert %[[red2]] : tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[mul3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training
func.func @fusedBatchNormGradV3_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : tensor<8x8x8x8xf32>
  // CHECK: return %[[x_backprop]]
  // CHECK-SAME: tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<0xf32>, tensor<*xf32>)
  func.return %0#0, %0#3, %0#4 : tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_mixed_precision
func.func @fusedBatchNormGradV3_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = mhlo.convert {{.*}} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training_mixed_precision
func.func @fusedBatchNormGradV3_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_NCHW
func.func @fusedBatchNormGradV3_noTraining_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = mhlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = array<i64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = mhlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = mhlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: mhlo.constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = mhlo.convert %[[grad]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = mhlo.reduce(%[[cgrad]] init: %[[init2]]) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = mhlo.convert %[[red2]] : tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[mul3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training_NCHW
func.func @fusedBatchNormGradV3_Training_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %{{.*}} = "mhlo.batch_norm_grad"(%{{.*}}, %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

//===----------------------------------------------------------------------===//
// Bias op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @biasAdd_default
func.func @biasAdd_default(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = shape.to_extent_tensor %[[ARG0_SHAPE]]
  // CHECK: %[[ARG1_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = mhlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_NHWC
func.func @biasAdd_NHWC(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = shape.to_extent_tensor %[[ARG0_SHAPE]]
  // CHECK: %[[ARG1_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = mhlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_NCHW
func.func @biasAdd_NCHW(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = shape.to_extent_tensor %[[ARG0_SHAPE]]
  // CHECK: %[[ARG1_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = mhlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_dynamic
func.func @biasAdd_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?x?xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = shape.to_extent_tensor %[[ARG0_SHAPE]]
  // CHECK: %[[ARG1_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = mhlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW"} : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  func.return %0 : tensor<?x?x?x?xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_partial_dynamic
func.func @biasAdd_partial_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<512xi32>) -> tensor<?x?x?x512xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = shape.to_extent_tensor %[[ARG0_SHAPE]]
  // CHECK: %[[ARG1_BCAST:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = mhlo.add %arg0, %[[ARG1_BCAST]]
  // CHECK: %[[CAST:.+]] = tensor.cast %[[RESULT]] : tensor<?x?x?x?xi32> to tensor<?x?x?x512xi32>
  // CHECK: return %[[CAST]] : tensor<?x?x?x512xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<?x?x?x?xi32>, tensor<512xi32>) -> tensor<?x?x?x512xi32>
  func.return %0 : tensor<?x?x?x512xi32>
}


//===----------------------------------------------------------------------===//
// ClipByValue
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @clip
func.func @clip(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<f32> {
  // CHECK: [[VAL:%.+]] = mhlo.clamp %arg1, %arg0, %arg2

  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return [[VAL]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @clip_dynamic
func.func @clip_dynamic(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: [[CLAMP:%.+]] = mhlo.clamp %arg1, %arg0, %arg2
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @clip_static_broadcast
func.func @clip_static_broadcast(%arg0 : tensor<5xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<5xf32> {
  // CHECK-DAG: [[SHPIDX:%.+]] = mhlo.constant dense<5>
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, [[SHPIDX]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}>
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg2, [[SHPIDX]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}>
  // CHECK-DAG: [[CLAMP:%.+]] = mhlo.clamp [[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]]
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<5xf32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<5xf32>
}


// CHECK-LABEL: @clip_dynamic_broadcast
func.func @clip_dynamic_broadcast(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?xf32> {
  // CHECK: [[SHP:%.+]] = shape.shape_of %arg0
  // CHECK: [[SHPIDX:%.+]] = arith.index_cast [[SHP]] : tensor<1xindex> to tensor<1xi32>
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, [[SHPIDX]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}>
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg2, [[SHPIDX]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}>
  // CHECK-DAG: [[CLAMP:%.+]] = mhlo.clamp [[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]]
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// DiagPart
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @diag_part
// CHECK-SAME: %[[ARG:.*]]: tensor<4x3x4x3xf32>
func.func @diag_part(%arg0: tensor<4x3x4x3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[RS:.*]] = mhlo.reshape %[[ARG]] : (tensor<4x3x4x3xf32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[IOTA0:.*]] = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[IOTA1:.*]] = "mhlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[COMP:.*]] = mhlo.compare EQ, %[[IOTA0]], %[[IOTA1]], NOTYPE : (tensor<12x12xi32>, tensor<12x12xi32>) -> tensor<12x12xi1>
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZERO_MAT:.*]] = "mhlo.broadcast"(%[[ZERO]]) <{broadcast_sizes = dense<12> : tensor<2xi64>}> : (tensor<f32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[SEL:.*]] = mhlo.select %[[COMP]], %[[RS]], %[[ZERO_MAT]] : tensor<12x12xi1>, tensor<12x12xf32>
  // CHECK-DAG: %[[RED:.*]] = mhlo.reduce(%[[SEL]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0] : (tensor<12x12xf32>, tensor<f32>) -> tensor<12xf32>
  // CHECK-DAG:  %[[RES:.*]] = mhlo.reshape %[[RED]] : (tensor<12xf32>) -> tensor<4x3xf32>
  // CHECK-DAG:  return %[[RES]] : tensor<4x3xf32>
  %0 = "tf.DiagPart"(%arg0) : (tensor<4x3x4x3xf32>) -> tensor<4x3xf32>
  func.return %0: tensor<4x3xf32>
}

//===----------------------------------------------------------------------===//
// MatrixDiagPart
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @matrix_diag_part
// CHECK-SAME: %[[ARG:.*]]: tensor<7x140x128xi32>
func.func @matrix_diag_part(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  // CHECK-DAG: %[[V0:.*]] = mhlo.constant dense<42> : tensor<i32>
  // CHECK-DAG: %[[V1:.*]] = mhlo.constant dense<[-10, 11]> : tensor<2xi32>
  // CHECK-DAG: %[[V2:.*]] = "mhlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V3:.*]] = "mhlo.iota"() <{iota_dimension = 2 : i64}> : () -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V4:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[V5:.*]] = "mhlo.broadcast"(%[[V4]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V6:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[V7:.*]] = "mhlo.broadcast"(%[[V6]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V8:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK-DAG: %[[V9:.*]] = "mhlo.broadcast"(%[[V8]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V10:.*]] = mhlo.constant dense<11> : tensor<i32>
  // CHECK-DAG: %[[V11:.*]] = "mhlo.broadcast"(%[[V10]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V12:.*]] = mhlo.constant dense<140> : tensor<i32>
  // CHECK-DAG: %[[V13:.*]] = "mhlo.broadcast"(%[[V12]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V14:.*]] = mhlo.constant dense<128> : tensor<i32>
  // CHECK-DAG: %[[V15:.*]] = "mhlo.broadcast"(%[[V14]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V16:.*]] = mhlo.constant dense<128> : tensor<i32>
  // CHECK-DAG: %[[V17:.*]] = "mhlo.broadcast"(%[[V16]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V18:.*]] = mhlo.subtract %[[V11]], %[[V2]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V19:.*]] = mhlo.negate %[[V18]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V20:.*]] = mhlo.minimum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V21:.*]] = mhlo.add %[[V13]], %[[V20]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V22:.*]] = mhlo.maximum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V23:.*]] = mhlo.subtract %[[V15]], %[[V22]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V24:.*]] = mhlo.minimum %[[V21]], %[[V23]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V25:.*]] = chlo.broadcast_compare %[[V18]], %[[V5]] {comparison_direction = #chlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V26:.*]] = mhlo.subtract %[[V17]], %[[V24]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V27:.*]] = mhlo.select %[[V25]], %[[V26]], %[[V5]] : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  // CHECK-DAG: %[[V28:.*]] = mhlo.maximum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V29:.*]] = mhlo.subtract %[[V28]], %[[V27]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V30:.*]] = mhlo.maximum %[[V19]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V31:.*]] = mhlo.subtract %[[V30]], %[[V27]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V32:.*]] = mhlo.add %[[V3]], %[[V29]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V33:.*]] = mhlo.add %[[V3]], %[[V31]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V34:.*]] = chlo.broadcast_compare %[[V32]], %[[V5]] {comparison_direction = #chlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V35:.*]] = chlo.broadcast_compare %[[V32]], %[[V15]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V36:.*]] = mhlo.and %[[V34]], %[[V35]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V37:.*]] = chlo.broadcast_compare %[[V33]], %[[V5]] {comparison_direction = #chlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V38:.*]] = chlo.broadcast_compare %[[V33]], %[[V13]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V39:.*]] = mhlo.and %[[V37]], %[[V38]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V40:.*]] = mhlo.and %[[V36]], %[[V39]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V41:.*]] = mhlo.reshape %[[V40]] : (tensor<1x22x128xi1>) -> tensor<22x128xi1>
  // CHECK-DAG: %[[V42:.*]] = "mhlo.concatenate"(%[[V33]], %[[V32]]) <{dimension = 0 : i64}> : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<2x22x128xi32>
  // CHECK-DAG: %[[V43:.*]] = "mhlo.gather"(%[[ARG]], %[[V42]]) <{dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [1, 2]>, indices_are_sorted = false, slice_sizes = dense<[7, 1, 1]> : tensor<3xi64>}> : (tensor<7x140x128xi32>, tensor<2x22x128xi32>) -> tensor<7x22x128xi32>
  // CHECK-DAG: %[[V44:.*]] = "mhlo.broadcast"(%[[V41]]) <{broadcast_sizes = dense<7> : tensor<1xi64>}> : (tensor<22x128xi1>) -> tensor<7x22x128xi1>
  // CHECK-DAG: %[[V45:.*]] = "mhlo.broadcast"(%[[V0]]) <{broadcast_sizes = dense<[7, 22, 128]> : tensor<3xi64>}> : (tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[V46:.*]] = mhlo.select %[[V44]], %[[V43]], %[[V45]] : tensor<7x22x128xi1>, tensor<7x22x128xi32>
  // CHECK: return %[[V46]] : tensor<7x22x128xi32>
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_zero_dim_complex
func.func @matrix_diag_part_zero_dim_complex(%arg0: tensor<4x0xcomplex<f32>>) -> tensor<0xcomplex<f32>> {
  %cst = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {value = dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>} : () -> tensor<complex<f32>>
  %0 = "tf.MatrixDiagPartV3"(%arg0, %cst, %cst_0) {align = "RIGHT_LEFT", device = ""} : (tensor<4x0xcomplex<f32>>, tensor<i32>, tensor<complex<f32>>) -> tensor<0xcomplex<f32>>
  // CHECK: return %{{[0-9]*}} : tensor<0xcomplex<f32>>
  return %0 : tensor<0xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_single_diagonal
func.func @matrix_diag_part_single_diagonal(%arg0: tensor<7x140x128xi32>) -> tensor<7x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<0> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x128xi32>
  // CHECK: %[[result:.*]] = mhlo.reshape {{.*}} : (tensor<7x1x128xi32>) -> tensor<7x128xi32>
  // CHECK: return %[[result]] : tensor<7x128xi32>
  func.return %2: tensor<7x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_ll
func.func @matrix_diag_part_align_ll(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "LEFT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[false:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK: %[[b_false:.*]] = "mhlo.broadcast"(%[[false]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = mhlo.select %[[b_false]], %{{[0-9]*}}, %{{[0-9]*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_lr
func.func @matrix_diag_part_align_lr(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "LEFT_RIGHT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[le:.*]] = chlo.broadcast_compare %{{[0-9]*}}, %{{[0-9]*}} {comparison_direction = #chlo<comparison_direction LE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} =  mhlo.select %[[le]], %{{[0-9]*}}, %{{[0-9]*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_rl
func.func @matrix_diag_part_align_rl(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[ge:.*]] = chlo.broadcast_compare %{{[0-9]*}}, %{{[0-9]*}} {comparison_direction = #chlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = mhlo.select %[[ge]], %{{[0-9]*}}, %{{[0-9]*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_rr
func.func @matrix_diag_part_align_rr(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_RIGHT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[true:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK: %[[b_true:.*]] = "mhlo.broadcast"(%[[true]]) <{broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>}> : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = mhlo.select %[[b_true]], %{{[0-9]*}}, %{{[0-9]*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_7d
// CHECK: (%arg0: tensor<3x5x7x9x11x13x17xf32>) -> tensor<3x5x7x9x11x4x10xf32>
func.func @matrix_diag_part_align_7d(%arg0: tensor<3x5x7x9x11x13x17xf32>) -> tensor<3x5x7x9x11x4x10xf32> {
  %0 = mhlo.constant dense<-1.> : tensor<f32>  // padding value
  %1 = mhlo.constant dense<[-6, -3]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = f32, align = "LEFT_RIGHT"
  } : (tensor<3x5x7x9x11x13x17xf32>, tensor<2xi32>, tensor<f32>) -> tensor<3x5x7x9x11x4x10xf32>
  func.return %2: tensor<3x5x7x9x11x4x10xf32>
}

//===----------------------------------------------------------------------===//
// Erf
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @erf
func.func @erf(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: mhlo.erf %arg0 : tensor<2x3xf32>
  %0 = "tf.Erf"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Erfc
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @erfc
func.func @erfc(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: chlo.erfc %arg0 : tensor<2x3xf32>
  %0 = "tf.Erfc"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Einsum.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @einsum
func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK:  mhlo.einsum
  %0 = "tf.Einsum"(%arg0, %arg1) {equation = "ab,bc->ac"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return %0: tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @unary_einsum
func.func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK:  mhlo.constant{{.*}}1.000000e+00
  // CHECK:  mhlo.einsum{{.*}}",ab->aa"
  %0 = "tf.Einsum"(%arg0) {equation = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  func.return %0: tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// FloorDiv and FloorMod.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @floordiv_broadcast_i32
func.func @floordiv_broadcast_i32(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[SUB]], [[DIV]]
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floordiv_reverse_broadcast_i32
func.func @floordiv_reverse_broadcast_i32(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]]
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[SUB]], [[DIV]]
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floordiv_f32
func.func @floordiv_f32(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:  %[[DIV:.*]] = chlo.broadcast_divide %arg0, %arg0
  // CHECK-NEXT:  %[[FLOOR:.*]] = mhlo.floor %[[DIV]]
  // CHECK-NEXT:  return %[[FLOOR]] : tensor<2xf32>
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0: tensor<2xf32>
}

// -----

// CHECK-LABEL: func @floordiv_bf16
func.func @floordiv_bf16(%arg0: tensor<2xbf16>) -> tensor<2xbf16> {
  // CHECK-NEXT:  mhlo.convert
  // CHECK-NEXT:  mhlo.convert
  // CHECK-NEXT:  chlo.broadcast_divide
  // CHECK-NEXT:  mhlo.floor
  // CHECK-NEXT:  mhlo.convert
  // CHECK-NEXT:  return
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  func.return %0: tensor<2xbf16>
}

// -----

// CHECK-LABEL: func @floordiv_f16_broadcast
func.func @floordiv_f16_broadcast(%arg0: tensor<2x3xf16>, %arg1: tensor<3xf16>) -> tensor<2x3xf16> {
  // CHECK-NEXT:  chlo.broadcast_divide
  // CHECK-NEXT:  mhlo.floor
  // CHECK-NEXT:  return
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
  func.return %0: tensor<2x3xf16>
}

// -----

// CHECK-LABEL: func @floordiv_dynamic
func.func @floordiv_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[SUB]], [[DIV]]
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floordiv_unsigned
func.func @floordiv_unsigned(%arg0: tensor<?x?xui32>, %arg1: tensor<?xui32>) -> tensor<?x?xui32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK: return [[DIV]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xui32>, tensor<?xui32>) -> tensor<?x?xui32>
  func.return %0: tensor<?x?xui32>
}

// -----

// CHECK-LABEL: func @floordiv_int
func.func @floordiv_int(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #chlo<comparison_direction NE>} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[SUB]], [[DIV]]
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_numerator
func.func @floormod_broadcast_numerator(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = array<i64>, comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]]
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[ADD]], [[REM]]
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_denominator
func.func @floormod_broadcast_denominator(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = array<i64>, comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[ADD]], [[REM]]
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_unsigned_broadcast_denominator
func.func @floormod_unsigned_broadcast_denominator(%arg0: tensor<2x3xui32>, %arg1: tensor<3xui32>) -> tensor<2x3xui32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-NEXT: return [[REM]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xui32>, tensor<3xui32>) -> tensor<2x3xui32>
  func.return %0: tensor<2x3xui32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_numerator
func.func @floormod_dynamic_broadcast_numerator_(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = array<i64>, comparison_direction = #chlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = array<i64: 1>, comparison_direction = #chlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] {broadcast_dimensions = array<i64: 1>}
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[ADD]], [[REM]]
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_denominator
func.func @floormod_dynamic_broadcast_denominator_(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: tf.FloorMod
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #chlo<comparison_direction NE>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #chlo<comparison_direction LT>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = array<i64>, comparison_direction = #chlo<comparison_direction LT>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #chlo<comparison_direction NE>} : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]] : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-DAG: [[SELECT:%.+]] = mhlo.select [[AND]], [[ADD]], [[REM]] : tensor<?x?x?xi1>, tensor<?x?x?xf32>
  // CHECK-NEXT: return [[SELECT]] : tensor<?x?x?xf32>
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0: tensor<?x?x?xf32>
}

//===----------------------------------------------------------------------===//
// OnesLike
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @ones_like
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x?xf32>)
func.func @ones_like(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[RES:.*]] = "chlo.constant_like"(%[[ARG]]) <{value = 1.0{{.*}}}>
  // CHECK: return %[[RES]]
  %0 = "tf.OnesLike"(%arg0) : (tensor<2x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}

//===----------------------------------------------------------------------===//
// ZerosLike
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @zeros_like
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x?xf32>)
func.func @zeros_like(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[RES:.*]] = "chlo.constant_like"(%[[ARG]]) <{value = 0.0{{.*}}}>
  // CHECK: return %[[RES]]
  %0 = "tf.ZerosLike"(%arg0) : (tensor<2x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}

//===----------------------------------------------------------------------===//
// BroadcastTo.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @broadcast_to
func.func @broadcast_to(%arg0: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %cst = "tf.Const"() { value = dense<16> : tensor<4xi32> } : () -> tensor<4xi32>

  // CHECK: [[CST:%.+]] = mhlo.constant
  // CHECK: "mhlo.dynamic_broadcast_in_dim"(%arg0, [[CST]])
  // CHECK-SAME: {broadcast_dimensions = dense<3> : tensor<1xi64>}
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}

//===----------------------------------------------------------------------===//
// Complex op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @complex
func.func @complex(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xcomplex<f32>> {
  // CHECK: chlo.broadcast_complex
  %1 = "tf.Complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @imag
func.func @imag(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: mhlo.imag
  %1 = "tf.Imag"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @real
func.func @real(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: mhlo.real
  %1 = "tf.Real"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

//===----------------------------------------------------------------------===//
// Concat op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @concat_v2
func.func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "mhlo.concatenate"({{.*}}) <{dimension = 0 : i64}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_neg_axis
func.func @concat_v2_neg_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "mhlo.concatenate"({{.*}}) <{dimension = 0 : i64}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>

  %axis = "tf.Const"() { value = dense<-2> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_1d_axis
func.func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  // CHECK: "mhlo.concatenate"({{.*}}) <{dimension = 1 : i64}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x6xf32>

  %axis = "tf.Const"() { value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi64>) -> tensor<3x6xf32>
  func.return %1 : tensor<3x6xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_non_const_axis
func.func @concat_v2_non_const_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %axis: tensor<i64>) -> tensor<3x6xf32> {
  // CHECK: "tf.ConcatV2"
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<3x6xf32>
  func.return %1 : tensor<3x6xf32>
}

//===----------------------------------------------------------------------===//
// Pad op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @padv2_1D
func.func @padv2_1D(%arg0: tensor<3xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK: "mhlo.pad"(%arg0, %arg1) <{
  // CHECK-SAME: edge_padding_high = dense<2> : tensor<1xi64>,
  // CHECK-SAME: edge_padding_low = dense<1> : tensor<1xi64>,
  // CHECK-SAME: interior_padding = dense<0> : tensor<1xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<6xf32>
  func.return %1 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func @padv2_2D
func.func @padv2_2D(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi64> } : () -> tensor<2x2xi64>
  // CHECK: "mhlo.pad"(%arg0, %arg1) <{
  // CHECK-SAME:    edge_padding_high = dense<[2, 4]> : tensor<2xi64>,
  // CHECK-SAME:    edge_padding_low = dense<[1, 3]> : tensor<2xi64>,
  // CHECK-SAME:    interior_padding = dense<0> : tensor<2xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<6x9xf32>
  func.return %1 : tensor<6x9xf32>
}

// -----

// CHECK-LABEL: func @padv2_i32_paddings
func.func @padv2_i32_paddings(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi32> } : () -> tensor<2x2xi32>
  // CHECK: "mhlo.pad"(%arg0, %arg1) <{
  // CHECK-SAME:    edge_padding_high = dense<[2, 4]> : tensor<2xi64>,
  // CHECK-SAME:    edge_padding_low = dense<[1, 3]> : tensor<2xi64>,
  // CHECK-SAME:    interior_padding = dense<0> : tensor<2xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<6x9xf32>
  func.return %1 : tensor<6x9xf32>
}

// -----

// CHECK-LABEL: func @padv2_dynamic
func.func @padv2_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1x2xi64>) -> tensor<?xf32> {
  // CHECK: "mhlo.transpose"({{.*}}) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<1x2xi64>) -> tensor<2x1xi64>
  // CHECK: mhlo.reshape {{.*}} : (tensor<2x1xi64>) -> tensor<2xi64>
  // CHECK: "mhlo.slice"({{.*}}) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: "mhlo.slice"({{.*}}) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: mhlo.dynamic_pad {{.*}} : (tensor<?xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  %1 = "tf.PadV2"(%arg0, %arg2, %arg1) : (tensor<?xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// Identity op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @identity
func.func @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.Identity"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @identityN
func.func @identityN(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
  // CHECK-NEXT:  return %arg0, %arg1 : tensor<1xi32>, tensor<1xf32>
  %0:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>)
  func.return %0#0, %0#1: tensor<1xi32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: func @stopgradient
func.func @stopgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.StopGradient"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @preventgradient
func.func @preventgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.PreventGradient"(%arg0) {message = "fin gradients"} : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @checkNumerics
func.func @checkNumerics(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xf32>
  %0 = "tf.CheckNumerics"(%arg0) {message = "check numerics"} : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

//===----------------------------------------------------------------------===//
// InfeedDequeueTuple legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @infeed_dequeue_tuple
func.func @infeed_dequeue_tuple() -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>) {
// CHECK: [[TOKEN:%.*]] = mhlo.create_token  : !mhlo.token
// CHECK: [[INFEED:%.*]]:3 = "mhlo.infeed"([[TOKEN]]) <{infeed_config = ""{{.*}}}> : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)
// CHECK: return [[INFEED]]#0, [[INFEED]]#1
  %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>)
  func.return %0#0, %0#1 : tensor<1x8x4x4xi32>, tensor<1x100x1xf32>
}

// -----

// CHECK-LABEL: func @infeed_dequeue_tuple_dynamic_error
func.func @infeed_dequeue_tuple_dynamic_error() -> (tensor<3x3xf32>, tensor<4x?xf32>) {
  // We expect legalization to fail for dynamic shapes:
  // CHECK: [[INFEED:%.*]] = "tf.InfeedDequeueTuple"{{.*}}
  %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3x3xf32>, tensor<4x?xf32>)
  func.return %0#0, %0#1 : tensor<3x3xf32>, tensor<4x?xf32>
}

// The following op sharding is used:
// Proto debug string:
//   type: TUPLE
//   tuple_shardings {
//     type: MAXIMAL
//     tile_assignment_dimensions: 1
//     tile_assignment_devices: 0
//   }
// Serialized string:
//   "\08\02*\08\08\01\1A\01\01\22\01\00"

// CHECK-LABEL: infeed_dequeue_tuple_sharding
func.func @infeed_dequeue_tuple_sharding() -> tensor<8xi32> {
  // CHECK: "mhlo.infeed"
  // An additional sharding is added at the end to account for token result.
  // Proto debug string:
  //   type: TUPLE
  //   tuple_shardings {
  //     type: MAXIMAL
  //     tile_assignment_dimensions: 1
  //     tile_assignment_devices: 0
  //   }
  //   tuple_shardings {
  //     type: MAXIMAL
  //     tile_assignment_dimensions: 1
  //     tile_assignment_devices: 0
  //   }
  // CHECK-SAME: mhlo.sharding = "\08\02*\08\08\01\1A\01\01\22\01\00*\08\08\01\1A\01\01\22\01\00"
  %0 = "tf.InfeedDequeueTuple"() {_XlaSharding = "\08\02*\08\08\01\1A\01\01\22\01\00"} : () -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

//===----------------------------------------------------------------------===//
// Nullary op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @const
func.func @const() -> tensor<2xi32> {
  // CHECK: mhlo.constant dense<0> : tensor<2xi32>
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
  func.return %0: tensor<2xi32>
}

// -----

// CHECK-LABEL: @const_dynamic_output
func.func @const_dynamic_output() -> tensor<*xi32> {
  // CHECK: [[CONST:%.*]] = mhlo.constant dense<0> : tensor<2xi32>
  // CHECK: [[CAST:%.*]] = tensor.cast [[CONST]] : tensor<2xi32> to tensor<*xi32>
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> (tensor<*xi32>)
  // CHECK: return [[CAST]]
  func.return %0: tensor<*xi32>
}

// -----

// CHECK-LABEL: @opaque_const
func.func @opaque_const() -> tensor<!tf_type.variant<tensor<2xi32>>> {
  // CHECK-NOT: mhlo.constant
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<!tf_type.variant>} : () -> tensor<!tf_type.variant<tensor<2xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<2xi32>>>
}

//===----------------------------------------------------------------------===//
// Matmul op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: matmul_notranspose
// CHECK-SAME: (%[[A:.*]]: tensor<5x7xf32>, %[[B:.*]]: tensor<7x11xf32>)
func.func @matmul_notranspose(%a: tensor<5x7xf32>, %b: tensor<7x11xf32>) -> tensor<5x11xf32> {
  // CHECK: "mhlo.dot"(%[[A]], %[[B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<5x7xf32>, tensor<7x11xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// -----

// CHECK-LABEL: matmul_transpose_b
// CHECK-SAME: (%[[A:.*]]: tensor<5x7xf32>, %[[B:.*]]: tensor<11x7xf32>)
func.func @matmul_transpose_b(%a: tensor<5x7xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_B:.*]] = "mhlo.transpose"(%[[B]]) <{permutation = dense<[1, 0]> : tensor<2xi64>}>
  // CHECK: "mhlo.dot"(%[[A]], %[[UPDATED_B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = true} : (tensor<5x7xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// -----

// CHECK-LABEL: matmul_transpose_both
// CHECK-SAME: (%[[A:.*]]: tensor<7x5xf32>, %[[B:.*]]: tensor<11x7xf32>)
func.func @matmul_transpose_both(%a: tensor<7x5xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_A:.*]] = "mhlo.transpose"(%[[A]]) <{permutation = dense<[1, 0]> : tensor<2xi64>}>
  // CHECK: %[[UPDATED_B:.*]] = "mhlo.transpose"(%[[B]]) <{permutation = dense<[1, 0]> : tensor<2xi64>}>
  // CHECK: "mhlo.dot"(%[[UPDATED_A]], %[[UPDATED_B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = true, transpose_b = true} : (tensor<7x5xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// Verify that MatMul with ranked inputs are lowered to HLO.
// CHECK-LABEL: matmul_ranked
func.func @matmul_ranked(%a: tensor<?x7xf32>, %b: tensor<7x?xf32>) -> tensor<?x?xf32> {
  // CHECK: "mhlo.dot"
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<?x7xf32>, tensor<7x?xf32>) -> tensor<?x?xf32>

  func.return %0 : tensor<?x?xf32>
}

// Verify SparseMatMul is legalized to dot.
// CHECK-LABEL: test_sparse_mat_mul
func.func @test_sparse_mat_mul(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  // CHECK: "mhlo.dot"
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = false} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

// SparseMatMul where one operand needs to be transposed and the other one not.
//
// CHECK-LABEL:   @test_sparse_mat_mul_with_transpose
// CHECK-SAME:      %[[ARG0:.*]]: tensor<3x4xf32>
// CHECK-SAME:      %[[ARG1:.*]]: tensor<5x4xf32>
// CHECK-SAME:      -> tensor<3x5xf32>
// CHECK:           %[[TRANSPOSE:.*]] = "mhlo.transpose"(%[[ARG1]])
// CHECK-SAME:        permutation = dense<[1, 0]>
// CHECK-SAME:        -> tensor<4x5xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.dot"(%[[ARG0]], %[[TRANSPOSE]])
// CHECK-SAME:        -> tensor<3x5xf32>
// CHECK:           return %[[RESULT]]
func.func @test_sparse_mat_mul_with_transpose(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<3x5xf32> {
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = true} : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

// SparseMatMul where one operand needs to be casted and the other one not.
//
// CHECK-LABEL:   @test_sparse_mat_mul_with_cast
// CHECK-SAME:      %[[ARG0:.*]]: tensor<3x4xf32>
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4x5xbf16>
// CHECK-SAME:      -> tensor<3x5xf32>
// CHECK:           %[[CAST:.*]] = mhlo.convert %[[ARG1]]
// CHECK-SAME:        -> tensor<4x5xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.dot"(%[[ARG0]], %[[CAST]])
// CHECK-SAME:        -> tensor<3x5xf32>
// CHECK:           return %[[RESULT]]
func.func @test_sparse_mat_mul_with_cast(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xbf16>) -> tensor<3x5xf32> {
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = false} : (tensor<3x4xf32>, tensor<4x5xbf16>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

//===----------------------------------------------------------------------===//
// MaxPool op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: maxpool_valid_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func.func @maxpool_valid_padding(%arg0: tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<-2147483648> : tensor<i32>
  // CHECK: "mhlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK: <{window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>}>
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 4, 4, 1]} : (tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32>
  func.return %0 : tensor<2x3x5x7xi32>
}

// -----

// CHECK-LABEL: maxpool_same_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func.func @maxpool_same_padding(%arg0: tensor<2x13x25x7xi32>) -> tensor<2x4x7x7xi32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 4, 1]} : (tensor<2x13x25x7xi32>) -> tensor<2x4x7x7xi32>
  func.return %0 : tensor<2x4x7x7xi32>
}

// -----

// CHECK-LABEL: maxpool_3d_valid_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func.func @maxpool_3d_valid_padding(%arg0: tensor<2x8x12x20x7xf32>) -> tensor<2x8x3x5x7xf32> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: "mhlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK: <{window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 4, 4, 1]> : tensor<5xi64>}>
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return

  %0 = "tf.MaxPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 2, 2, 1], padding = "VALID", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x12x20x7xf32>) -> tensor<2x8x3x5x7xf32>
  func.return %0 : tensor<2x8x3x5x7xf32>
}

// -----

// CHECK-LABEL: maxpool_3d_same_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func.func @maxpool_3d_same_padding(%arg0: tensor<2x8x13x25x7xf32>) -> tensor<2x8x4x7x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>

  %0 = "tf.MaxPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 2, 3, 1], padding = "SAME", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x13x25x7xf32>) -> tensor<2x8x4x7x7xf32>
  func.return %0 : tensor<2x8x4x7x7xf32>
}

// -----

// CHECK-LABEL: maxpool_explicit_padding
func.func @maxpool_explicit_padding(%arg0: tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32> {
  // CHECK: tf.MaxPool
  // TODO(b/165938852): need to support explicit padding in max_pool.

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "EXPLICIT", strides = [1, 4, 4, 1]} : (tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32>
  func.return %0 : tensor<2x3x5x7xi32>
}

//===----------------------------------------------------------------------===//
// MaxPoolGrad op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @max_pool_grad_valid
// CHECK-SAME: %[[INPUT:.*]]: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %[[GRAD:.*]]: tensor<10x12x12x64xf32>
func.func @max_pool_grad_valid(%orig_input: tensor<10x24x24x64xf32>, %orig_output: tensor<10x12x12x64xf32>, %grad: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) <{window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}> ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.compare GE, %[[VALUE_A]], %[[VALUE_B]], NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<i1>
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<f32>
  // CHECK: }) : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  // CHECK: return %[[RESULT]] : tensor<10x24x24x64xf32>
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
  func.return %result : tensor<10x24x24x64xf32>
}

// -----

// CHECK-LABEL: @max_pool_3d_grad_valid
// CHECK-SAME: %[[INPUT:.*]]: tensor<10x8x24x24x64xf32>, %arg1: tensor<10x8x12x12x64xf32>, %[[GRAD:.*]]: tensor<10x8x12x12x64xf32>
func.func @max_pool_3d_grad_valid(%orig_input: tensor<10x8x24x24x64xf32>, %orig_output: tensor<10x8x12x12x64xf32>, %grad: tensor<10x8x12x12x64xf32>) -> tensor<10x8x24x24x64xf32> {
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "mhlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) <{window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>}> ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.compare GE, %[[VALUE_A]], %[[VALUE_B]], NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<i1>
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<f32>
  // CHECK: }) : (tensor<10x8x24x24x64xf32>, tensor<10x8x12x12x64xf32>, tensor<f32>) -> tensor<10x8x24x24x64xf32>
  // CHECK: return %[[RESULT]] : tensor<10x8x24x24x64xf32>
  %result = "tf.MaxPool3DGrad"(%orig_input, %orig_output, %grad) {data_format = "NDHWC", ksize = [1, 1, 2, 2, 1], padding = "VALID", strides = [1, 1, 2, 2, 1]} : (tensor<10x8x24x24x64xf32>, tensor<10x8x12x12x64xf32>, tensor<10x8x12x12x64xf32>) -> tensor<10x8x24x24x64xf32>
  func.return %result : tensor<10x8x24x24x64xf32>
}

// -----

// CHECK-LABEL: @max_pool_grad_same
func.func @max_pool_grad_same(%orig_input: tensor<2x13x25x7xf32>, %orig_output: tensor<2x4x7x7xf32>, %grad: tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 3, 1],
     padding = "SAME",
     strides = [1, 4, 4, 1]
  } : (tensor<2x13x25x7xf32>, tensor<2x4x7x7xf32>, tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32>
  func.return %result : tensor<2x13x25x7xf32>
}

// -----

// CHECK-LABEL: @max_pool_3d_grad_same
func.func @max_pool_3d_grad_same(%orig_input: tensor<2x8x13x25x7xf32>, %orig_output: tensor<2x8x4x7x7xf32>, %grad: tensor<2x8x4x7x7xf32>) -> tensor<2x8x13x25x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>
  %result = "tf.MaxPool3DGrad"(%orig_input, %orig_output, %grad) {data_format = "NDHWC", ksize = [1, 1, 2, 3, 1], padding = "SAME", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x13x25x7xf32>, tensor<2x8x4x7x7xf32>, tensor<2x8x4x7x7xf32>) -> tensor<2x8x13x25x7xf32>
  func.return %result : tensor<2x8x13x25x7xf32>
}

//===----------------------------------------------------------------------===//
// OneHot op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:one_hot
func.func @one_hot(%indices: tensor<3xi32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  // CHECK: %[[IOTA:.*]] = "mhlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<3x5xi32>
  // CHECK: %[[BCAST_ARG0:.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<3xi32>) -> tensor<3x5xi32>
  // CHECK: %[[COMPARE:.*]] = mhlo.compare EQ, %[[BCAST_ARG0]], %[[IOTA]], NOTYPE : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<3x5xi1>
  // CHECK: %[[ON_VALUE:.*]] = "mhlo.broadcast"(%arg1) <{broadcast_sizes = dense<[3, 5]> : tensor<2xi64>}> : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[OFF_VALUE:.*]] = "mhlo.broadcast"(%arg2) <{broadcast_sizes = dense<[3, 5]> : tensor<2xi64>}> : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[COMPARE]], %[[ON_VALUE]], %[[OFF_VALUE]] : tensor<3x5xi1>, tensor<3x5xf32>
  // CHECK: return %[[RESULT]] : tensor<3x5xf32>
  %depth = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

//===----------------------------------------------------------------------===//
// tf.OutfeedEnqueueTuple legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @outfeed_enqueue_tuple
// CHECK-SAME: [[VAL_0:%.*]]: tensor<3xi32>, [[VAL_1:%.*]]: tensor<4xf32>)
func.func @outfeed_enqueue_tuple(%data_1: tensor<3xi32>, %data_2: tensor<4xf32>) -> () {
// CHECK: [[TOKEN:%.*]] = mhlo.create_token  : !mhlo.token
// CHECK: "mhlo.outfeed"([[VAL_0]], [[VAL_1]], [[TOKEN]]) <{outfeed_config = ""}> : (tensor<3xi32>, tensor<4xf32>, !mhlo.token) -> !mhlo.token
  "tf.OutfeedEnqueueTuple"(%data_1, %data_2) : (tensor<3xi32>, tensor<4xf32>) -> ()
  func.return
}

//===----------------------------------------------------------------------===//
// Pack op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @pack
func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: mhlo.reshape {{.*}} : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: mhlo.reshape {{.*}} : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "mhlo.concatenate"({{.*}}) <{dimension = 0 : i64}> : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  %0 = "tf.Pack"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
// PartitionedCall op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @partitioned_call
func.func @partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: call @pcall_func(%arg0) : (tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}


func.func @pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @partitioned_call_multi_input
func.func @partitioned_call_multi_input(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: call @pcall_multi_input(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_input} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}


func.func @pcall_multi_input(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @partitioned_call_multi_in_out
func.func @partitioned_call_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: call @pcall_multi_in_out(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}


func.func @pcall_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %arg1, %arg0 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @unhandled_partitioned_call
func.func @unhandled_partitioned_call(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<i32>, tensor<i32>) {
  // The argument types don't match the parameter types for the
  // pcall_multi_in_out function. That's fine for a PartitionedCallOp but not
  // for a standard CallOp, so this op can't be lowered.
  // CHECK: "tf.PartitionedCall"
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}


// CHECK-LABEL: func @unhandled_partitioned_call_2
func.func @unhandled_partitioned_call_2(%arg0: tensor<i32>, %arg1: tensor<*xi32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: "tf.PartitionedCall"
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<i32>, tensor<*xi32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: func @no_args_and_results
func.func @no_args_and_results() {
  // CHECK: call @callee() : () -> ()
  // CHECK: call @callee() : () -> ()
  // CHECK: call @callee() : () -> ()
  "tf.PartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @callee} : () -> ()
  "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @callee} : () -> ()
  "tf.LegacyCall"() {config = "", config_proto = "", executor_type = "", f = @callee} : () -> ()
  func.return
}

func.func @callee() {
  func.return
}

//===----------------------------------------------------------------------===//
// ReverseV2 op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @reverse_func_32
func.func @reverse_func_32(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) <{dimensions = dense<0> : tensor<1xi64>}>
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi32>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_64
func.func @reverse_func_64(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> (tensor<1xi64>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) <{dimensions = dense<0> : tensor<1xi64>}>
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi64>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_neg
func.func @reverse_func_neg(%arg0: tensor<5x5xi32>) -> tensor<5x5xi32> {
  %axis = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) <{dimensions = dense<1> : tensor<1xi64>}>
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5x5xi32>, tensor<1xi32>) -> tensor<5x5xi32>

  // CHECK: return [[VAL]] : tensor<5x5xi32>
  func.return %reversed : tensor<5x5xi32>
}

//===----------------------------------------------------------------------===//
// StatefulPartitionedCall op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @stateful_partitioned_call
// CHECK-SAME: [[ARG:%.+]]: tensor<i32>
func.func @stateful_partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: call @stateful_pcall_func([[ARG]]) : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @stateful_partitioned_call_multi_in_out
// CHECK-SAME: ([[ARG0:%.+]]: tensor<i32>, [[ARG1:%.+]]: tensor<i32>)
func.func @stateful_partitioned_call_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: call @stateful_pcall_multi_in_out([[ARG0]], [[ARG1]]) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %0, %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_multi_in_out} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

func.func @stateful_pcall_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %arg1, %arg0 : tensor<i32>, tensor<i32>
}

//===----------------------------------------------------------------------===//
// Elu op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @elu
func.func @elu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) <{value = 0.000000e+00 : f32}> : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-DAG: %[[PRED:.*]] = mhlo.compare GT, %arg0, %[[ZERO]]
  // CHECK-DAG: %[[EXP:.*]] = mhlo.exponential_minus_one %arg0
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[PRED]], %arg0, %[[EXP]]
  // CHECK: return %[[RESULT]]
  %0 = "tf.Elu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

// -----

// CHECK-LABEL: func @elu_grad
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<4x8xf32>, %[[FEATURES:.*]]: tensor<?x?xf32>)
func.func @elu_grad(%gradients: tensor<4x8xf32>, %features: tensor<?x?xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[PRED:.*]] = chlo.broadcast_compare %[[FEATURES]], %[[ZERO]] {broadcast_dimensions = array<i64>, comparison_direction = #chlo<comparison_direction GT>}
  // CHECK-DAG: %[[ADD1:.*]] = chlo.broadcast_add %[[FEATURES]], %[[ONE]] {broadcast_dimensions = array<i64>}
  // CHECK-DAG: %[[MULGRAD:.*]] = mhlo.multiply %[[GRADIENTS]], %[[ADD1]] : (tensor<4x8xf32>, tensor<?x?xf32>) -> tensor<4x8xf32>
  // CHECK: %[[RESULT:.*]] = mhlo.select %[[PRED]], %[[GRADIENTS]], %[[MULGRAD]]
  // CHECK: return %[[RESULT]]
  %2 = "tf.EluGrad"(%gradients, %features) : (tensor<4x8xf32>, tensor<?x?xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Relu op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @relu
func.func @relu(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = array<i64>} : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu_unsigned
func.func @relu_unsigned(%arg0: tensor<?xui32>) -> tensor<?xui32> {
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<ui32>
  // CHECK: chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = array<i64>} : (tensor<ui32>, tensor<?xui32>) -> tensor<?xui32>
  %0 = "tf.Relu"(%arg0) : (tensor<?xui32>) -> tensor<?xui32>
  func.return %0: tensor<?xui32>
}

// -----

// CHECK-LABEL: func @relu6
func.func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[SIX:.*]] = mhlo.constant dense<6> : tensor<i32>
  // CHECK: mhlo.clamp %[[ZERO]], %arg0, %[[SIX]] : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu6_unsigned
func.func @relu6_unsigned(%arg0: tensor<?xui32>) -> tensor<?xui32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<ui32>
  // CHECK-DAG: %[[SIX:.*]] = mhlo.constant dense<6> : tensor<ui32>
  // CHECK: mhlo.clamp %[[ZERO]], %arg0, %[[SIX]] : (tensor<ui32>, tensor<?xui32>, tensor<ui32>) -> tensor<?xui32>
  %0 = "tf.Relu6"(%arg0) : (tensor<?xui32>) -> tensor<?xui32>
  func.return %0: tensor<?xui32>
}

// -----

// CHECK-LABEL: func @leaky_relu
func.func @leaky_relu(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg0) <{value = 2.000000e-01 : f32}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) <{value = 0.000000e+00 : f32}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[LEAKY:.*]] = mhlo.multiply %[[INP:.*]], %[[ALPHA]] : tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[CMP:.*]] = mhlo.compare GT, %[[INP]], %[[ZERO]], NOTYPE : (tensor<1x4x4x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xi1>
    // CHECK-NEXT: %[[RES:.*]] = mhlo.select %[[CMP]], %[[INP]], %[[LEAKY]] : tensor<1x4x4x3xi1>, tensor<1x4x4x3xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4x3xf32>
    %0 = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    func.return %0 : tensor<1x4x4x3xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu_grad
func.func @leaky_relu_grad(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg1) <{value = 2.000000e-01 : f32}> : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg1) <{value = 0.000000e+00 : f32}> : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: %[[LEAKYGRAD:.*]] = mhlo.multiply %[[GRADIENT:.*]], %[[ALPHA]] : tensor<1x4x4xf32>
    // CHECK-NEXT: %[[CMP:.*]] = mhlo.compare GT, %[[INP:.*]], %[[ZERO]], NOTYPE : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
    // CHECK-NEXT: %[[RES:.*]] = mhlo.select %[[CMP]], %[[GRADIENT]], %[[LEAKYGRAD]] : tensor<1x4x4xi1>, tensor<1x4x4xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4xf32>
    %0 = "tf.LeakyReluGrad"(%arg0, %arg1) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    func.return %0 : tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: func @softsign
func.func @softsign(%arg0: tensor<4x10xf32>) -> tensor<4x10xf32> {
    // CHECK-NEXT: %[[ONE:.*]] = "chlo.constant_like"(%arg0) <{value = 1.000000e+00 : f32}> : (tensor<4x10xf32>) -> tensor<4x10xf32>
    // CHECK-NEXT: %[[ABS:.*]] = mhlo.abs %{{.*}} : tensor<4x10xf32>
    // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %[[ONE]], %[[ABS]] : tensor<4x10xf32>
    // CHECK-NEXT: %[[DIV:.*]] = mhlo.divide %{{.*}}, %[[ADD]] : tensor<4x10xf32>
    // CHECK-NEXT: return %[[DIV]] : tensor<4x10xf32>
    %0 = "tf.Softsign"(%arg0) : (tensor<4x10xf32>) -> tensor<4x10xf32>
    func.return %0 : tensor<4x10xf32>
}

// -----

// CHECK-LABEL: func @softsign_grad
func.func @softsign_grad(%arg0: tensor<4x10xf32>, %arg1: tensor<4x10xf32>) -> tensor<4x10xf32> {

    // CHECK-NEXT: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
    // CHECK-NEXT: %[[ABS:.*]] = mhlo.abs %{{.*}} : tensor<4x10xf32>
    // CHECK-NEXT: %[[BROADCAST_ADD:.*]] = chlo.broadcast_add %[[ONE]], %[[ABS]] {broadcast_dimensions = array<i64>} : (tensor<f32>, tensor<4x10xf32>) -> tensor<4x10xf32>
    // CHECK-NEXT: %[[MUL:.*]] = mhlo.multiply %[[BROADCAST_ADD]], %[[BROADCAST_ADD]] : tensor<4x10xf32>
    // CHECK-NEXT: %[[BROADCAST_DIV:.*]] = chlo.broadcast_divide %{{.*}}, %[[MUL]] : (tensor<4x10xf32>, tensor<4x10xf32>) -> tensor<4x10xf32>
    // CHECK-NEXT: return %[[BROADCAST_DIV]] : tensor<4x10xf32>
    %0 = "tf.SoftsignGrad"(%arg0, %arg1) : (tensor<4x10xf32>, tensor<4x10xf32>) -> tensor<4x10xf32>
    func.return %0 : tensor<4x10xf32>
}

//===----------------------------------------------------------------------===//
// Roll op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @Roll_0D
func.func @Roll_0D(%arg0: tensor<512xi32>, %shift: tensor<i32>) -> tensor<512xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  //      CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  //      CHECK-DAG: %[[AXIS_SIZE:.*]] = mhlo.constant dense<512> : tensor<i32>
  //      CHECK: %[[T1:.+]] = mhlo.remainder %arg1, %[[AXIS_SIZE]] : tensor<i32>
  //      CHECK: %[[T2:.+]] = mhlo.add %[[T1]], %[[AXIS_SIZE]] : tensor<i32>
  //      CHECK: %[[T3:.+]] = mhlo.remainder %[[T2]], %[[AXIS_SIZE]] : tensor<i32>
  //      CHECK: %[[CONCAT:.+]] = "mhlo.concatenate"(%arg0, %arg0) <{dimension = 0 : i64}>
  //      CHECK: %[[OFFSET:.+]] = mhlo.subtract %[[AXIS_SIZE]], %[[T3]] : tensor<i32>
  //      CHECK: "mhlo.dynamic_slice"(%[[CONCAT]], %[[OFFSET]])
  // CHECK-SAME:    {slice_sizes = dense<512> : tensor<1xi64>}
  // CHECK-SAME:    (tensor<1024xi32>, tensor<i32>) -> tensor<512xi32>
  %0 = "tf.Roll"(%arg0, %shift, %axis) {device = ""} : (tensor<512xi32>, tensor<i32>, tensor<i32>) -> tensor<512xi32>
  func.return %0 : tensor<512xi32>
}

//===----------------------------------------------------------------------===//
// Select op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @select_batch_static
func.func @select_batch_static(%arg0: tensor<2xi1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: %[[BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %{{.*}}) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<2xi1>, tensor<3xindex>) -> tensor<2x6x8xi1>
  // CHECK: mhlo.select %[[BCAST]], %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_r1
func.func @select_batch_static_r1(%arg0: tensor<i1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: mhlo.select %arg0, %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_all_same
func.func @select_batch_static_all_same(%arg0: tensor<2x6x8xi1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: mhlo.select %arg0, %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2x6x8xi1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_dynamic_r1
func.func @select_batch_dynamic_r1(%arg0: tensor<?xi1>, %arg1: tensor<?x?x8xi32>, %arg2: tensor<?x?x8xi32>) -> tensor<?x?x8xi32> {
  // CHECK-NEXT: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  // CHECK-NEXT: %[[SHAPE1:.*]] = shape.shape_of %arg1 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPE2:.*]] = shape.shape_of %arg2 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ1:.*]] = shape.cstr_eq %[[SHAPE1]], %[[SHAPE2]] : tensor<3xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[HEAD:.*]], %[[TAIL:.*]] = "shape.split_at"(%[[SHAPE1]], %[[C1]]) : (tensor<3xindex>, index) -> (tensor<1xindex>, tensor<2xindex>)
  // CHECK-NEXT: %[[SHAPEEQ2:.*]] = shape.cstr_eq %[[SHAPE0]], %[[HEAD]] : tensor<1xindex>, tensor<1xindex>
  // CHECK-NEXT: %[[SHAPEEQ:.*]] = shape.assuming_all %[[SHAPEEQ1]], %[[SHAPEEQ2]]
  // CHECK-NEXT: %[[ASSUMING:.*]] = shape.assuming %[[SHAPEEQ]] -> (tensor<?x?x8xi32>) {
  // CHECK-NEXT: %[[SHAPE1E:.*]] = shape.to_extent_tensor %[[SHAPE1]] : tensor<3xindex> -> tensor<3xindex>
  // CHECK-NEXT: %[[BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[SHAPE1E]]) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<?xi1>, tensor<3xindex>) -> tensor<?x?x8xi1>
  // CHECK-NEXT: %[[SELECT:.*]] = mhlo.select %[[BCAST]], %arg1, %arg2 : tensor<?x?x8xi1>, tensor<?x?x8xi32>
  // CHECK-NEXT: shape.assuming_yield %[[SELECT]] : tensor<?x?x8xi32>
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<?xi1>, tensor<?x?x8xi32>, tensor<?x?x8xi32>) -> tensor<?x?x8xi32>
  func.return %0: tensor<?x?x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_dynamic
func.func @select_batch_dynamic(%arg0: tensor<?x?x8xi1>, %arg1: tensor<?x?x8xi32>, %arg2: tensor<?x?x8xi32>) -> tensor<?x?x8xi32> {
  // CHECK-NEXT: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?x?x8xi1> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPE1:.*]] = shape.shape_of %arg1 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPE2:.*]] = shape.shape_of %arg2 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ1:.*]] = shape.cstr_eq %[[SHAPE1]], %[[SHAPE2]] : tensor<3xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ2:.*]] = shape.cstr_eq %[[SHAPE0]], %[[SHAPE1]] : tensor<3xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ:.*]] = shape.assuming_all %[[SHAPEEQ1]], %[[SHAPEEQ2]]
  // CHECK-NEXT: %[[ASSUMING:.*]] = shape.assuming %[[SHAPEEQ]] -> (tensor<?x?x8xi32>) {
  // CHECK-NEXT: %[[SELECT:.*]] = mhlo.select %arg0, %arg1, %arg2 : tensor<?x?x8xi1>, tensor<?x?x8xi32>
  // CHECK-NEXT: shape.assuming_yield %[[SELECT]] : tensor<?x?x8xi32>
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<?x?x8xi1>, tensor<?x?x8xi32>, tensor<?x?x8xi32>) -> tensor<?x?x8xi32>
  func.return %0: tensor<?x?x8xi32>
}

// -----

// CHECK-LABEL: testSelectInvalidUnranked
func.func @testSelectInvalidUnranked(%arg0: tensor<6x7xi1>, %arg1: tensor<*xf16>, %arg2: tensor<*xf16>) -> tensor<*xf16> {
  // CHECK-NEXT: tf.Select
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<6x7xi1>, tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
  func.return %0: tensor<*xf16>
}

// -----

// CHECK-LABEL: testSelectThenUnranked
func.func @testSelectThenUnranked(%arg0: tensor<3xi1>, %arg1: tensor<*xf16>, %arg2: tensor<3x2xf16>) -> tensor<*xf16> {
  // CHECK-NEXT: tf.Select
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<*xf16>, tensor<3x2xf16>) -> tensor<*xf16>
  func.return %0: tensor<*xf16>
}

// -----

// CHECK-LABEL: testSelectElseUnranked
func.func @testSelectElseUnranked(%arg0: tensor<3xi1>, %arg1: tensor<3x2xf16>, %arg2: tensor<*xf16>) -> tensor<*xf16> {
  // CHECK-NEXT: tf.Select
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<3x2xf16>, tensor<*xf16>) -> tensor<*xf16>
  func.return %0: tensor<*xf16>
}

// -----

// CHECK-LABEL: func @selectv2_dynamic_ranked
func.func @selectv2_dynamic_ranked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  // CHECK: chlo.broadcast_select
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  func.return %0: tensor<2x?x8xi32>
}

//===----------------------------------------------------------------------===//
// Fast Fourier Transform op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @fft_1D
func.func @fft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: "mhlo.fft"(%arg0) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT>}> : (tensor<8xcomplex<f32>>
  %0 = "tf.FFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @ifft_1D
func.func @ifft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: "mhlo.fft"(%arg0) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type IFFT>}> : (tensor<8xcomplex<f32>>
  %0 = "tf.IFFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D
func.func @rfft_1D(%arg0: tensor<8xf32>) -> tensor<5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: "mhlo.fft"(%arg0) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>}> : (tensor<8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<8xf32>, tensor<1xi32>) -> tensor<5xcomplex<f32>>
  func.return %0 : tensor<5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_padded
func.func @rfft_1D_padded(%arg0: tensor<7xf32>) -> tensor<5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[PADDED:.*]] = "mhlo.pad"(%arg0, %{{.*}}) <{edge_padding_high = dense<1> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>}> : (tensor<7xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK: "mhlo.fft"(%[[PADDED]]) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>}> : (tensor<8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<7xf32>, tensor<1xi32>) -> tensor<5xcomplex<f32>>
  func.return %0 : tensor<5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_sliced
func.func @rfft_1D_sliced(%arg0: tensor<2x9xf32>) -> tensor<2x5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[2, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<2x9xf32>) -> tensor<2x8xf32>
  // CHECK: "mhlo.fft"(%[[SLICED]]) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>}> : (tensor<2x8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<2x9xf32>, tensor<1xi32>) -> tensor<2x5xcomplex<f32>>
  func.return %0 : tensor<2x5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @irfft_1D
func.func @irfft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xf32> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<8xcomplex<f32>>) -> tensor<5xcomplex<f32>>
  // CHECK: "mhlo.fft"(%[[SLICED]]) <{fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT>}> : (tensor<5xcomplex<f32>>
  %0 = "tf.IRFFT"(%arg0, %fftlength) : (tensor<8xcomplex<f32>>, tensor<1xi32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: fft_1D_dynamic
func.func @fft_1D_dynamic(%arg0: tensor<?xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: "tf.FFT"
  %0 = "tf.FFT"(%arg0) : (tensor<?xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: rfft_1D_dynamic
func.func @rfft_1D_dynamic(%arg0: tensor<?xf32>) -> tensor<8xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: "tf.RFFT"
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<?xf32>, tensor<1xi32>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

//===----------------------------------------------------------------------===//
// Shape op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @shape_1D
func.func @shape_1D(%arg0: tensor<?xf32>) -> tensor<1xi32> {
  // CHECK: [[SHAPE:%.+]] = shape.shape_of %arg0
  // CHECK: [[TENSOR:%.+]] = arith.index_cast [[SHAPE]] : tensor<1xindex> to tensor<1xi32>
  %0 = "tf.Shape"(%arg0) : (tensor<?xf32>) -> tensor<1xi32>

  // CHECK: return [[TENSOR]]
  func.return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @shape_2D
func.func @shape_2D(%arg0: tensor<?x?xf32>) -> tensor<2xi32> {
  // CHECK: [[SHAPE:%.+]] = shape.shape_of %arg0
  // CHECK: [[TENSOR:%.+]] = arith.index_cast [[SHAPE]] : tensor<2xindex> to tensor<2xi32>
  %0 = "tf.Shape"(%arg0) : (tensor<?x?xf32>) -> tensor<2xi32>

  // CHECK: return [[TENSOR]]
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @shape_rankless
func.func @shape_rankless(%arg0: tensor<*xf32>) -> tensor<?xi32> {
  // CHECK: [[SHAPE:%.+]] = shape.shape_of %arg0
  // CHECK: [[TENSOR:%.+]] = arith.index_cast [[SHAPE]] : tensor<?xindex> to tensor<?xi32>
  %0 = "tf.Shape"(%arg0) : (tensor<*xf32>) -> tensor<?xi32>

  // CHECK: return [[TENSOR]]
  func.return %0 : tensor<?xi32>
}

//===----------------------------------------------------------------------===//
// Transpose op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @transpose_noop
func.func @transpose_noop(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %permutation = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: return %arg0
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @transpose_2d
func.func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "mhlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_3d_int32
func.func @transpose_3d_int32(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  // CHECK: "mhlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @transpose_3d
func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> (tensor<3xi64>)
  // CHECK: "mhlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_2d
func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "mhlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

//===----------------------------------------------------------------------===//
// Unary op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @abs
func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.abs %arg0 : tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @abs_dynamic
func.func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.abs %arg0 : tensor<?xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @acos
// CHLO-LABEL: @acos
func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  chlo.acos %arg0 : tensor<2xf32>
  // CHLO: %[[TEMP_0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_1:.*]] = mhlo.subtract %[[TEMP_0]], %arg0 : tensor<2xf32>
  // CHLO: %[[TEMP_2:.*]] = mhlo.add %[[TEMP_0]], %arg0 : tensor<2xf32>
  // CHLO: %[[TEMP_3:.*]] = mhlo.multiply %[[TEMP_1]], %[[TEMP_2]] : tensor<2xf32>
  // CHLO: %[[TEMP_4:.*]] = mhlo.sqrt %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_5:.*]] = mhlo.atan2 %[[TEMP_4]], %arg0 : tensor<2xf32>
  // CHLO: return %[[TEMP_5]] : tensor<2xf32>
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @acos_complex
// CHLO-LABEL: @acos_complex
func.func @acos_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: chlo.acos
  // CHLO: %[[TEMP_0:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_1:.*]] = mhlo.abs %[[TEMP_0]] : tensor<2xf32>
  // CHLO: %[[TEMP_2:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_3:.*]] = mhlo.abs %[[TEMP_2]] : tensor<2xf32>
  // CHLO: %[[TEMP_4:.*]] = mhlo.maximum %[[TEMP_1]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_5:.*]] = mhlo.constant dense<3.40282347E+38> : tensor<2xf32>
  // CHLO: %[[TEMP_6:.*]] = mhlo.sqrt %[[TEMP_5]] : tensor<2xf32>
  // CHLO: %[[TEMP_7:.*]] = mhlo.constant dense<8.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_8:.*]] = mhlo.divide %[[TEMP_6]], %[[TEMP_7]] : tensor<2xf32>
  // CHLO: %[[TEMP_9:.*]] = mhlo.compare  GE, %[[TEMP_4]], %[[TEMP_8]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_10:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_11:.*]] = mhlo.compare  LE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_12:.*]] = mhlo.constant dense<5.000000e-01> : tensor<2xf32>
  // CHLO: %[[TEMP_13:.*]] = mhlo.add %[[TEMP_1]], %[[TEMP_10]] : tensor<2xf32>
  // CHLO: %[[TEMP_14:.*]] = mhlo.abs %[[TEMP_13]] : tensor<2xf32>
  // CHLO: %[[TEMP_15:.*]] = mhlo.maximum %[[TEMP_14]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_16:.*]] = mhlo.minimum %[[TEMP_14]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_17:.*]] = mhlo.compare  EQ, %[[TEMP_15]], %[[TEMP_16]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_18:.*]] = mhlo.constant dense<1.41421354> : tensor<2xf32>
  // CHLO: %[[TEMP_19:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_15]] : tensor<2xf32>
  // CHLO: %[[TEMP_20:.*]] = mhlo.divide %[[TEMP_16]], %[[TEMP_15]] : tensor<2xf32>
  // CHLO: %[[TEMP_21:.*]] = mhlo.multiply %[[TEMP_20]], %[[TEMP_20]] : tensor<2xf32>
  // CHLO: %[[TEMP_22:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_21]] : tensor<2xf32>
  // CHLO: %[[TEMP_23:.*]] = mhlo.sqrt %[[TEMP_22]] : tensor<2xf32>
  // CHLO: %[[TEMP_24:.*]] = mhlo.compare  EQ, %[[TEMP_23]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_25:.*]] = mhlo.constant dense<0.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_26:.*]] = mhlo.compare  GT, %[[TEMP_21]], %[[TEMP_25]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_27:.*]] = mhlo.and %[[TEMP_24]], %[[TEMP_26]] : tensor<2xi1>
  // CHLO: %[[TEMP_28:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_21]] : tensor<2xf32>
  // CHLO: %[[TEMP_29:.*]] = mhlo.constant dense<2.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_30:.*]] = mhlo.divide %[[TEMP_28]], %[[TEMP_29]] : tensor<2xf32>
  // CHLO: %[[TEMP_31:.*]] = mhlo.add %[[TEMP_15]], %[[TEMP_30]] : tensor<2xf32>
  // CHLO: %[[TEMP_32:.*]] = mhlo.multiply %[[TEMP_15]], %[[TEMP_23]] : tensor<2xf32>
  // CHLO: %[[TEMP_33:.*]] = mhlo.select %[[TEMP_27]], %[[TEMP_31]], %[[TEMP_32]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_34:.*]] = mhlo.select %[[TEMP_17]], %[[TEMP_19]], %[[TEMP_33]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_35:.*]] = mhlo.subtract %[[TEMP_1]], %[[TEMP_10]] : tensor<2xf32>
  // CHLO: %[[TEMP_36:.*]] = mhlo.abs %[[TEMP_35]] : tensor<2xf32>
  // CHLO: %[[TEMP_37:.*]] = mhlo.maximum %[[TEMP_36]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_38:.*]] = mhlo.minimum %[[TEMP_36]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_39:.*]] = mhlo.compare  EQ, %[[TEMP_37]], %[[TEMP_38]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_40:.*]] = mhlo.multiply %[[TEMP_18]], %[[TEMP_37]] : tensor<2xf32>
  // CHLO: %[[TEMP_41:.*]] = mhlo.divide %[[TEMP_38]], %[[TEMP_37]] : tensor<2xf32>
  // CHLO: %[[TEMP_42:.*]] = mhlo.multiply %[[TEMP_41]], %[[TEMP_41]] : tensor<2xf32>
  // CHLO: %[[TEMP_43:.*]] = mhlo.add %[[TEMP_10]], %[[TEMP_42]] : tensor<2xf32>
  // CHLO: %[[TEMP_44:.*]] = mhlo.sqrt %[[TEMP_43]] : tensor<2xf32>
  // CHLO: %[[TEMP_45:.*]] = mhlo.compare  EQ, %[[TEMP_44]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_46:.*]] = mhlo.compare  GT, %[[TEMP_42]], %[[TEMP_25]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_47:.*]] = mhlo.and %[[TEMP_45]], %[[TEMP_46]] : tensor<2xi1>
  // CHLO: %[[TEMP_48:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_42]] : tensor<2xf32>
  // CHLO: %[[TEMP_49:.*]] = mhlo.divide %[[TEMP_48]], %[[TEMP_29]] : tensor<2xf32>
  // CHLO: %[[TEMP_50:.*]] = mhlo.add %[[TEMP_37]], %[[TEMP_49]] : tensor<2xf32>
  // CHLO: %[[TEMP_51:.*]] = mhlo.multiply %[[TEMP_37]], %[[TEMP_44]] : tensor<2xf32>
  // CHLO: %[[TEMP_52:.*]] = mhlo.select %[[TEMP_47]], %[[TEMP_50]], %[[TEMP_51]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_53:.*]] = mhlo.select %[[TEMP_39]], %[[TEMP_40]], %[[TEMP_52]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_54:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_53]] : tensor<2xf32>
  // CHLO: %[[TEMP_55:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_54]] : tensor<2xf32>
  // CHLO: %[[TEMP_56:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_1]] : tensor<2xf32>
  // CHLO: %[[TEMP_57:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_56]] : tensor<2xf32>
  // CHLO: %[[TEMP_58:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_59:.*]] = mhlo.add %[[TEMP_34]], %[[TEMP_13]] : tensor<2xf32>
  // CHLO: %[[TEMP_60:.*]] = mhlo.divide %[[TEMP_58]], %[[TEMP_59]] : tensor<2xf32>
  // CHLO: %[[TEMP_61:.*]] = mhlo.subtract %[[TEMP_53]], %[[TEMP_35]] : tensor<2xf32>
  // CHLO: %[[TEMP_62:.*]] = mhlo.add %[[TEMP_60]], %[[TEMP_61]] : tensor<2xf32>
  // CHLO: %[[TEMP_63:.*]] = mhlo.multiply %[[TEMP_57]], %[[TEMP_62]] : tensor<2xf32>
  // CHLO: %[[TEMP_64:.*]] = mhlo.sqrt %[[TEMP_63]] : tensor<2xf32>
  // CHLO: %[[TEMP_65:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_59]] : tensor<2xf32>
  // CHLO: %[[TEMP_66:.*]] = mhlo.add %[[TEMP_53]], %[[TEMP_35]] : tensor<2xf32>
  // CHLO: %[[TEMP_67:.*]] = mhlo.divide %[[TEMP_57]], %[[TEMP_66]] : tensor<2xf32>
  // CHLO: %[[TEMP_68:.*]] = mhlo.add %[[TEMP_65]], %[[TEMP_67]] : tensor<2xf32>
  // CHLO: %[[TEMP_69:.*]] = mhlo.sqrt %[[TEMP_68]] : tensor<2xf32>
  // CHLO: %[[TEMP_70:.*]] = mhlo.multiply %[[TEMP_3]], %[[TEMP_69]] : tensor<2xf32>
  // CHLO: %[[TEMP_71:.*]] = mhlo.select %[[TEMP_11]], %[[TEMP_64]], %[[TEMP_70]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_72:.*]] = mhlo.select %[[TEMP_9]], %[[TEMP_3]], %[[TEMP_71]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_73:.*]] = mhlo.constant dense<9.99999995E+11> : tensor<2xf32>
  // CHLO: %[[TEMP_74:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_73]] : tensor<2xf32>
  // CHLO: %[[TEMP_75:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_74]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_76:.*]] = mhlo.constant dense<9.99999997E-7> : tensor<2xf32>
  // CHLO: %[[TEMP_77:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_76]] : tensor<2xf32>
  // CHLO: %[[TEMP_78:.*]] = mhlo.constant dense<1.000000e+02> : tensor<2xf32>
  // CHLO: %[[TEMP_79:.*]] = mhlo.multiply %[[TEMP_8]], %[[TEMP_78]] : tensor<2xf32>
  // CHLO: %[[TEMP_80:.*]] = mhlo.select %[[TEMP_75]], %[[TEMP_77]], %[[TEMP_79]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_81:.*]] = mhlo.compare  GE, %[[TEMP_3]], %[[TEMP_80]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_82:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_3]], %[[TEMP_1]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_83:.*]] = mhlo.select %[[TEMP_81]], %[[TEMP_80]], %[[TEMP_8]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_84:.*]] = mhlo.compare  GE, %[[TEMP_82]], %[[TEMP_83]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_85:.*]] = mhlo.log %[[TEMP_29]] : tensor<2xf32>
  // CHLO: %[[TEMP_86:.*]] = mhlo.log %[[TEMP_82]] : tensor<2xf32>
  // CHLO: %[[TEMP_87:.*]] = mhlo.add %[[TEMP_85]], %[[TEMP_86]] : tensor<2xf32>
  // CHLO: %[[TEMP_88:.*]] = mhlo.constant dense<0x7F800000> : tensor<2xf32>
  // CHLO: %[[TEMP_89:.*]] = mhlo.compare  EQ, %[[TEMP_3]], %[[TEMP_88]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_90:.*]] = mhlo.not %[[TEMP_89]] : tensor<2xi1>
  // CHLO: %[[TEMP_91:.*]] = mhlo.and %[[TEMP_81]], %[[TEMP_90]] : tensor<2xi1>
  // CHLO: %[[TEMP_92:.*]] = mhlo.divide %[[TEMP_1]], %[[TEMP_3]] : tensor<2xf32>
  // CHLO: %[[TEMP_93:.*]] = mhlo.select %[[TEMP_91]], %[[TEMP_92]], %[[TEMP_25]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_94:.*]] = mhlo.multiply %[[TEMP_93]], %[[TEMP_93]] : tensor<2xf32>
  // CHLO: %[[TEMP_95:.*]] = mhlo.log_plus_one %[[TEMP_94]] : tensor<2xf32>
  // CHLO: %[[TEMP_96:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_95]] : tensor<2xf32>
  // CHLO: %[[TEMP_97:.*]] = mhlo.add %[[TEMP_87]], %[[TEMP_96]] : tensor<2xf32>
  // CHLO: %[[TEMP_98:.*]] = mhlo.constant dense<1.17549435E-38> : tensor<2xf32>
  // CHLO: %[[TEMP_99:.*]] = mhlo.sqrt %[[TEMP_98]] : tensor<2xf32>
  // CHLO: %[[TEMP_100:.*]] = mhlo.constant dense<4.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_101:.*]] = mhlo.multiply %[[TEMP_99]], %[[TEMP_100]] : tensor<2xf32>
  // CHLO: %[[TEMP_102:.*]] = mhlo.compare  LT, %[[TEMP_3]], %[[TEMP_101]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_103:.*]] = mhlo.compare  LT, %[[TEMP_1]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_104:.*]] = mhlo.and %[[TEMP_102]], %[[TEMP_103]] : tensor<2xi1>
  // CHLO: %[[TEMP_105:.*]] = mhlo.multiply %[[TEMP_13]], %[[TEMP_35]] : tensor<2xf32>
  // CHLO: %[[TEMP_106:.*]] = mhlo.add %[[TEMP_55]], %[[TEMP_10]] : tensor<2xf32>
  // CHLO: %[[TEMP_107:.*]] = mhlo.divide %[[TEMP_105]], %[[TEMP_106]] : tensor<2xf32>
  // CHLO: %[[TEMP_108:.*]] = mhlo.negate %[[TEMP_107]] : tensor<2xf32>
  // CHLO: %[[TEMP_109:.*]] = mhlo.compare  GE, %[[TEMP_1]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_110:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_58]] : tensor<2xf32>
  // CHLO: %[[TEMP_111:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_59]] : tensor<2xf32>
  // CHLO: %[[TEMP_112:.*]] = mhlo.multiply %[[TEMP_12]], %[[TEMP_66]] : tensor<2xf32>
  // CHLO: %[[TEMP_113:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_112]] : tensor<2xf32>
  // CHLO: %[[TEMP_114:.*]] = mhlo.constant dense<1.500000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_115:.*]] = mhlo.compare  LE, %[[TEMP_55]], %[[TEMP_114]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_116:.*]] = mhlo.divide %[[TEMP_110]], %[[TEMP_61]] : tensor<2xf32>
  // CHLO: %[[TEMP_117:.*]] = mhlo.add %[[TEMP_111]], %[[TEMP_116]] : tensor<2xf32>
  // CHLO: %[[TEMP_118:.*]] = mhlo.subtract %[[TEMP_55]], %[[TEMP_10]] : tensor<2xf32>
  // CHLO: %[[TEMP_119:.*]] = mhlo.select %[[TEMP_115]], %[[TEMP_117]], %[[TEMP_118]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_120:.*]] = mhlo.select %[[TEMP_109]], %[[TEMP_113]], %[[TEMP_119]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_121:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_108]], %[[TEMP_120]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_122:.*]] = mhlo.multiply %[[TEMP_121]], %[[TEMP_106]] : tensor<2xf32>
  // CHLO: %[[TEMP_123:.*]] = mhlo.sqrt %[[TEMP_122]] : tensor<2xf32>
  // CHLO: %[[TEMP_124:.*]] = mhlo.divide %[[TEMP_3]], %[[TEMP_123]] : tensor<2xf32>
  // CHLO: %[[TEMP_125:.*]] = mhlo.add %[[TEMP_121]], %[[TEMP_123]] : tensor<2xf32>
  // CHLO: %[[TEMP_126:.*]] = mhlo.log_plus_one %[[TEMP_125]] : tensor<2xf32>
  // CHLO: %[[TEMP_127:.*]] = mhlo.select %[[TEMP_104]], %[[TEMP_124]], %[[TEMP_126]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_128:.*]] = mhlo.select %[[TEMP_84]], %[[TEMP_97]], %[[TEMP_127]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_129:.*]] = mhlo.complex %[[TEMP_72]], %[[TEMP_128]] : tensor<2xcomplex<f32>>
  // CHLO: %[[TEMP_130:.*]] = mhlo.real %[[TEMP_129]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_131:.*]] = mhlo.real %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_132:.*]] = mhlo.atan2 %[[TEMP_130]], %[[TEMP_131]] : tensor<2xf32>
  // CHLO: %[[TEMP_133:.*]] = mhlo.imag %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_134:.*]] = mhlo.constant dense<0.000000e+00> : tensor<2xf32>
  // CHLO: %[[TEMP_135:.*]] = mhlo.compare  LT, %[[TEMP_133]], %[[TEMP_134]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  // CHLO: %[[TEMP_136:.*]] = mhlo.imag %[[TEMP_129]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  // CHLO: %[[TEMP_137:.*]] = mhlo.negate %[[TEMP_136]] : tensor<2xf32>
  // CHLO: %[[TEMP_138:.*]] = mhlo.select %[[TEMP_135]], %[[TEMP_136]], %[[TEMP_137]] : tensor<2xi1>, tensor<2xf32>
  // CHLO: %[[TEMP_139:.*]] = mhlo.complex %[[TEMP_132]], %[[TEMP_138]] : tensor<2xcomplex<f32>>
  // CHLO: return %[[TEMP_139:.*]] : tensor<2xcomplex<f32>>
  %0 = "tf.Acos"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @acos_dynamic
// CHLO-LABEL: @acos_dynamic
func.func @acos_dynamic(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  chlo.acos %arg0 : tensor<*xf32>
  // `tf.Acos` is lowered to `chlo.constant_like` operations which can only be
  // lowered further on ranked tensors.  Unranked CHLO must be transformed to
  // ranked code before further lowering.
  // CHLO: "tf.Acos"
  %0 = "tf.Acos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @cast_dynamic_i2f
func.func @cast_dynamic_i2f(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK: mhlo.convert %arg0 : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @cast_i2f
func.func @cast_i2f(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: mhlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @cast_c2f
func.func @cast_c2f(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK: mhlo.convert %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @ceil
func.func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.ceil %arg0 : tensor<2xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @ceil_dynamic
func.func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.ceil %arg0 : tensor<?xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @complex_abs
func.func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK:  mhlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.ComplexAbs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @cos
func.func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.cosine %arg0 : tensor<2xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tan
func.func @tan(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.tan %arg0 : tensor<2xf32>
  %0 = "tf.Tan"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @cos_dynamic
func.func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.cosine %arg0 : tensor<?xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @exp
func.func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.exponential %arg0 : tensor<2xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @expm1
func.func @expm1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.exponential_minus_one %arg0 : tensor<2xf32>
  %0 = "tf.Expm1"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @exp_dynamic
func.func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.exponential %arg0 : tensor<?xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @floor
func.func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.floor %arg0 : tensor<2xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @floor_dynamic
func.func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.floor %arg0 : tensor<?xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  // CHECK:  mhlo.is_finite %arg0 : (tensor<2xf32>) -> tensor<2xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// -----

// CHECK-LABEL: func @is_finite_dynamic
func.func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK:  mhlo.is_finite %arg0 : (tensor<?xf32>) -> tensor<?xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @log
func.func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.log %arg0 : tensor<2xf32>
  %0 = "tf.Log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @log_dynamic
func.func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.log %arg0 : tensor<?xf32>
  %0 = "tf.Log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @log1p
func.func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.log_plus_one %arg0 : tensor<2xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @log1p_dynamic
func.func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.log_plus_one %arg0 : tensor<?xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @neg
func.func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.negate %arg0 : tensor<2xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @neg_dynamic
func.func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.negate %arg0 : tensor<?xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: mhlo.logistic
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @sigmoid_complex
func.func @sigmoid_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: mhlo.logistic
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @sigmoid_grad
func.func @sigmoid_grad(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: [[MUL0:%.+]] =  mhlo.multiply %arg1, %arg0 : tensor<2xf32>
  // CHECK-DAG: [[ONE:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK-DAG: [[SUB:%.+]] =  mhlo.subtract [[ONE]], %arg0 : tensor<2xf32>
  // CHECK-DAG: [[MUL1:%.+]] =  mhlo.multiply [[MUL0]], [[SUB]] : tensor<2xf32>
  // CHECK: return [[MUL1]]
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @sigmoid_grad_complex
func.func @sigmoid_grad_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK-DAG: [[MUL0:%.+]] =  mhlo.multiply %arg1, %arg0 : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[ONE:%.+]] = mhlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[SUB:%.+]] =  mhlo.subtract [[ONE]], %arg0 : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[MUL1:%.+]] =  mhlo.multiply [[MUL0]], [[SUB]] : tensor<2xcomplex<f32>>
  // CHECK: return [[MUL1]]
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @sigmoid_grad_dynamic
func.func @sigmoid_grad_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: chlo.broadcast_multiply {{.*}} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: chlo.broadcast_subtract {{.*}} {broadcast_dimensions = array<i64>} : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: chlo.broadcast_multiply {{.*}} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

