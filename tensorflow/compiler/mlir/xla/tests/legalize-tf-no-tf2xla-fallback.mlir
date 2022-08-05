// RUN: xla-opt "-xla-legalize-tf-no-fallback=allow-partial-conversion" -split-input-file %s | FILECHECK_OPTS="" FileCheck %s

//===----------------------------------------------------------------------===//
// BatchNorm op legalizations.
//===----------------------------------------------------------------------===//

// -----

// fusedBatchNormV2 is almost identical to fusedBatchNormV3 (and uses the same
// code), so only do a couple of basic checks.

// CHECK-LABEL: fusedBatchNormV2_noTraining
func.func @fusedBatchNormV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV2_training
func.func @fusedBatchNormV2_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.constant
  // CHECK: chlo.broadcast_multiply %[[VAR]], {{.*}} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining
func.func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining_mixedPrecision
// CHECK-SAME:  ([[X:%.*]]: tensor<8x8x8x8xbf16>, [[SCALE:%.*]]: tensor<8xf32>, [[OFFSET:%.*]]: tensor<8xf32>, [[MEAN:%.*]]: tensor<8xf32>, [[VARIANCE:%.*]]: tensor<8xf32>)
func.func @fusedBatchNormV3_noTraining_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) {
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert([[X]]) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK: [[Y:%.*]] = "mhlo.batch_norm_inference"([[CONVERT_X]], [[SCALE]], [[OFFSET]], [[MEAN]], [[VARIANCE]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)
  // CHECK: [[Y_CONVERT:%.*]] = mhlo.convert([[Y]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK: [[DUMMY:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<0xf32>
  // CHECK: [[DUMMY_CAST:%.*]] = tensor.cast [[DUMMY]] : tensor<0xf32> to tensor<*xf32>
  // CHECK: return [[Y_CONVERT]], [[MEAN]], [[VARIANCE]], [[MEAN]], [[VARIANCE]], [[DUMMY_CAST]]
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training
func.func @fusedBatchNormV3_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.constant
  // CHECK: chlo.broadcast_multiply %[[VAR]], {{.*}} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: func @fusedBatchNormV3_training_batchVariance
func.func @fusedBatchNormV3_training_batchVariance(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: return %[[VAR]]
  func.return %0#4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_exponentialAvgFactor
func.func @fusedBatchNormV3_training_exponentialAvgFactor(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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
  // CHECK: mhlo.convert(%arg0) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: mhlo.convert({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_NCHW
func.func @fusedBatchNormV3_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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
  // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
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

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
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
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
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
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_noTraining_mixed_precision
func.func @fusedBatchNormGradV2_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = mhlo.convert({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_Training_mixed_precision
func.func @fusedBatchNormGradV2_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert(%[[grad_operand]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
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

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
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
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = mhlo.convert({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training_mixed_precision
func.func @fusedBatchNormGradV3_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = mhlo.convert(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert(%[[grad_operand]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
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

  // CHECK-NEXT: %[[add:.*]] = chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = mhlo.rsqrt %[[add]] : tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: mhlo.constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[init]]) applies mhlo.add across dimensions = [0, 2, 3] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
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
  // CHECK: %{{.*}} = "mhlo.batch_norm_grad"(%{{.*}}, %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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
  // CHECK: [[VAL:%.+]] = "mhlo.clamp"(%arg1, %arg0, %arg2)

  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return [[VAL]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @clip_dynamic
func.func @clip_dynamic(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: [[CLAMP:%.+]] = "mhlo.clamp"(%arg1, %arg0, %arg2)
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @clip_static_broadcast
func.func @clip_static_broadcast(%arg0 : tensor<5xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<5xf32> {
  // CHECK-DAG: [[SHPIDX:%.+]] = mhlo.constant dense<5>
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, [[SHPIDX]]) {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg2, [[SHPIDX]]) {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[CLAMP:%.+]] = "mhlo.clamp"([[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]])
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<5xf32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<5xf32>
}


// CHECK-LABEL: @clip_dynamic_broadcast
func.func @clip_dynamic_broadcast(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?xf32> {
  // CHECK: [[SHP:%.+]] = shape.shape_of %arg0
  // CHECK: [[SHPIDX:%.+]] = arith.index_cast [[SHP]] : tensor<1xindex> to tensor<1xi32>
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, [[SHPIDX]]) {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg2, [[SHPIDX]]) {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[CLAMP:%.+]] = "mhlo.clamp"([[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]])
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
  // CHECK: %[[RS:.*]] = "mhlo.reshape"(%[[ARG]]) : (tensor<4x3x4x3xf32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[IOTA0:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[IOTA1:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[COMP:.*]] = "mhlo.compare"(%[[IOTA0]], %[[IOTA1]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<12x12xi32>, tensor<12x12xi32>) -> tensor<12x12xi1>
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZERO_MAT:.*]] = "mhlo.broadcast"(%[[ZERO]]) {broadcast_sizes = dense<12> : tensor<2xi64>} : (tensor<f32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[SEL:.*]] = "mhlo.select"(%[[COMP]], %[[RS]], %[[ZERO_MAT]]) : (tensor<12x12xi1>, tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[RED:.*]] = mhlo.reduce(%[[SEL]] init: %[[ZERO]]) applies mhlo.add across dimensions = [0] : (tensor<12x12xf32>, tensor<f32>) -> tensor<12xf32>
  // CHECK-DAG:  %[[RES:.*]] = "mhlo.reshape"(%[[RED]]) : (tensor<12xf32>) -> tensor<4x3xf32>
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
  // CHECK-DAG: %[[V2:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V3:.*]] = "mhlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V4:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[V5:.*]] = "mhlo.broadcast"(%[[V4]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V6:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[V7:.*]] = "mhlo.broadcast"(%[[V6]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V8:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK-DAG: %[[V9:.*]] = "mhlo.broadcast"(%[[V8]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V10:.*]] = mhlo.constant dense<11> : tensor<i32>
  // CHECK-DAG: %[[V11:.*]] = "mhlo.broadcast"(%[[V10]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V12:.*]] = mhlo.constant dense<140> : tensor<i32>
  // CHECK-DAG: %[[V13:.*]] = "mhlo.broadcast"(%[[V12]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V14:.*]] = mhlo.constant dense<128> : tensor<i32>
  // CHECK-DAG: %[[V15:.*]] = "mhlo.broadcast"(%[[V14]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V16:.*]] = mhlo.constant dense<128> : tensor<i32>
  // CHECK-DAG: %[[V17:.*]] = "mhlo.broadcast"(%[[V16]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V18:.*]] = mhlo.subtract %[[V11]], %[[V2]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V19:.*]] = mhlo.negate %[[V18]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V20:.*]] = mhlo.minimum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V21:.*]] = mhlo.add %[[V13]], %[[V20]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V22:.*]] = mhlo.maximum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V23:.*]] = mhlo.subtract %[[V15]], %[[V22]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V24:.*]] = mhlo.minimum %[[V21]], %[[V23]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V25:.*]] = chlo.broadcast_compare %[[V18]], %[[V5]] {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V26:.*]] = mhlo.subtract %[[V17]], %[[V24]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V27:.*]] = "mhlo.select"(%[[V25]], %[[V26]], %[[V5]]) : (tensor<1x22x128xi1>, tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi32>
  // CHECK-DAG: %[[V28:.*]] = mhlo.maximum %[[V18]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V29:.*]] = mhlo.subtract %[[V28]], %[[V27]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V30:.*]] = mhlo.maximum %[[V19]], %[[V5]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V31:.*]] = mhlo.subtract %[[V30]], %[[V27]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V32:.*]] = mhlo.add %[[V3]], %[[V29]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V33:.*]] = mhlo.add %[[V3]], %[[V31]] : tensor<1x22x128xi32>
  // CHECK-DAG: %[[V34:.*]] = chlo.broadcast_compare %[[V32]], %[[V5]] {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V35:.*]] = chlo.broadcast_compare %[[V32]], %[[V15]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V36:.*]] = mhlo.and %[[V34]], %[[V35]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V37:.*]] = chlo.broadcast_compare %[[V33]], %[[V5]] {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V38:.*]] = chlo.broadcast_compare %[[V33]], %[[V13]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-DAG: %[[V39:.*]] = mhlo.and %[[V37]], %[[V38]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V40:.*]] = mhlo.and %[[V36]], %[[V39]] : tensor<1x22x128xi1>
  // CHECK-DAG: %[[V41:.*]] = "mhlo.reshape"(%[[V40]]) : (tensor<1x22x128xi1>) -> tensor<22x128xi1>
  // CHECK-DAG: %[[V42:.*]] = "mhlo.concatenate"(%[[V33]], %[[V32]]) {dimension = 0 : i64} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<2x22x128xi32>
  // CHECK-DAG: %[[V43:.*]] = "mhlo.gather"(%[[ARG]], %[[V42]]) {dimension_numbers = #mhlo.gather<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [1, 2]>, indices_are_sorted = false, slice_sizes = dense<[7, 1, 1]> : tensor<3xi64>} : (tensor<7x140x128xi32>, tensor<2x22x128xi32>) -> tensor<7x22x128xi32>
  // CHECK-DAG: %[[V44:.*]] = "mhlo.broadcast"(%[[V41]]) {broadcast_sizes = dense<7> : tensor<1xi64>} : (tensor<22x128xi1>) -> tensor<7x22x128xi1>
  // CHECK-DAG: %[[V45:.*]] = "mhlo.broadcast"(%[[V0]]) {broadcast_sizes = dense<[7, 22, 128]> : tensor<3xi64>} : (tensor<i32>) -> tensor<7x22x128xi32>
  // CHECK: %[[V46:.*]] = "mhlo.select"(%[[V44]], %[[V43]], %[[V45]]) : (tensor<7x22x128xi1>, tensor<7x22x128xi32>, tensor<7x22x128xi32>) -> tensor<7x22x128xi32>
  // CHECK: return %[[V46]] : tensor<7x22x128xi32>
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
  func.return %2: tensor<7x22x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_single_diagonal
func.func @matrix_diag_part_single_diagonal(%arg0: tensor<7x140x128xi32>) -> tensor<7x128xi32> {
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<0> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x128xi32>
  // CHECK: %[[result:.*]] = "mhlo.reshape"({{.*}}) : (tensor<7x1x128xi32>) -> tensor<7x128xi32>
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
  // CHECK: %[[b_false:.*]] = "mhlo.broadcast"(%[[false]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = "mhlo.select"(%[[b_false]], %{{[0-9]*}}, %{{[0-9]*}}) : (tensor<1x22x128xi1>, tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi32>
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
  // CHECK: %[[le:.*]] = chlo.broadcast_compare %{{[0-9]*}}, %{{[0-9]*}} {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = "mhlo.select"(%[[le]], %{{[0-9]*}}, %{{[0-9]*}}) : (tensor<1x22x128xi1>, tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi32>
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
  // CHECK: %[[ge:.*]] = chlo.broadcast_compare %{{[0-9]*}}, %{{[0-9]*}} {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = "mhlo.select"(%[[ge]], %{{[0-9]*}}, %{{[0-9]*}}) : (tensor<1x22x128xi1>, tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi32>
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
  // CHECK: %[[b_true:.*]] = "mhlo.broadcast"(%[[true]]) {broadcast_sizes = dense<[1, 22, 128]> : tensor<3xi64>} : (tensor<i1>) -> tensor<1x22x128xi1>
  // CHECK: %{{[0-9]*}} = "mhlo.select"(%[[b_true]], %{{[0-9]*}}, %{{[0-9]*}}) : (tensor<1x22x128xi1>, tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi32>
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
  // CHECK: chlo.erf %arg0 : tensor<2x3xf32>
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
  // CHECK:  mhlo.unary_einsum
  %0 = "tf.Einsum"(%arg0) {equation = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  func.return %0: tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// FloorDiv and FloorMod.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @floordiv_broadcast_i32
func.func @floordiv_broadcast_i32(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[SUB]], [[DIV]])
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floordiv_reverse_broadcast_i32
func.func @floordiv_reverse_broadcast_i32(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]]
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[SUB]], [[DIV]])
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
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[SUB]], [[DIV]])
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floordiv_unranked
func.func @floordiv_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NOT: tf.FloorDiv
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}

// -----

// CHECK-LABEL: func @floordiv_int
func.func @floordiv_int(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK-DAG: [[DIV:%.+]] = chlo.broadcast_divide %arg0, %arg1 : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[DIV]], %arg1 : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[MUL]], %arg0 {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi1>
  // CHECK-DAG: [[ZEROS1:%.+]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  // CHECK-DAG: [[ZEROS2:%.+]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ONES:%.+]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-DAG: [[SUB:%.+]] = chlo.broadcast_subtract [[DIV]], [[ONES]]
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[SUB]], [[DIV]])
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_numerator
func.func @floormod_broadcast_numerator(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = dense<> : tensor<0xi64>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]]
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[ADD]], [[REM]])
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_denominator
func.func @floormod_broadcast_denominator(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = dense<> : tensor<0xi64>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[ADD]], [[REM]])
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_unsigned_broadcast_denominator
func.func @floormod_unsigned_broadcast_denominator(%arg0: tensor<2x3xui32>, %arg1: tensor<3xui32>) -> tensor<2x3xui32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-NEXT: return [[REM]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xui32>, tensor<3xui32>) -> tensor<2x3xui32>
  func.return %0: tensor<2x3xui32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_numerator
func.func @floormod_dynamic_broadcast_numerator(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = dense<> : tensor<0xi64>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = #mhlo<comparison_direction NE>}
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[ADD]], [[REM]])
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_denominator
func.func @floormod_dynamic_broadcast_denominator_(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: tf.FloorMod
  // CHECK-DAG: [[REM:%.+]] = chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-DAG: [[ZL:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[CMP1:%.+]] = chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[ZR:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[CMP2:%.+]] = chlo.broadcast_compare %arg1, [[ZR]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[CMP3:%.+]] = chlo.broadcast_compare [[REM]], [[ZR]] {broadcast_dimensions = dense<> : tensor<0xi64>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[CMP4:%.+]] = chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[AND:%.+]] = chlo.broadcast_and [[CMP1]], [[CMP4]] : (tensor<?x?x?xi1>, tensor<?x?x?xi1>) -> tensor<?x?x?xi1>
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add %arg1, [[REM]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-DAG: [[SELECT:%.+]] = "mhlo.select"([[AND]], [[ADD]], [[REM]]) : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: return [[SELECT]] : tensor<?x?x?xf32>
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0: tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @floormod_unranked
func.func @floormod_unranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK-NOT: tf.FloorMod
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

//===----------------------------------------------------------------------===//
// OnesLike
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @ones_like
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x?xf32>)
func.func @ones_like(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[RES:.*]] = "chlo.constant_like"(%[[ARG]]) {value = 1.0{{.*}}}
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
  // CHECK: %[[RES:.*]] = "chlo.constant_like"(%[[ARG]]) {value = 0.0{{.*}}}
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

// -----

// CHECK-LABEL: func @broadcast_scalar_to_unranked
// CHECK: (%[[ARG0:.*]]: tensor<f32>, %[[SHAPE:.*]]: tensor<?xi32>)
func.func @broadcast_scalar_to_unranked(%arg0: tensor<f32>, %shape: tensor<?xi32>) -> tensor<*xf32> {
  // CHECK: "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]])
  // CHECK-SAME: {broadcast_dimensions = dense<> : tensor<0xi64>}
  %0 = "tf.BroadcastTo"(%arg0, %shape) : (tensor<f32>, tensor<?xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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
  // CHECK: "mhlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_neg_axis
func.func @concat_v2_neg_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "mhlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>

  %axis = "tf.Const"() { value = dense<-2> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_1d_axis
func.func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  // CHECK: "mhlo.concatenate"({{.*}}) {dimension = 1 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x6xf32>

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

// -----

// CHECK-LABEL: func @concat_v2_unranked
func.func @concat_v2_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  // CHECK: "tf.ConcatV2"
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<*xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// Pad op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @padv2_1D
func.func @padv2_1D(%arg0: tensor<3xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK: "mhlo.pad"(%arg0, %arg1) {
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
  // CHECK: "mhlo.pad"(%arg0, %arg1) {
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
  // CHECK: "mhlo.pad"(%arg0, %arg1) {
  // CHECK-SAME:    edge_padding_high = dense<[2, 4]> : tensor<2xi64>,
  // CHECK-SAME:    edge_padding_low = dense<[1, 3]> : tensor<2xi64>,
  // CHECK-SAME:    interior_padding = dense<0> : tensor<2xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<6x9xf32>
  func.return %1 : tensor<6x9xf32>
}

// -----

// CHECK-LABEL: func @padv2_dynamic
func.func @padv2_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1x2xi64>) -> tensor<?xf32> {
  // CHECK: "mhlo.transpose"({{.*}}) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<1x2xi64>) -> tensor<2x1xi64>
  // CHECK: "mhlo.reshape"({{.*}}) : (tensor<2x1xi64>) -> tensor<2xi64>
  // CHECK: "mhlo.slice"({{.*}}) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: "mhlo.slice"({{.*}}) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: "mhlo.dynamic_pad"({{.*}}) : (tensor<?xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
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
// CHECK: [[TOKEN:%.*]] = "mhlo.create_token"() : () -> !mhlo.token
// CHECK: [[INFEED:%.*]]:3 = "mhlo.infeed"([[TOKEN]]) {infeed_config = ""{{.*}}} : (!mhlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !mhlo.token)
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

// CHECK-LABEL: @const_with_tensor_proto_attr
func.func @const_with_tensor_proto_attr() -> tensor<!tf_type.variant<tensor<2xi32>>> {
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
  // CHECK: %[[UPDATED_B:.*]] = "mhlo.transpose"(%[[B]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK: "mhlo.dot"(%[[A]], %[[UPDATED_B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = true} : (tensor<5x7xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// -----

// CHECK-LABEL: matmul_transpose_both
// CHECK-SAME: (%[[A:.*]]: tensor<7x5xf32>, %[[B:.*]]: tensor<11x7xf32>)
func.func @matmul_transpose_both(%a: tensor<7x5xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_A:.*]] = "mhlo.transpose"(%[[A]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK: %[[UPDATED_B:.*]] = "mhlo.transpose"(%[[B]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
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

// Verify that MatMul with unranked inputs are lowered to HLO.
// CHECK-LABEL: matmul_unranked
func.func @matmul_unranked(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "mhlo.dot"
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  func.return %0 : tensor<*xf32>
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
// CHECK:           %[[CAST:.*]] = mhlo.convert(%[[ARG1]])
// CHECK-SAME:        -> tensor<4x5xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.dot"(%[[ARG0]], %[[CAST]])
// CHECK-SAME:        -> tensor<3x5xf32>
// CHECK:           return %[[RESULT]]
func.func @test_sparse_mat_mul_with_cast(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xbf16>) -> tensor<3x5xf32> {
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = false} : (tensor<3x4xf32>, tensor<4x5xbf16>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

//===----------------------------------------------------------------------===//
// MatrixBandPart op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: matrix_band_part
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func.func @matrix_band_part(%arg0: tensor<64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<64x64xbf16> {
  // CHECK-DAG: %[[M:.*]] = mhlo.constant dense<64> : tensor<i64>
  // CHECK-DAG: %[[N:.*]] = mhlo.constant dense<64> : tensor<i64>

  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK-DAG: %[[A:.*]] = "mhlo.compare"(%[[LOWER]], %[[ZERO]]) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-DAG: %[[B:.*]] = "mhlo.select"(%[[A]], %[[M]], %[[LOWER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>

  // CHECK-DAG: %[[C:.*]] = "mhlo.compare"(%[[UPPER]], %[[ZERO]]) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-DAG: %[[D:.*]] = "mhlo.select"(%[[C]], %[[N]], %[[UPPER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK-DAG: %[[F:.*]] = mhlo.negate %[[B]] : tensor<i64>

  // CHECK-DAG: %[[X:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<64x64xi64>
  // CHECK-DAG: %[[Y:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<64x64xi64>
  // CHECK-DAG: %[[OFFSET:.*]] = mhlo.subtract %[[X]], %[[Y]] : tensor<64x64xi64>
  // CHECK-DAG: %[[G:.*]] = chlo.broadcast_compare %[[F]], %[[OFFSET]] {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<i64>, tensor<64x64xi64>) -> tensor<64x64xi1>

  // CHECK-DAG: %[[I:.*]] = chlo.broadcast_compare %[[OFFSET]], %[[D]] {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<64x64xi64>, tensor<i64>) -> tensor<64x64xi1>

  // CHECK-DAG: %[[J:.*]] = mhlo.and %[[G]], %[[I]] : tensor<64x64xi1>

  // CHECK-DAG: %[[ZERO2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<64x64xbf16>

  // CHECK-DAG: %[[R:.*]] = chlo.broadcast_select %[[J]], %[[INPUT]], %[[ZERO2]]
  // CHECK-DAG: return %[[R]]
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<64x64xbf16>
  func.return %0 : tensor<64x64xbf16>
}

// -----

// CHECK-LABEL: matrix_band_part_2
// CHECK-SAME: (%[[INPUT:.*]]: tensor<12x24x48xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func.func @matrix_band_part_2(%arg0: tensor<12x24x48xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<12x24x48xbf16> {
  // CHECK-DAG: %[[X:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<24x48xi64>
  // CHECK-DAG: %[[Y:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<24x48xi64>
  // CHECK-DAG: %[[OFFSET:.*]] = mhlo.subtract %[[X]], %[[Y]] : tensor<24x48xi64>

  // CHECK-DAG: %[[G:.*]] = chlo.broadcast_compare %[[F]], %[[OFFSET]] {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<i64>, tensor<24x48xi64>) -> tensor<24x48xi1>

  // CHECK-DAG: %[[I:.*]] = chlo.broadcast_compare %[[OFFSET]], %[[D]] {comparison_direction = #mhlo<comparison_direction LE>} : (tensor<24x48xi64>, tensor<i64>) -> tensor<24x48xi1>
  // CHECK-DAG: %[[J:.*]] = mhlo.and %[[G]], %[[I]] : tensor<24x48xi1>

  // CHECK-DAG: %[[ZERO2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<12x24x48xbf16>

  // CHECK-DAG: %[[R:.*]] = chlo.broadcast_select %[[J]], %[[INPUT]], %[[ZERO2]]
  // CHECK-DAG: return %[[R]]
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<12x24x48xbf16>, tensor<i64>, tensor<i64>) -> tensor<12x24x48xbf16>
  func.return %0 : tensor<12x24x48xbf16>
}

// -----

// CHECK-LABEL: matrix_band_part_3
// CHECK-SAME: (%[[INPUT:.*]]: tensor<*xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func.func @matrix_band_part_3(%arg0: tensor<*xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<*xbf16> {
  // CHECK: "tf.MatrixBandPart"
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<*xbf16>, tensor<i64>, tensor<i64>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

// -----

// CHECK-LABEL: matrix_band_part_4
// CHECK-SAME: (%[[INPUT:.*]]: tensor<24x48xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func.func @matrix_band_part_4(%arg0: tensor<24x48xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<24x48xbf16> {
  // This one should lower.
  // CHECK-NOT: "tf.MatrixBandPart"
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<24x48xbf16>, tensor<i64>, tensor<i64>) -> tensor<24x48xbf16>
  func.return %0 : tensor<24x48xbf16>
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
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return
  // CHECK: {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>}

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
  // CHECK: mhlo.maximum
  // CHECK: mhlo.return
  // CHECK: {window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 4, 4, 1]> : tensor<5xi64>}

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
  // CHECK: %[[RESULT:.*]] = "mhlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = "mhlo.compare"(%[[VALUE_A]], %[[VALUE_B]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: "mhlo.return"(%[[SELECT_RESULT]]) : (tensor<i1>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: "mhlo.return"(%[[SELECT_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
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
  // CHECK: %[[RESULT:.*]] = "mhlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = "mhlo.compare"(%[[VALUE_A]], %[[VALUE_B]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: "mhlo.return"(%[[SELECT_RESULT]]) : (tensor<i1>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = mhlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: "mhlo.return"(%[[SELECT_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>} : (tensor<10x8x24x24x64xf32>, tensor<10x8x12x12x64xf32>, tensor<f32>) -> tensor<10x8x24x24x64xf32>
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
  // CHECK: %[[IOTA:.*]] = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<3x5xi32>
  // CHECK: %[[BCAST_ARG0:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<3x5xi32>
  // CHECK: %[[COMPARE:.*]] = "mhlo.compare"(%[[BCAST_ARG0]], %[[IOTA]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<3x5xi1>
  // CHECK: %[[ON_VALUE:.*]] = "mhlo.broadcast"(%arg1) {broadcast_sizes = dense<[3, 5]> : tensor<2xi64>} : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[OFF_VALUE:.*]] = "mhlo.broadcast"(%arg2) {broadcast_sizes = dense<[3, 5]> : tensor<2xi64>} : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[COMPARE]], %[[ON_VALUE]], %[[OFF_VALUE]]) : (tensor<3x5xi1>, tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
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
// CHECK: [[TOKEN:%.*]] = "mhlo.create_token"() : () -> !mhlo.token
// CHECK: "mhlo.outfeed"([[VAL_0]],  [[VAL_1]], [[TOKEN]]) {outfeed_config = ""} : (tensor<3xi32>, tensor<4xf32>, !mhlo.token) -> !mhlo.token
  "tf.OutfeedEnqueueTuple"(%data_1, %data_2) : (tensor<3xi32>, tensor<4xf32>) -> ()
  func.return
}

//===----------------------------------------------------------------------===//
// Pack op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @pack
func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "mhlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "mhlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "mhlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

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

//===----------------------------------------------------------------------===//
// ReverseV2 op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @reverse_func_32
func.func @reverse_func_32(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi32>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_64
func.func @reverse_func_64(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> (tensor<1xi64>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi64>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_neg
func.func @reverse_func_neg(%arg0: tensor<5x5xi32>) -> tensor<5x5xi32> {
  %axis = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "mhlo.reverse"(%arg0) {dimensions = dense<1> : tensor<1xi64>}
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
  // CHECK-DAG: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-DAG: %[[PRED:.*]] = "mhlo.compare"(%arg0, %[[ZERO]]) {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: %[[EXP:.*]] = mhlo.exponential_minus_one %arg0
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[PRED]], %arg0, %[[EXP]])
  // CHECK: return %[[RESULT]]
  %0 = "tf.Elu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

// -----

// CHECK-LABEL: func @elu_unranked
func.func @elu_unranked(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK-DAG: %[[PRED:.*]] = "mhlo.compare"(%arg0, %[[ZERO]]) {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: %[[EXP:.*]] = mhlo.exponential_minus_one %arg0
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[PRED]], %arg0, %[[EXP]])
  // CHECK: return %[[RESULT]]
  %0 = "tf.Elu"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @elu_grad
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<4x8xf32>, %[[FEATURES:.*]]: tensor<?x?xf32>)
func.func @elu_grad(%gradients: tensor<4x8xf32>, %features: tensor<?x?xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[PRED:.*]] = chlo.broadcast_compare %[[FEATURES]], %[[ZERO]] {broadcast_dimensions = dense<> : tensor<0xi64>, comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: %[[ADD1:.*]] = chlo.broadcast_add %[[FEATURES]], %[[ONE]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: %[[MULGRAD:.*]] = mhlo.multiply(%[[GRADIENTS]], %[[ADD1]]) : (tensor<4x8xf32>, tensor<?x?xf32>) -> tensor<4x8xf32>
  // CHECK: %[[RESULT:.*]] = "mhlo.select"(%[[PRED]], %[[GRADIENTS]], %[[MULGRAD]])
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
  // CHECK: chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu_unranked
func.func @relu_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// -----

// CHECK-LABEL: func @relu6
func.func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[SIX:.*]] = mhlo.constant dense<6> : tensor<i32>
  // CHECK: "mhlo.clamp"(%[[ZERO]], %arg0, %[[SIX]]) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu6_unranked
func.func @relu6_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[SIX:.*]] = mhlo.constant dense<6> : tensor<i32>
  // CHECK: "mhlo.clamp"(%[[ZERO]], %arg0, %[[SIX]]) : (tensor<i32>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// -----

// CHECK-LABEL: func @relu_grad_unranked
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<?x?xf32>, %[[FEATURES:.*]]: tensor<?x?xf32>)
func.func @relu_grad_unranked(%gradients: tensor<?x?xf32>, %features: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "chlo.constant_like"(%arg1) {value = 0.000000e+00 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[PRED:.*]] = "mhlo.compare"(%arg1, %0) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK-DAG: %[[RESULT:.*]] = "mhlo.select"(%[[PRED]], %[[GRADIENTS]], %[[ZERO]]) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: return %[[RESULT]] : tensor<?x?xf32>
  %2 = "tf.ReluGrad"(%gradients, %features) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu
func.func @leaky_relu(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg0) {value = 2.000000e-01 : f32} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[LEAKY:.*]] = mhlo.multiply %[[INP:.*]], %[[ALPHA]] : tensor<1x4x4x3xf32>
    // CHECK-NEXT: %[[CMP:.*]] = "mhlo.compare"(%[[INP]], %[[ZERO]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x4x4x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xi1>
    // CHECK-NEXT: %[[RES:.*]] = "mhlo.select"(%[[CMP]], %[[INP]], %[[LEAKY]]) : (tensor<1x4x4x3xi1>, tensor<1x4x4x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4x3xf32>
    %0 = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    func.return %0 : tensor<1x4x4x3xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu_unranked
func.func @leaky_relu_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg0) {value = 2.000000e-01 : f32} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg0) {value = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: %[[LEAKY:.*]] = mhlo.multiply %[[INP:.*]], %[[ALPHA]] : tensor<*xf32>
    // CHECK-NEXT: %[[CMP:.*]] = "mhlo.compare"(%[[INP]], %[[ZERO]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
    // CHECK-NEXT: %[[RES:.*]] = "mhlo.select"(%[[CMP]], %[[INP]], %[[LEAKY]]) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<*xf32>
    %0 = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu_grad
func.func @leaky_relu_grad(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg1) {value = 2.000000e-01 : f32} : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg1) {value = 0.000000e+00 : f32} : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: %[[LEAKYGRAD:.*]] = mhlo.multiply %[[GRADIENT:.*]], %[[ALPHA]] : tensor<1x4x4xf32>
    // CHECK-NEXT: %[[CMP:.*]] = "mhlo.compare"(%[[INP:.*]], %[[ZERO]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
    // CHECK-NEXT: %[[RES:.*]] = "mhlo.select"(%[[CMP]], %[[GRADIENT]], %[[LEAKYGRAD]]) : (tensor<1x4x4xi1>, tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4xf32>
    %0 = "tf.LeakyReluGrad"(%arg0, %arg1) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    func.return %0 : tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu_grad_unranked
func.func @leaky_relu_grad_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = "chlo.constant_like"(%arg1) {value = 2.000000e-01 : f32} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: %[[ZERO:.*]] = "chlo.constant_like"(%arg1) {value = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: %[[LEAKYGRAD:.*]] = mhlo.multiply %[[GRADIENT:.*]], %[[ALPHA]] : tensor<*xf32>
    // CHECK-NEXT: %[[CMP:.*]] = "mhlo.compare"(%[[INP:.*]], %[[ZERO]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
    // CHECK-NEXT: %[[RES:.*]] = "mhlo.select"(%[[CMP]], %[[GRADIENT]], %[[LEAKYGRAD]]) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: return %[[RES]] : tensor<*xf32>
    %0 = "tf.LeakyReluGrad"(%arg0, %arg1) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @softsign
func.func @softsign(%arg0: tensor<4x10xf32>) -> tensor<4x10xf32> {
    // CHECK-NEXT: %[[ONE:.*]] = "chlo.constant_like"(%arg0) {value = 1.000000e+00 : f32} : (tensor<4x10xf32>) -> tensor<4x10xf32>
    // CHECK-NEXT: %[[ABS:.*]] = mhlo.abs %{{.*}} : tensor<4x10xf32>
    // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %[[ONE]], %[[ABS]] : tensor<4x10xf32>
    // CHECK-NEXT: %[[DIV:.*]] = mhlo.divide %{{.*}}, %[[ADD]] : tensor<4x10xf32>
    // CHECK-NEXT: return %[[DIV]] : tensor<4x10xf32>
    %0 = "tf.Softsign"(%arg0) : (tensor<4x10xf32>) -> tensor<4x10xf32>
    func.return %0 : tensor<4x10xf32>
}

// -----

// CHECK-LABEL: func @softsign_unranked
func.func @softsign_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK-NEXT: %[[ONE:.*]] = "chlo.constant_like"(%arg0) {value = 1.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK-NEXT: %[[ABS:.*]] = mhlo.abs %{{.*}} : tensor<*xf32>
    // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %[[ONE]], %[[ABS]] : tensor<*xf32>
    // CHECK-NEXT: %[[DIV:.*]] = mhlo.divide %{{.*}}, %[[ADD]] : tensor<*xf32>
    // CHECK-NEXT: return %[[DIV]] : tensor<*xf32>
    %0 = "tf.Softsign"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @softsign_grad
func.func @softsign_grad(%arg0: tensor<4x10xf32>, %arg1: tensor<4x10xf32>) -> tensor<4x10xf32> {

    // CHECK-NEXT: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
    // CHECK-NEXT: %[[ABS:.*]] = mhlo.abs %{{.*}} : tensor<4x10xf32>
    // CHECK-NEXT: %[[BROADCAST_ADD:.*]] = chlo.broadcast_add %[[ONE]], %[[ABS]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<4x10xf32>) -> tensor<4x10xf32>
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
  //      CHECK: %[[CONCAT:.+]] = "mhlo.concatenate"(%arg0, %arg0) {dimension = 0 : i64}
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
  // CHECK: %[[BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %{{.*}}) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi1>, tensor<3xindex>) -> tensor<2x6x8xi1>
  // CHECK: "mhlo.select"(%[[BCAST]], %arg1, %arg2)
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_r1
func.func @select_batch_static_r1(%arg0: tensor<i1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: "mhlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_all_same
func.func @select_batch_static_all_same(%arg0: tensor<2x6x8xi1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: "mhlo.select"(%arg0, %arg1, %arg2)
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
  // CHECK-NEXT: %[[BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[SHAPE1E]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xi1>, tensor<3xindex>) -> tensor<?x?x8xi1>
  // CHECK-NEXT: %[[SELECT:.*]] = "mhlo.select"(%[[BCAST]], %arg1, %arg2) : (tensor<?x?x8xi1>, tensor<?x?x8xi32>, tensor<?x?x8xi32>) -> tensor<?x?x8xi32>
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
  // CHECK-NEXT: %[[SELECT:.*]] = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<?x?x8xi1>, tensor<?x?x8xi32>, tensor<?x?x8xi32>) -> tensor<?x?x8xi32>
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

// -----

// CHECK-LABEL: func @selectv2_unranked
func.func @selectv2_unranked(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: chlo.broadcast_select
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

//===----------------------------------------------------------------------===//
// Fast Fourier Transform op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @fft_1D
func.func @fft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: "mhlo.fft"(%arg0) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT>} : (tensor<8xcomplex<f32>>
  %0 = "tf.FFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @ifft_1D
func.func @ifft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: "mhlo.fft"(%arg0) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type IFFT>} : (tensor<8xcomplex<f32>>
  %0 = "tf.IFFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D
func.func @rfft_1D(%arg0: tensor<8xf32>) -> tensor<8xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: "mhlo.fft"(%arg0) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>} : (tensor<8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<8xf32>, tensor<1xi32>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_padded
func.func @rfft_1D_padded(%arg0: tensor<7xf32>) -> tensor<8xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[PADDED:.*]] = "mhlo.pad"(%arg0, %{{.*}}) {edge_padding_high = dense<1> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<7xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK: "mhlo.fft"(%[[PADDED]]) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>} : (tensor<8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<7xf32>, tensor<1xi32>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_sliced
func.func @rfft_1D_sliced(%arg0: tensor<2x9xf32>) -> tensor<2x8xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<2x8xf32>
  // CHECK: "mhlo.fft"(%[[SLICED]]) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>} : (tensor<2x8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<2x9xf32>, tensor<1xi32>) -> tensor<2x8xcomplex<f32>>
  func.return %0 : tensor<2x8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @irfft_1D
func.func @irfft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<5xf32> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<8xcomplex<f32>>) -> tensor<5xcomplex<f32>>
  // CHECK: "mhlo.fft"(%[[SLICED]]) {fft_length = dense<8> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT>} : (tensor<5xcomplex<f32>>
  %0 = "tf.IRFFT"(%arg0, %fftlength) : (tensor<8xcomplex<f32>>, tensor<1xi32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
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

// -----

// CHECK-LABEL: @transpose_unranked_2d
func.func @transpose_unranked_2d(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "mhlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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

// CHECK-LABEL: func @abs_unranked
func.func @abs_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.abs %arg0 : tensor<*xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @acos
func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  chlo.acos %arg0 : tensor<2xf32>
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @acos_complex
func.func @acos_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: chlo.acos %arg0 : tensor<2xcomplex<f32>>
  %0 = "tf.Acos"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @acos_dynamic
func.func @acos_dynamic(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  chlo.acos %arg0 : tensor<*xf32>
  %0 = "tf.Acos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @tan
// CHECK-SAME: (%[[ARG:.*]]: tensor<2xf32>) -> tensor<2xf32>
func.func @tan(%arg : tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: chlo.tan %[[ARG]] : tensor<2xf32>
  %result = "tf.Tan"(%arg) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %result : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tan_unranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>) -> tensor<*xf32>
func.func @tan_unranked(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: chlo.tan %[[ARG]] : tensor<*xf32>
  %result = "tf.Tan"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %result : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @cast_dynamic_i2f
func.func @cast_dynamic_i2f(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK: mhlo.convert(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @cast_i2f
func.func @cast_i2f(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: mhlo.convert(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @cast_c2f
func.func @cast_c2f(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK: mhlo.convert(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
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

// CHECK-LABEL: func @ceil_unranked
func.func @ceil_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.ceil %arg0 : tensor<*xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @complex_abs
func.func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK:  mhlo.abs(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
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

// CHECK-LABEL: func @cos_dynamic
func.func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.cosine %arg0 : tensor<?xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @cos_unranked
func.func @cos_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.cosine %arg0 : tensor<*xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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

// CHECK-LABEL: func @exp_unranked
func.func @exp_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.exponential %arg0 : tensor<*xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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

// CHECK-LABEL: func @floor_unranked
func.func @floor_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.floor %arg0 : tensor<*xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @invert_op_unranked
func.func @invert_op_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK:  mhlo.not %arg0 : tensor<*xi32>
  %0 = "tf.Invert"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  // CHECK:  mhlo.is_finite(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// -----

// CHECK-LABEL: func @is_finite_dynamic
func.func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK:  mhlo.is_finite(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: func @is_finite_unranked
func.func @is_finite_unranked(%arg0: tensor<*xf32>) -> tensor<*xi1> {
  // CHECK:  mhlo.is_finite(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
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

// CHECK-LABEL: func @log_unranked
func.func @log_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.log %arg0 : tensor<*xf32>
  %0 = "tf.Log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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

// CHECK-LABEL: func @log1p_unranked
func.func @log1p_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.log_plus_one %arg0 : tensor<*xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @not_op_unranked
func.func @not_op_unranked(%arg0: tensor<*xi1>) -> tensor<*xi1> {
  // CHECK:  mhlo.not %arg0 : tensor<*xi1>
  %0 = "tf.LogicalNot"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
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

// CHECK-LABEL: func @neg_unranked
func.func @neg_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.negate %arg0 : tensor<*xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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

// CHECK-LABEL: @sigmoid_unranked
func.func @sigmoid_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: mhlo.logistic
  %0 = "tf.Sigmoid"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}


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
  // CHECK: chlo.broadcast_subtract {{.*}} {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: chlo.broadcast_multiply {{.*}} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @sin
func.func @sin(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.sine %arg0 : tensor<2xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @sin_dynamic
func.func @sin_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.sine %arg0 : tensor<?xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @sin_unranked
func.func @sin_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.sine %arg0 : tensor<*xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @rsqrt
func.func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.rsqrt %arg0 : tensor<2xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @rsqrt_dynamic
func.func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.rsqrt %arg0 : tensor<?xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @rsqrt_unranked
func.func @rsqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.rsqrt %arg0 : tensor<*xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @sqrt
func.func @sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.sqrt %arg0 : tensor<2xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @sqrt_dynamic
func.func @sqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.sqrt %arg0 : tensor<?xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @sqrt_unranked
func.func @sqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.sqrt %arg0 : tensor<*xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @tanh
func.func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  mhlo.tanh %arg0 : tensor<2xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @tanh_dynamic
func.func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  mhlo.tanh %arg0 : tensor<?xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @tanh_unranked
func.func @tanh_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  mhlo.tanh %arg0 : tensor<*xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @bitcast
func.func @bitcast(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @bitcast_dynamic
func.func @bitcast_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @bitcast_unranked
func.func @bitcast_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @bitcast_same_widths
func.func @bitcast_same_widths(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @bitcast_smaller_input_width
func.func @bitcast_smaller_input_width(%arg0: tensor<8xi8>) -> tensor<i64> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<8xi8>) -> tensor<i64>
  %0 = "tf.Bitcast"(%arg0) : (tensor<8xi8>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @bitcast_smaller_output_width
func.func @bitcast_smaller_output_width(%arg0: tensor<2xf32>) -> tensor<2x2xf16> {
  // CHECK:  "mhlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2x2xf16>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2x2xf16>
  func.return %0 : tensor<2x2xf16>
}

// -----

// CHECK-LABEL: reshape
func.func @reshape(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) -> tensor<2x1xf32> {
  // CHECK:  "mhlo.reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  func.return %0 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: reshape_dynamic
func.func @reshape_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK:  "chlo.dynamic_reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: reshape_unranked
// CHECK-SAME: %[[INPUT:.*]]: tensor<*xf32>
// CHECK-SAME: %[[TARGET_SHAPE:.*]]: tensor<2xi32>
func.func @reshape_unranked(%arg0: tensor<*xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK:  "chlo.dynamic_reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: squeeze
func.func @squeeze(%arg0: tensor<1x1x10xf32>) -> tensor<1x10xf32> {
  // CHECK: "mhlo.reshape"
  %0 = "tf.Squeeze"(%arg0) : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
  func.return %0 : tensor<1x10xf32>
}

// -----

// CHECK-LABEL: squeeze_dynamic
func.func @squeeze_dynamic(%arg0: tensor<?x10xf32>) -> tensor<*xf32> {
  // CHECK: "tf.Squeeze"
  %0 = "tf.Squeeze"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: expand_dims
func.func @expand_dims(%arg0: tensor<2xf32>, %axis: tensor<i32>) -> tensor<1x2xf32> {
  // CHECK: "mhlo.reshape"
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<2xf32>, tensor<i32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: expand_dims_dynamic
func.func @expand_dims_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x1x?xf32> {
  %axis = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> (tensor<i32>)

  // CHECK-DAG: %[[SHAPEOF:.+]] = shape.shape_of %arg0
  // CHECK-DAG: %[[CST0:.+]] = arith.constant 0
  // CHECK-DAG: %[[CST1:.+]] = arith.constant 1
  // CHECK-DAG: %[[GETEXTENT0:.+]] = tensor.extract %[[SHAPEOF]][%[[CST0]]]
  // CHECK-DAG: %[[CST1_0:.+]] = arith.constant 1
  // CHECK-DAG: %[[GETEXTENT1:.+]] = tensor.extract %[[SHAPEOF]][%[[CST1_0]]]
  // CHECK-DAG: %[[TOEXTENTS:.+]] = tensor.from_elements %[[GETEXTENT0]], %[[CST1]], %[[GETEXTENT1]]
  // CHECK-DAG: %[[RESHAPE:.+]] = "mhlo.dynamic_reshape"(%arg0, %[[TOEXTENTS]])
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?xf32>, tensor<i32>) -> tensor<?x1x?xf32>

  // CHECK: return %[[RESHAPE]]
  func.return %0 : tensor<?x1x?xf32>
}

// -----

// CHECK-LABEL: expand_dynamic_dims_rank1_axis
func.func @expand_dynamic_dims_rank1_axis(%arg0: tensor<?x?x4xf32>) -> tensor<?x1x?x4xf32> {
  %axis = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>

  // CHECK-DAG: %[[SHAPEOF:.+]] = shape.shape_of %arg0
  // CHECK-DAG: %[[CST0:.+]] = arith.constant 0
  // CHECK-DAG: %[[CST1:.+]] = arith.constant 1
  // CHECK-DAG: %[[GETEXTENT0:.+]] = tensor.extract %[[SHAPEOF]][%[[CST0]]]
  // CHECK-DAG: %[[CST1_0:.+]] = arith.constant 1
  // CHECK-DAG: %[[GETEXTENT1:.+]] = tensor.extract %[[SHAPEOF]][%[[CST1_0]]]
  // CHECK-DAG: %[[CST2:.+]] = arith.constant 2
  // CHECK-DAG: %[[GETEXTENT2:.+]] = tensor.extract %[[SHAPEOF]][%[[CST2]]]
  // CHECK-DAG: %[[TOEXTENTS:.+]] = tensor.from_elements %[[GETEXTENT0]], %[[CST1]], %[[GETEXTENT1]], %[[GETEXTENT2]]
  // CHECK-DAG: %[[RESHAPE:.+]] = "mhlo.dynamic_reshape"(%arg0, %[[TOEXTENTS]])
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<?x?x4xf32>, tensor<1xi32>) -> tensor<?x1x?x4xf32>

  // CHECK: return %[[RESHAPE]]
  func.return %0 : tensor<?x1x?x4xf32>
}

// -----

// CHECK-LABEL: func @sign
// CHECK-SAME: [[ARG:%arg.*]]: tensor<1x2x3x4xf32>
func.func @sign(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: [[SIGN:%.*]] = mhlo.sign [[ARG]]
  // CHECK: return [[SIGN]] : tensor<1x2x3x4xf32>
  %0 = "tf.Sign"(%arg0) : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
  func.return %0 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: func @sign_dynamic
func.func @sign_dynamic(%arg0: tensor<?x2x3x?xf32>) -> tensor<?x2x3x?xf32> {
  // CHECK: [[SIGN:%.*]] = mhlo.sign %arg0 : tensor<?x2x3x?xf32>
  // CHECK: return [[SIGN]] : tensor<?x2x3x?xf32>
  %0 = "tf.Sign"(%arg0) : (tensor<?x2x3x?xf32>) -> (tensor<?x2x3x?xf32>)
  func.return %0 : tensor<?x2x3x?xf32>
}

// -----

// CHECK-LABEL: slice_constant_start
func.func @slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = mhlo.constant dense<1> : tensor<i64>
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"(%arg0, %[[START]])
  // CHECK-DAG-SAME: {slice_sizes = dense<2> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: slice_i32_consts
func.func @slice_i32_consts(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: "mhlo.dynamic_slice"(%arg0, %[[START]]) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i32>) -> tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: slice_constant_start_negative_one_size
func.func @slice_constant_start_negative_one_size(%arg0: tensor<4xi32>) -> tensor<3xi32> {
  // CHECK: %[[START:.*]] = mhlo.constant dense<1> : tensor<i64>
  // CHECK: %[[RESULT:.*]] =  "mhlo.dynamic_slice"(%arg0, %[[START]]) {slice_sizes = dense<3> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<3xi32>
  // CHECK: return %[[RESULT]] : tensor<3xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[-1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: slice_constant_start_dynamic_shape
func.func @slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK-DAG: %[[START1:.*]] = mhlo.constant dense<1> : tensor<i64>
  // CHECK-DAG: %[[START2:.*]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"
  // CHECK-DAG-SAME: (%arg0, %[[START1]], %[[START2]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 4]> : tensor<2xi64>} :
  // CHECK-DAG-SAME: (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %starts = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<?x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: slice_variable_start
func.func @slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[SLICED_START1:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START1:.*]] = "mhlo.reshape"(%[[SLICED_START1]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[SLICED_START2:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<2> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START2:.*]] = "mhlo.reshape"(%[[SLICED_START2]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"(%arg0, %[[RESHAPED_START1]], %[[RESHAPED_START2]]) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: slice_mhlo_sizes
func.func @slice_mhlo_sizes(%arg0: tensor<1x1024x4xf32>, %arg1: tensor<3xi32>) -> tensor<1x512x4xf32> {
  // CHECK-NOT: "tf.Slice"
  %0 = "mhlo.constant"() {value = dense<[1, 512, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Slice"(%arg0, %arg1, %0) : (tensor<1x1024x4xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x512x4xf32>
  func.return %1 : tensor<1x512x4xf32>
}

// -----

// CHECK-LABEL: slice_variable_start_negative_one_size
func.func @slice_variable_start_negative_one_size(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[RESULT:.*]] = "tf.Slice"
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, -1]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: slice_real_dynamic_slice
func.func @slice_real_dynamic_slice(%arg0: tensor<4xi32>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>) -> tensor<*xi32> {
  // CHECK: tensor.extract {{.*}} : tensor<1xi64>
  // CHECK: tensor.extract {{.*}} : tensor<1xi64>
  // CHECK: arith.index_cast {{.*}} : index to i64
  // CHECK: arith.cmpi eq, {{.*}} : i64
  // CHECK: arith.addi {{.*}} : i64
  // CHECK: tensor.dim {{.*}} : tensor<4xi32>
  // CHECK: arith.index_cast {{.*}} : index to i64
  // CHECK: select {{.*}} : i64
  // CHECK: arith.index_cast {{.*}} : i64 to index
  // CHECK: arith.index_cast {{.*}} : i64 to index
  // CHECK: tensor.from_elements {{.*}} : tensor<1xindex>
  // CHECK: tensor.from_elements {{.*}} : tensor<1xindex>
  // CHECK: tensor.from_elements {{.*}} : tensor<1xindex>
  %0 = "tf.Slice"(%arg0, %arg1, %arg2) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

//===----------------------------------------------------------------------===//
// StridedSlice op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: simple_strided_slice
func.func @simple_strided_slice(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: mhlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  func.return %output : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: dynamic_strided_slice
func.func @dynamic_strided_slice(%input: tensor<?x8xf32>) -> tensor<?x2xf32> {
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "tf.StridedSlice"
  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<?x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x2xf32>
  func.return %output : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: strided_slice_negative_indices
func.func @strided_slice_negative_indices(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[-1, -2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-4, -8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, -3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "mhlo.reverse"(%arg0) {dimensions = dense<[0, 1]> : tensor<2xi64>}

  // CHECK: mhlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  func.return %output : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: dynamic_strided_slice_negative_indices
func.func @dynamic_strided_slice_negative_indices(%input: tensor<?x8xf32>) -> tensor<?x2xf32> {
  %begin = "tf.Const"() {value = dense<[-1, -2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-4, -8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, -3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: tf.StridedSlice
  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<?x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x2xf32>
  func.return %output : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: strided_slice_range_clamping
func.func @strided_slice_range_clamping(%input: tensor<4x8xf32>) -> tensor<1x3xf32> {
  %begin = "tf.Const"() {value = dense<[-4, -10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[1, 10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: mhlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 0]>
  // CHECK-DAG-SAME: limit_indices = dense<[1, 8]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<1x3xf32>
  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
  func.return %output : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: strided_slice_empty
func.func @strided_slice_empty(%input: tensor<4xf32>) -> tensor<0xf32> {
  %begin = "tf.Const"() {value = dense<[-4]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %end = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %strides = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: mhlo.constant dense<> : tensor<0xf32>
  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xf32>
  func.return %output : tensor<0xf32>
}

// -----

// CHECK-LABEL: strided_slice_begin_end_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<4x128x1024xf32>
func.func @strided_slice_begin_end_mask(%input: tensor<4x128x1024xf32>) {

  // For StridedSlice
  // Dim #:        0,   1,    2
  // Input shape: [4, 128, 1024]
  // Begin:        1,   4,   -3
  // End:          8,  65,   42
  // Stride:       1,   4,   -1
  // Begin mask:   0,   0,    1  (= 1)
  // End mask:     1,   0,    0  (= 4)

  // So result shape:
  // Dim #0: begin mask (1) -> begin = 0; end 8 canonicalized to 4: so 4
  // Dim #1: 4 to 65 stride 4: so 16
  // Dim #2: begin -3 + 1024 = 1021; end mask (1) -> end = -1: so 1022
  // result shape: [4, 16, 1022]

  %begin = "tf.Const"() {value = dense<[1, 4, -3]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %end = "tf.Const"() {value = dense<[8, 65, 42]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %strides = "tf.Const"() {value = dense<[1, 4, -1]> : tensor<3xi32>} : () -> (tensor<3xi32>)

  // CHECK: %[[REVERSE:.*]] = "mhlo.reverse"(%[[INPUT]])

  // CHECK: %[[SLICE:.*]] = "mhlo.slice"(%[[REVERSE]])
  // CHECK-DAG-SAME: limit_indices = dense<[4, 65, 1024]>
  // CHECK-DAG-SAME: start_indices = dense<[0, 4, 2]>
  // CHECK-DAG-SAME: strides = dense<[1, 4, 1]>
  // CHECK-SAME: -> tensor<4x16x1022xf32>

  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 1, end_mask = 4} : (tensor<4x128x1024xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x16x1022xf32>

  // CHECK: "mhlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<4x16x1022xf32>

  func.return
}

// -----

// CHECK-LABEL: strided_slice_shrink_axis_mask
// CHECK-SAME: %[[INPUT:.+]]: tensor<4x128x1024xf32>
func.func @strided_slice_shrink_axis_mask(%input: tensor<4x128x1024xf32>) {

  // For StridedSlice
  // Dim #:            0,   1,    2
  // Input shape:     [4, 128, 1024]
  // Begin:            1,   4,   -3
  // End:              8,  65,   42
  // Stride:           1,   4,   -1
  // Begin mask:       1,   0,    0  (= 1)
  // End mask:         0,   0,    1  (= 4)
  // Shrink axis mask: 1,   0,    1  (= 5)

  // So result shape:
  // Dim #0: shrink axis, take value at [1]
  // Dim #1: 4 to 65 stride 4: so 16
  // Dim #2: shrink axis, take value at [-3]
  // result shape: [16]

  // As output shape of StridedSlice differs, a reshape will follow.

  %begin = "tf.Const"() {value = dense<[1, 4, -3]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %end = "tf.Const"() {value = dense<[8, 65, 42]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %strides = "tf.Const"() {value = dense<[1, 4, -1]> : tensor<3xi32>} : () -> (tensor<3xi32>)

  // CHECK: %[[SLICE:.*]] = "mhlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[1, 65, 1022]>
  // CHECK-DAG-SAME: start_indices = dense<[0, 4, 1021]>
  // CHECK-DAG-SAME: strides = dense<[1, 4, 1]>
  // CHECK-SAME: -> tensor<1x16x1xf32>

  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 1, end_mask = 4, shrink_axis_mask = 5} : (tensor<4x128x1024xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<16xf32>

  // CHECK: "mhlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<16xf32>

  func.return
}

// -----

// CHECK-LABEL: strided_slice_ellipsis_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<2x4x8x16x32x64xf32>
func.func @strided_slice_ellipsis_mask(%input: tensor<2x4x8x16x32x64xf32>) {
  // For StridedSlice input[1, ..., 8:, :10, 2:6:2]
  // The ellipsis mask is applied to dim #1, #2, i.e, we get canonicalized
  // slice input[1, :, :, 8:, :10, 2:6:2]

  // The start, limit indices and strides attributes of mhlo.slice would
  // reflect the canonicalized slice.
  // As output shape of StridedSlice differs, a reshape will follow.

  %begin = "tf.Const"() {value = dense<[1, 0, 8, 1, 2]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %end = "tf.Const"() {value = dense<[2, 0, 10, 10, 6]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %strides = "tf.Const"() {value = dense<[1, 1, 1, 1, 2]> : tensor<5xi32>} : () -> (tensor<5xi32>)

  // CHECK: %[[SLICE:.*]] = "mhlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4, 8, 16, 10, 6]> : tensor<6xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: strides = dense<[1, 1, 1, 1, 1, 2]> : tensoe<6xi64>
  // CHECK-SAME: -> tensor<1x4x8x8x10x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 8, end_mask = 4, shrink_axis_mask = 1, ellipsis_mask = 2} : (tensor<2x4x8x16x32x64xf32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<4x8x8x10x2xf32>

  // CHECK: "mhlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<4x8x8x10x2xf32>

  func.return
}

// -----

// CHECK-LABEL: strided_slice_new_axis_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<2x4x8x16x32x64xf32>
func.func @strided_slice_new_axis_mask(%input: tensor<2x4x8x16x32x64xf32>) {
  // For StridedSlice input[1, tf.new_axis, ..., 8:, :10, 2:6:2, tf.new_axis]
  // New axis mask is at index 1 and 6 of sparse spec, so
  // new_axis_mask = 2^1 + 2^6 = 66
  // The ellipsis mask is applied to dim #1, #2 of input i.e, we get
  // canonicalized slice input[1, :, :, 8:, :10, 2:6:2]
  // This is then reshaped to add the new axes.

  // The start, limit indices and strides attributes of mhlo.slice would
  // reflect the canonicalized slice.
  // As output shape of StridedSlice differs, a reshape will follow to reflect
  // new axes added.

  %begin = "tf.Const"() {value = dense<[1, 0, 0, 8, 1, 2, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %end = "tf.Const"() {value = dense<[2, 0, 0, 10, 10, 6, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %strides = "tf.Const"() {value = dense<[1, 1, 1, 1, 1, 2, 1]> : tensor<7xi32>} : () -> (tensor<7xi32>)

  // CHECK: %[[SLICE:.*]] = "mhlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4, 8, 16, 10, 6]> : tensor<6xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: strides = dense<[1, 1, 1, 1, 1, 2]> : tensoe<6xi64>
  // CHECK-SAME: -> tensor<1x4x8x8x10x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 16, end_mask = 8, shrink_axis_mask = 1, ellipsis_mask = 4, new_axis_mask = 66} : (tensor<2x4x8x16x32x64xf32>, tensor<7xi32>, tensor<7xi32>, tensor<7xi32>) -> tensor<1x4x8x8x10x2x1xf32>

  // CHECK: "mhlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<1x4x8x8x10x2x1xf32>

  func.return
}

// -----

// CHECK-LABEL: strided_slice_implicit_ellipsis_mask(
// CHECK-SAME: [[INPUT:%.*]]: tensor<10x16x2xf32>
func.func @strided_slice_implicit_ellipsis_mask(%input: tensor<10x16x2xf32>) -> tensor<2x16x2xf32> {
  // StridedSlice gets input[8:10], which is same as input[8:10, ...]
  // The start_indices, limit_indices, and strides attribute of mhlo.slice
  // reflect the canonicalized slice.
  %begin = "tf.Const"() {value = dense<8> : tensor<1xi32>} : () -> tensor<1xi32>
  %end = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: [[SLICE:%.*]] = "mhlo.slice"([[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[10, 16, 2]> : tensor<3xi64>
  // CHECK-DAG-SAME: start_indices = dense<[8, 0, 0]> : tensor<3xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<3xi64>
  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[SLICE]]) : (tensor<2x16x2xf32>) -> tensor<2x16x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = f32} : (tensor<10x16x2xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2x16x2xf32>
  // CHECK: return [[RESHAPE]] : tensor<2x16x2xf32>
  func.return %0 : tensor<2x16x2xf32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end
func.func @strided_slice_nonconstant_begin_end(%arg0: tensor<i32>, %arg1: tensor<32x1x97xi32>) -> (tensor<1x97xi32>) {
  // In this case, the `begin` and `end` inputs are unknown at compile time --
  // so the StridedSlice needs to slice these vectors and use that as input to
  // an HLO dynamic slice.
  %begin = "tf.Pack"(%arg0) {N = 1 : i64, T = i32, axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.AddV2"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %end = "tf.Pack"(%2) {N = 1 : i64, T = i32, axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  // CHECK: %[[A:.*]] = "mhlo.reshape"(%arg0) : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[BEGIN:.*]] = "mhlo.concatenate"(%[[A]])
  // CHECK-DAG-SAME: {dimension = 0 : i64} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[INDEX:.*]] = "mhlo.slice"(%[[BEGIN]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INDEX2:.*]] = "mhlo.reshape"(%[[INDEX]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[CMP:.*]] = chlo.broadcast_compare %[[INDEX2]], %[[ZERO]]
  // CHECK-DAG-SAME: {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DIM:.*]] = mhlo.constant dense<32> : tensor<i32>
  // CHECK-NEXT: %[[WRAP:.*]] = chlo.broadcast_add %[[DIM]], %[[INDEX2]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[INDEX3:.*]] = "mhlo.select"(%[[CMP]], %[[WRAP]], %[[INDEX2]]) :
  // CHECK-DAG-SAME: (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICED:.*]] = "mhlo.dynamic_slice"
  // CHECK-DAG-SAME: (%arg1, %[[INDEX3]], %[[ZERO]], %[[ZERO]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 1, 97]> : tensor<3xi64>} :
  // CHECK-DAG-SAME: (tensor<32x1x97xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x97xi32>
  // CHECK-NEXT: %[[FINAL:.*]] = "mhlo.reshape"(%[[SLICED]]) : (tensor<1x1x97xi32>) -> tensor<1x97xi32>
  %result = "tf.StridedSlice"(%arg1, %begin, %end, %1) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  // CHECK-NEXT: return %[[FINAL]] : tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_with_start_end_mask
// CHECK-SAME: (%[[INPUT:.*]]: tensor<32x1x97xi32>, %[[BEGIN:.*]]: tensor<3xi32>, %[[END:.*]]: tensor<3xi32>)
func.func @strided_slice_nonconstant_begin_end_with_start_end_mask(%input: tensor<32x1x97xi32>, %begin: tensor<3xi32>, %end: tensor<3xi32>) -> (tensor<1x97xi32>) {
  %strides = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>

  // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[INDEX:.*]] = "mhlo.slice"(%[[BEGIN]])
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>
  // CHECK-DAG-SAME: limit_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>
  // CHECK-NEXT: %[[INDEX2:.*]] = "mhlo.reshape"(%[[INDEX]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[CMP:.*]] = chlo.broadcast_compare %[[INDEX2]], %[[ZERO]]
  // CHECK-DAG-SAME: {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DIM:.*]] = mhlo.constant dense<32> : tensor<i32>
  // CHECK-NEXT: %[[WRAP:.*]] = chlo.broadcast_add %[[DIM]], %[[INDEX2]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[INDEX3:.*]] = "mhlo.select"(%[[CMP]], %[[WRAP]], %[[INDEX2]]) :
  // CHECK-DAG-SAME: (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICED:.*]] = "mhlo.dynamic_slice"
  // CHECK-DAG-SAME: (%arg1, %[[INDEX3]], %[[ZERO]], %[[ZERO]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 1, 97]> : tensor<3xi64>} :
  // CHECK-DAG-SAME: (tensor<32x1x97xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x97xi32>
  // CHECK-NEXT: %[[FINAL:.*]] = "mhlo.reshape"(%[[SLICED]]) : (tensor<1x1x97xi32>) -> tensor<1x97xi32>
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 6 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_stride_1
func.func @strided_slice_nonconstant_begin_end_stride_1(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>, %strides: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // Dynamic stride: when `begin` and `end` inputs are unknown at compile time,
  // `strides` must be known.
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 4 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_stride_2
func.func @strided_slice_nonconstant_begin_end_stride_2(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // Invalid stride (not equal to 1): when `begin` and `end` inputs are unknown
  // at compile time, `strides` must be known to have all 1 values.
  %strides = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 4 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_invalid_elem_count
func.func @strided_slice_nonconstant_begin_end_invalid_elem_count(%input: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>) -> tensor<6x10xf32> {
  %strides = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  // When begin/end are dynamic, the number of output elements must be equal to
  // the number of input elements sliced.
  // CHECK: tf.StridedSlice
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<6x10xf32>
  func.return %0 : tensor<6x10xf32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_new_axis_mask
func.func @strided_slice_nonconstant_begin_end_and_new_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // New axis mask: When `begin` and `end` inputs are unknown at compile time,
  // we can't support a new_axis mask.
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 15 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_ellipsis_mask
func.func @strided_slice_nonconstant_begin_end_and_ellipsis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This ellipsis mask is not supported because it does not refer to the last
  // dimension.
  // [0, 1, 0] = 2
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 2 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_valid_ellipsis_mask
func.func @strided_slice_nonconstant_begin_end_and_valid_ellipsis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This ellipsis mask is supported because it refers to the last dimension.
  // [1, 0, 0] = 4
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: mhlo.dynamic_slice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 4 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_valid_shrink_axis_mask
func.func @strided_slice_nonconstant_begin_end_and_valid_shrink_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This shrink_axis mask is supported because it refers to a major dimension.
  // [1, 1, 1] = 7
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: mhlo.dynamic_slice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 7 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}

// -----

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_invalid_shrink_axis_mask
func.func @strided_slice_nonconstant_begin_end_and_invalid_shrink_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This shrink_axis mask is unsupported because it does not refer to a major
  // dimension.
  // [0, 1, 0] = 2
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  func.return %result : tensor<1x97xi32>
}


//===----------------------------------------------------------------------===//
// Reduction op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @mean
func.func @mean(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[MEAN:.*]] = chlo.broadcast_divide %[[REDUCED]], %{{.*}} {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = mhlo.convert(%[[MEAN]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @mean_scalar_dim
func.func @mean_scalar_dim(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // Verify that tf.Mean op with scalar attributes are lowered successfully.

  // CHECK-NOT: tf.Mean
  %dimension = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<i64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @mean_dynamic
func.func @mean_dynamic(%arg0: tensor<?x?xf16>) -> tensor<?x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?x?xf16> -> tensor<2xindex>
  // CHECK-DAG: %[[C1_1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C1_2:.*]] = arith.constant 1 : index
  // CHECK: %[[REDUCED_DIM:.*]] = tensor.extract %[[SHAPE0]][%[[C1_2]]] : tensor<2xindex>
  // CHECK: %[[MUL:.*]] = arith.muli %[[C1_1]], %[[REDUCED_DIM]] : index
  // CHECK: %[[INDEX_CAST:.*]] = arith.index_cast %[[MUL]] : index to i64
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[INDEX_CAST]] : tensor<i64>
  // CHECK: %[[CONVERT:.*]] = mhlo.convert(%[[TENSOR]]) : (tensor<i64>) -> tensor<f32>
  // CHECK: %[[MEAN:.*]] = chlo.broadcast_divide %[[REDUCED]], %[[CONVERT]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK: %[[MEAN_CONVERTED:.*]] = mhlo.convert(%[[MEAN]]) : (tensor<?xf32>) -> tensor<?xf16>
  // CHECK: %[[SHAPE1:.*]] = shape.shape_of %[[MEAN_CONVERTED]] : tensor<?xf16> -> tensor<1xindex>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[UNREDUCED_DIM:.*]] = tensor.extract %[[SHAPE1]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[RESULT_SHAPE:.*]] = tensor.from_elements %[[UNREDUCED_DIM]], %[[C1]] : tensor<2xindex>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_reshape"(%[[MEAN_CONVERTED]], %[[RESULT_SHAPE]]) : (tensor<?xf16>, tensor<2xindex>) -> tensor<?x1xf16>
  // CHECK: return %[[RESULT]] : tensor<?x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<?x?xf16>, tensor<1xi64>) -> tensor<?x1xf16>
  func.return %0 : tensor<?x1xf16>
}

// -----

// CHECK-LABEL: func @sum
func.func @sum(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = mhlo.convert(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Sum"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @sum_dynamic
func.func @sum_dynamic(%arg0: tensor<4x?xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = mhlo.convert(%arg0) : (tensor<4x?xf16>) -> tensor<4x?xf32>
    // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<-0.000000e+00> : tensor<f32>
    // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.add across dimensions = [1] : (tensor<4x?xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: %[[CAST_BACK:.*]] = mhlo.convert(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Sum"(%arg0, %dimension) { keep_dims = true }: (tensor<4x?xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @max
func.func @max(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert %arg0 : tensor<4x8xf16>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<0xFC00> : tensor<f16>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.maximum across dimensions = [1] : (tensor<4x8xf16>, tensor<f16>) -> tensor<4xf16>
  // CHECK: %[[CAST_BACK:.*]] = mhlo.convert %[[REDUCED]] : tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @max_qint
// Regression test to ensure we don't crash getting the initial value for
// tf.Max when using quantized integer types.
func.func @max_qint(%arg0: tensor<4x8x!tf_type.qint8>) -> tensor<4x1x!tf_type.qint8> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8x!tf_type.qint8>, tensor<1xi64>) -> tensor<4x1x!tf_type.qint8>
  func.return %0 : tensor<4x1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @max_dynamic
func.func @max_dynamic(%arg0: tensor<4x?xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = mhlo.convert %arg0 : tensor<4x?xf16>
    // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<0xFC00> : tensor<f16>
    // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.maximum across dimensions = [1] : (tensor<4x?xf16>, tensor<f16>) -> tensor<4xf16>
    // CHECK: %[[CAST_BACK:.*]] = mhlo.convert %[[REDUCED]] : tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x?xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @min
func.func @min(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert %arg0 : tensor<4x8xf16>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<0x7C00> : tensor<f16>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.minimum across dimensions = [1] : (tensor<4x8xf16>, tensor<f16>) -> tensor<4xf16>
  // CHECK: %[[CAST_BACK:.*]] = mhlo.convert %[[REDUCED]] : tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Min"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @min_qint
// Regression test to ensure we don't crash getting the initial value for
// tf.Min when using quantized integer types.
func.func @min_qint(%arg0: tensor<4x8x!tf_type.qint8>) -> tensor<4x1x!tf_type.qint8> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Min"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8x!tf_type.qint8>, tensor<1xi64>) -> tensor<4x1x!tf_type.qint8>
  func.return %0 : tensor<4x1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @prod
func.func @prod(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = mhlo.convert(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = mhlo.reduce(%[[CAST]] init: %[[INITIAL]]) applies mhlo.multiply across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = mhlo.convert(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Prod"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  func.return %0 : tensor<4x1xf16>
}

// -----

// CHECK-LABEL: func @prod_qint
// Regression test to ensure we don't crash getting the initial value for
// tf.Prod when using quantized integer types.
func.func @prod_qint(%arg0: tensor<4x8x!tf_type.qint8>) -> tensor<4x1x!tf_type.qint8> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Prod"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8x!tf_type.qint8>, tensor<1xi64>) -> tensor<4x1x!tf_type.qint8>
  func.return %0 : tensor<4x1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: @all
func.func @all(%input: tensor<4x8xi1>) -> tensor<4xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<true> : tensor<i1>
  // CHECK: mhlo.reduce(%{{.*}} init: %[[INIT]]) applies mhlo.and across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  %0 = "tf.All"(%input, %dims) : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: @all_keep_dim
func.func @all_keep_dim(%input: tensor<4x8xi1>) -> tensor<4x1xi1> {
  // CHECK: "mhlo.reshape"(%{{.*}}) : (tensor<4xi1>) -> tensor<4x1xi1>
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.All"(%input, %dims) {keep_dims = true} : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  func.return %0 : tensor<4x1xi1>
}

// -----

// CHECK-LABEL: @all_dynamic
func.func @all_dynamic(%input: tensor<4x?xi1>) -> tensor<4x1xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[ARG:.*]] = mhlo.convert %{{.*}} : tensor<4x?xi1>
  // CHECK: mhlo.reduce(%[[ARG]]
  %0 = "tf.All"(%input, %dims) {keep_dims = true} : (tensor<4x?xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  func.return %0 : tensor<4x1xi1>
}

// -----

// CHECK-LABEL: @any
func.func @any(%input: tensor<4x8xi1>) -> tensor<4xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK: mhlo.reduce(%{{.*}} init: %[[INIT]]) applies mhlo.or across dimensions = [1] : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  %0 = "tf.Any"(%input, %dims) : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: @any_keep_dim
func.func @any_keep_dim(%input: tensor<4x8xi1>) -> tensor<4x1xi1> {
  // CHECK: "mhlo.reshape"(%{{.*}}) : (tensor<4xi1>) -> tensor<4x1xi1>
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Any"(%input, %dims) {keep_dims = true} : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  func.return %0 : tensor<4x1xi1>
}

// -----

// CHECK-LABEL: @any_dynamic
func.func @any_dynamic(%input: tensor<4x?xi1>) -> tensor<4x1xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[ARG:.*]] = mhlo.convert %{{.*}} : tensor<4x?xi1>
  // CHECK: mhlo.reduce(%[[ARG]]
  %0 = "tf.Any"(%input, %dims) {keep_dims = true} : (tensor<4x?xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  func.return %0 : tensor<4x1xi1>
}

//===----------------------------------------------------------------------===//
// Tile op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @tile_by_reshape
func.func @tile_by_reshape(%arg0: tensor<4x8xf32>) -> tensor<28x24xf32> {
  // CHECK: %[[BROADCASTED:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<4x8xf32>) -> tensor<7x4x3x8xf32>
  // CHECK: %[[RESULT:.*]] = "mhlo.reshape"(%[[BROADCASTED]]) : (tensor<7x4x3x8xf32>) -> tensor<28x24xf32>
  // CHECK: return %[[RESULT]] : tensor<28x24xf32>
  %multiples = "tf.Const"() { value = dense<[7,3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %0 = "tf.Tile"(%arg0, %multiples) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<28x24xf32>
  func.return %0 : tensor<28x24xf32>
}

// -----

// CHECK-LABEL: func @tile_just_broadcast
func.func @tile_just_broadcast(%arg0: tensor<1x1xf32>) -> tensor<7x3xf32> {
  // CHECK: %[[RESULT:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<7x3xf32>
  // CHECK: return %[[RESULT]] : tensor<7x3xf32>
  %multiples = "tf.Const"() { value = dense<[7,3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %0 = "tf.Tile"(%arg0, %multiples) : (tensor<1x1xf32>, tensor<2xi64>) -> tensor<7x3xf32>
  func.return %0 : tensor<7x3xf32>
}

// -----

// CHECK-LABEL: func @tile_dynamic_shape
func.func @tile_dynamic_shape(%arg0: tensor<?x8xf32>) -> tensor<?x24xf32> {
  %multiples = "tf.Const"() { value = dense<[7,3]> : tensor<2xi32> } : () -> tensor<2xi32>
  // CHECK: tensor.dim {{.*}} : tensor<?x8xf32>
  // CHECK: tensor.from_elements  {{.*}} : tensor<4xindex>
  // CHECK: "mhlo.dynamic_broadcast_in_dim"({{.*}}) {broadcast_dimensions = dense<[1, 3]> : tensor<2xi64>} : (tensor<?x8xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  // CHECK: muli {{.*}} : index
  // CHECK: tensor.from_elements {{.*}} : tensor<2xindex>
  // CHECK: "mhlo.dynamic_reshape"({{.*}}) : (tensor<?x?x?x?xf32>, tensor<2xindex>) -> tensor<?x24xf32>
  %0 = "tf.Tile"(%arg0, %multiples) : (tensor<?x8xf32>, tensor<2xi32>) -> tensor<?x24xf32>
  func.return %0 : tensor<?x24xf32>
}

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @argmax_i64_input_i32_output_axis_0
func.func @argmax_i64_input_i32_output_axis_0(%arg0: tensor<3x7xi64>) -> tensor<7xi32> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<-9223372036854775808> : tensor<i64>
  // CHECK-NEXT: %[[INDEX_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x7xi64> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<3x7xi32>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: reducer(%[[ARG1:.*]]: tensor<i64>, %[[ARG3:.*]]: tensor<i64>) (%[[ARG2:.*]]: tensor<i32>, %[[ARG4:.*]]: tensor<i32>)
  // CHECK: %[[COMPARE:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG3]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:  %[[RESULT1:.*]] = "mhlo.select"(%[[COMPARE]], %[[ARG1]], %[[ARG3]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK: %[[COMPARE_EQ:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG3]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:  %[[MIN:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]]
  // CHECK:  %[[RESULT2:.*]] = "mhlo.select"(%[[COMPARE]], %[[ARG2]], %[[ARG4]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK:  %[[RESULT3:.*]] = "mhlo.select"(%[[COMPARE_EQ]], %[[MIN]], %[[RESULT2]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: "mhlo.return"(%[[RESULT1]], %[[RESULT3]]) : (tensor<i64>, tensor<i32>) -> ()
  // CHECK: return %[[REDUCE]]#1 : tensor<7xi32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x7xi64>, tensor<i32>) -> tensor<7xi32>
  func.return %0 : tensor<7xi32>
}

// -----

// CHECK-LABEL: func @argmax_f32_input_i64_output_axis_1
func.func @argmax_f32_input_i64_output_axis_1(%arg0: tensor<3x7xf32>) -> tensor<3xi64> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK-NEXT: %[[INDEX_INIT:.*]] = mhlo.constant  dense<0> : tensor<i64>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x7xf32> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<3x7xi64>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<3xi64>
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x7xf32>, tensor<i32>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

// CHECK-LABEL: func @argmax_i1_input_i64_output_axis_1
func.func @argmax_i1_input_i64_output_axis_1(%arg0: tensor<3x7xi1>) -> tensor<3xi64> {
  // CHECK-DAG: %[[INIT:.*]] = mhlo.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[INDEX_INIT:.*]] = mhlo.constant  dense<0> : tensor<i64>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x7xi1> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<3x7xi64>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<3xi64>
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x7xi1>, tensor<i32>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

// CHECK-LABEL: func @argmax_dynamic_shape_input_output
func.func @argmax_dynamic_shape_input_output(%arg0: tensor<3x?xi32>) -> tensor<?xi32> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<-2147483648> : tensor<i32>
  // CHECK-NEXT: %[[INDEX_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x?xi32> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<3x?xi32>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<?xi32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x?xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @argmax_dynamic_shape_input
func.func @argmax_dynamic_shape_input(%arg0: tensor<3x?xi32>) -> tensor<3xi32> {
  // CHECK-DAG: %[[INIT:.*]] = mhlo.constant dense<-2147483648> : tensor<i32>
  // CHECK-DAG: %[[INDEX_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x?xi32> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<3x?xi32>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<3xi32>
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x?xi32>, tensor<i32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: func @argmin_i64_input_i32_output_axis_0
func.func @argmin_i64_input_i32_output_axis_0(%arg0: tensor<3x7xi64>) -> tensor<7xi32> {
  // CHECK: %[[INIT:.*]] = mhlo.constant dense<9223372036854775807> : tensor<i64>
  // CHECK-NEXT: %[[INDEX_INIT:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<3x7xi64> -> tensor<2xindex>
  // CHECK: %[[INDEX:.*]] = "mhlo.dynamic_iota"(%[[SHAPE]]) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<3x7xi32>
  // CHECK: %[[REDUCE:.*]]:2 = mhlo.reduce(%arg0 init: %[[INIT]]), (%[[INDEX]] init: %[[INDEX_INIT]])
  // CHECK: reducer(%[[ARG1:.*]]: tensor<i64>, %[[ARG3:.*]]: tensor<i64>) (%[[ARG2:.*]]: tensor<i32>, %[[ARG4:.*]]: tensor<i32>)
  // CHECK: %[[COMPARE:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG3]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LE>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:  %[[RESULT1:.*]] = "mhlo.select"(%[[COMPARE]], %[[ARG1]], %[[ARG3]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK: %[[COMPARE_EQ:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG3]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:  %[[MIN:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]]
  // CHECK:  %[[RESULT2:.*]] = "mhlo.select"(%[[COMPARE]], %[[ARG2]], %[[ARG4]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK:  %[[RESULT3:.*]] = "mhlo.select"(%[[COMPARE_EQ]], %[[MIN]], %[[RESULT2]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: "mhlo.return"(%[[RESULT1]], %[[RESULT3]]) : (tensor<i64>, tensor<i32>) -> ()
  // CHECK: return %[[REDUCE]]#1 : tensor<7xi32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMin"(%arg0, %axis) : (tensor<3x7xi64>, tensor<i32>) -> tensor<7xi32>
  func.return %0 : tensor<7xi32>
}

//===----------------------------------------------------------------------===//
// Random op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @rng_uniform
func.func @rng_uniform(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[CONV:.*]] = mhlo.convert(%arg0) : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK: %[[F32:.*]] = "mhlo.rng"(%[[ZERO]], %[[ONE]], %[[CONV]]) {{.*UNIFORM.*}} -> tensor<12x?x64xf32>
  %0 = "tf.RandomUniform"(%arg0) : (tensor<3xi32>) -> tensor<12x?x64xf32>
  // CHECK: return %[[F32]]
  func.return %0 : tensor<12x?x64xf32>
}

// -----

// CHECK-LABEL: func @rng_std_normal
func.func @rng_std_normal(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[CONV:.*]] = mhlo.convert(%arg0) : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK: %[[F32:.*]] = "mhlo.rng"(%[[ZERO]], %[[ONE]], %[[CONV]]) {{.*NORMAL.*}} -> tensor<12x?x64xf32>
  %0 = "tf.RandomStandardNormal"(%arg0) : (tensor<3xi32>) -> tensor<12x?x64xf32>
  // CHECK: return %[[F32]]
  func.return %0 : tensor<12x?x64xf32>
}

//===----------------------------------------------------------------------===//
// Range op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @range
// CHECK-SAME: [[START:%.*]]: tensor<f32>, [[DELTA:%.*]]: tensor<f32>
func.func @range(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<5xf32> {
  %1 = "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "range/limit", value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: [[IOTA:%.*]] = "mhlo.iota"
  // CHECK-DAG: [[MUL:%.*]] = chlo.broadcast_multiply [[IOTA]], [[DELTA]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK: chlo.broadcast_add [[MUL]], [[START]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  %3 = "tf.Range"(%arg0, %1, %arg1) {Tidx = "tfdtype$DT_FLOAT", device = "", name = "range"} : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
  func.return %3 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @range_dynamic
// CHECK-SAME: [[START:%.*]]: tensor<f32>, [[DELTA:%.*]]: tensor<f32>
func.func @range_dynamic(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<?xf32> {
  // CHECK-DAG: [[SUB:%.+]] = mhlo.subtract %arg1, %arg0
  // CHECK-DAG: [[ABS1:%.+]] = mhlo.abs [[SUB]]
  // CHECK-DAG: [[CONVERT1:%.+]] = mhlo.convert [[ABS1]]
  // CHECK-DAG: [[CONVERT2:%.+]] = mhlo.convert %arg2
  // CHECK-DAG: [[DIV:%.+]] = mhlo.divide [[CONVERT1]], [[CONVERT2]]
  // CHECK-DAG: [[CEIL:%.+]] = mhlo.ceil [[DIV]]
  // CHECK-DAG: [[CONVERT3:%.+]] = mhlo.convert([[CEIL]])
  // CHECK-DAG: [[RESHAPE:%.+]] = "mhlo.reshape"([[CONVERT3]])
  // CHECK-DAG: [[IOTA:%.+]] = "mhlo.dynamic_iota"([[RESHAPE]]) {iota_dimension = 0 : i64}
  // CHECK-DAG: [[CONVERT3:%.+]] = mhlo.convert %arg0
  // CHECK-DAG: [[CONVERT4:%.+]] = mhlo.convert %arg2
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[IOTA]], [[CONVERT4]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add [[MUL]], [[CONVERT3]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  %2 = "tf.Range"(%arg0, %arg1, %arg2) {Tidx = "tfdtype$DT_FLOAT", device = "", name = "range"} : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>

  // CHECK: return [[ADD]]
  func.return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @range_int_dynamic
// CHECK-SAME: [[START:%.*]]: tensor<i32>, [[DELTA:%.*]]: tensor<i32>
func.func @range_int_dynamic(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
  // CHECK-DAG: [[SUB:%.+]] = mhlo.subtract %arg1, %arg0
  // CHECK-DAG: [[ABS1:%.+]] = mhlo.abs [[SUB]]
  // CHECK-DAG: [[CONVERT1:%.+]] = mhlo.convert([[ABS1]])
  // CHECK-DAG: [[CONVERT2:%.+]] = mhlo.convert(%arg2)
  // CHECK-DAG: [[DIV:%.+]] = mhlo.divide [[CONVERT1]], [[CONVERT2]]
  // CHECK-DAG: [[CEIL:%.+]] = mhlo.ceil [[DIV]]
  // CHECK-DAG: [[CONVERT3:%.+]] = mhlo.convert([[CEIL]])
  // CHECK-DAG: [[RESHAPE:%.+]] = "mhlo.reshape"([[CONVERT3]])
  // CHECK-DAG: [[IOTA:%.+]] = "mhlo.dynamic_iota"([[RESHAPE]]) {iota_dimension = 0 : i64}
  // CHECK-DAG: [[CONVERT3:%.+]] = mhlo.convert %arg0
  // CHECK-DAG: [[CONVERT4:%.+]] = mhlo.convert %arg2
  // CHECK-DAG: [[MUL:%.+]] = chlo.broadcast_multiply [[IOTA]], [[CONVERT4]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[ADD:%.+]] = chlo.broadcast_add [[MUL]], [[CONVERT3]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  %2 = "tf.Range"(%arg0, %arg1, %arg2) {Tidx = "tfdtype$DT_FLOAT", device = "", name = "range"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>

  // CHECK: return [[ADD]]
  func.return %2 : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @linspace_static
// CHECK-SAME: [[START:%.*]]: tensor<f32>, [[STOP:%.*]]: tensor<f32>
func.func @linspace_static(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // CHECK-DAG: [[NUM:%.*]] = mhlo.constant dense<4>
  // CHECK-DAG: [[NUM_F32:%.*]] = mhlo.convert([[NUM]])
  // CHECK-DAG: [[ONE:%.*]] = mhlo.constant dense<1.000000e+00>
  // CHECK-DAG: [[STEP_DENOMINATOR:%.*]] = chlo.broadcast_subtract [[NUM_F32]], [[ONE]]
  // CHECK-DAG: [[STEP_NUMERATOR:%.*]] = chlo.broadcast_subtract [[STOP]], [[START]]
  // CHECK-DAG: [[STEP:%.*]] = chlo.broadcast_divide [[STEP_NUMERATOR]], [[STEP_DENOMINATOR]]
  // CHECK-DAG: [[IOTA:%.*]] = "mhlo.iota"() {iota_dimension = 0 : i64}
  // CHECK-DAG: [[MUL:%.*]] = chlo.broadcast_multiply [[IOTA]], [[STEP]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK-DAG: [[LINSPACE:%.*]] = chlo.broadcast_add [[MUL]], [[START]] {broadcast_dimensions = dense<> : tensor<0xi64>}
  // CHECK: return [[LINSPACE]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.LinSpace"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @linspace_dynamic
func.func @linspace_dynamic(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<?xf32> {
  // CHECK: "tf.LinSpace"
  %0 = "tf.LinSpace"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @linspace_invalid_num
func.func @linspace_invalid_num(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK: mhlo.constant dense<> : tensor<0xi32>
  // CHECK: "tf.LinSpace"
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %1 = "tf.LinSpace"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<0xi32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// LegacyCall op legalizations.
//===----------------------------------------------------------------------===//

// -----

func.func @identity_func(%arg0: tensor<10x2xf32>) -> tensor<10x2xf32> {
  func.return %arg0: tensor<10x2xf32>
}

// CHECK-LABEL: testSimpleLegacyCallOp
func.func @testSimpleLegacyCallOp(%arg0: tensor<10x2xf32>) -> tensor<10x2xf32> {
  // CHECK: %[[RESULT:.*]] = call @identity_func(%arg0) : (tensor<10x2xf32>) -> tensor<10x2xf32>
  %0 = "tf.LegacyCall"(%arg0) {f = @identity_func} : (tensor<10x2xf32>) -> tensor<10x2xf32>
  // CHECK: return %[[RESULT]]
  func.return %0: tensor<10x2xf32>
}

// -----

func.func @select_first(%arg0: tensor<10x2xf32>, %arg1: tensor<10x2xf32>) -> tensor<10x2xf32> {
  func.return %arg0: tensor<10x2xf32>
}

// CHECK-LABEL: testMultiInputLegacyCallOp
func.func @testMultiInputLegacyCallOp(%arg0: tensor<10x2xf32>, %arg1: tensor<10x2xf32>) -> tensor<10x2xf32> {
  // CHECK: %[[RESULT:.*]] = call @select_first(%arg0, %arg1) : (tensor<10x2xf32>, tensor<10x2xf32>) -> tensor<10x2xf32>
  %0 = "tf.LegacyCall"(%arg0, %arg1) {_disable_call_shape_inference = true, _tpu_replicate = "cluster", device = "", f = @select_first} : (tensor<10x2xf32>, tensor<10x2xf32>) -> tensor<10x2xf32>
  // CHECK: return %[[RESULT]]
  func.return %0: tensor<10x2xf32>
}

//===----------------------------------------------------------------------===//
// Conv op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: conv_simple
func.func @conv_simple(%arg0: tensor<256x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32> {

  // CHECK: mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [4, 5], pad = [[0, 1], [2, 3]], rhs_dilate = [2, 3]}
  // CHECK-SAME: batch_group_count = 1
  // CHECK-SAME: feature_group_count = 2

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
  func.return %0 : tensor<256x8x7x16xf32>
}

// -----

// CHECK-LABEL: conv3d_simple
func.func @conv3d_simple(%arg0: tensor<256x32x32x32x6xf32>, %arg1: tensor<3x3x3x3x16xf32>) -> tensor<256x7x6x5x16xf32> {

  // CHECK: mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]
  // CHECK-SAME{LITERAL}: window = {stride = [5, 6, 7], pad = [[1, 2], [2, 3], [2, 3]], rhs_dilate = [2, 3, 4]}
  // CHECK-SAME: batch_group_count = 1
  // CHECK-SAME: feature_group_count = 2

  %0 = "tf.Conv3D"(%arg0, %arg1) {data_format = "NDHWC", dilations = [1, 2, 3, 4, 1], padding = "SAME", strides = [1, 5, 6, 7, 1]} : (tensor<256x32x32x32x6xf32>, tensor<3x3x3x3x16xf32>) -> tensor<256x7x6x5x16xf32>
  func.return %0 : tensor<256x7x6x5x16xf32>
}

// -----

// CHECK-LABEL: depthwiseconv_simple
func.func @depthwiseconv_simple(%arg0: tensor<?x4x5x3xf32>, %arg1: tensor<2x2x3x3xf32>) -> tensor<?x3x4x9xf32> {
  // CHECK: %[[RESHAPED_FILTER:.*]] = "mhlo.reshape"(%arg1) : (tensor<2x2x3x3xf32>) -> tensor<2x2x1x9xf32>
  // CHECK: mhlo.convolution(%arg0, %[[RESHAPED_FILTER]])
  // CHECK-SAME: feature_group_count = 3
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {
    data_format = "NHWC",
    device = "",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1]
  } : (tensor<?x4x5x3xf32>, tensor<2x2x3x3xf32>) -> tensor<?x3x4x9xf32>
  func.return %0 : tensor<?x3x4x9xf32>
}

// -----

// CHECK-LABEL: conv_valid_padding
func.func @conv_valid_padding(%arg0: tensor<1x4x5x1xf32>, %arg1: tensor<3x3x1x1xf32>) -> tensor<1x2x3x1xf32> {
  // CHECK: mhlo.convolution(%arg0, %arg1)

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x4x5x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x3x1xf32>
  func.return %0 : tensor<1x2x3x1xf32>
}

// -----

// CHECK-LABEL: conv_explicit_paddings
func.func @conv_explicit_paddings(%arg0: tensor<256x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x9x7x16xf32> {

  // CHECK: mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME{LITERAL}: pad = [[6, 0], [3, 3]]

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "EXPLICIT", explicit_paddings = [0, 0, 6, 0, 3, 3, 0, 0], strides = [1, 4, 5, 1]} : (tensor<256x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<256x9x7x16xf32>
  func.return %0 : tensor<256x9x7x16xf32>
}

// -----

// CHECK-LABEL: @conv2d_backprop_input
func.func @conv2d_backprop_input(
    %filter: tensor<3x3x1x32xf32>,
    %out_backprop: tensor<100x26x26x32xf32>
  ) -> tensor<100x28x28x1xf32> {
    // CHECK: %[[REV_FILTER:.*]] = "mhlo.reverse"(%arg0) {dimensions = dense<[0, 1]> : tensor<2xi64>}
    // CHECK: %[[RESULT:.*]] = mhlo.convolution(%arg1, %[[REV_FILTER]])
    // CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]
    // CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
    // CHECK-SAME: batch_group_count = 1 : i64
    // CHECK-SAME: feature_group_count = 1 : i64
    // CHECK: return %[[RESULT]]
  %input_sizes = "tf.Const" () { value = dense<[100,28,28,1]> : tensor<4xi32> } : () -> tensor<4xi32>
  %result = "tf.Conv2DBackpropInput"(%input_sizes, %filter, %out_backprop) {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1],
    use_cudnn_on_gpu = true
  } : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<100x26x26x32xf32>) -> tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: @conv2d_backprop_input_grouped
func.func @conv2d_backprop_input_grouped(
    %filter: tensor<2x2x5x21xf32>,
    %out_backprop: tensor<5x2x2x21xf32>
  ) -> tensor<5x3x3x15xf32> {
  %input_sizes = "tf.Const" () { value = dense<[5, 3, 3, 15]> : tensor<4xi32> } : () -> tensor<4xi32>

  // Verify filter transformation for grouped convolution.

  // CHECK: %[[RESHAPE:.*]] = "mhlo.reshape"(%arg0) : (tensor<2x2x5x21xf32>) -> tensor<2x2x5x3x7xf32>
  // CHECK: %[[TRANSPOSE:.*]] = "mhlo.transpose"(%[[RESHAPE]])
  // CHECK-SAME: permutation = dense<[0, 1, 3, 2, 4]>
  // CHECK-SAME: (tensor<2x2x5x3x7xf32>) -> tensor<2x2x3x5x7xf32>
  // CHECK: "mhlo.reshape"(%[[TRANSPOSE]]) : (tensor<2x2x3x5x7xf32>) -> tensor<2x2x15x7xf32>

  %result = "tf.Conv2DBackpropInput"(%input_sizes, %filter, %out_backprop) {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1],
    use_cudnn_on_gpu = true
  } : (tensor<4xi32>, tensor<2x2x5x21xf32>, tensor<5x2x2x21xf32>) -> tensor<5x3x3x15xf32>
  func.return %result : tensor<5x3x3x15xf32>
}


// CHECK-LABEL: @conv3d_backprop_input
func.func @conv3d_backprop_input(%filter: tensor<3x3x3x1x6xf32>, %out_backprop: tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32> {
  // CHECK: %[[REV_FILTER:.*]] = "mhlo.reverse"(%arg0) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
  // CHECK: %[[RESULT:.*]] = mhlo.convolution(%arg1, %[[REV_FILTER]])
  // CHECK-SAME: dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, o, i]->[b, 0, 1, 2, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 1, 1], pad = [[1, 1], [1, 1], [1, 1]], lhs_dilate = [1, 1, 1], rhs_dilate = [1, 1, 1]}
  // CHECK-SAME: batch_group_count = 1 : i64,
  // CHECK-SAME: feature_group_count = 1 : i64

  // CHECK: return %[[RESULT]]
  %input_sizes = "tf.Const" () {value = dense<[2, 8, 8, 8, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
  %result = "tf.Conv3DBackpropInputV2"(%input_sizes, %filter, %out_backprop) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1],  padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<5xi32>, tensor<3x3x3x1x6xf32>, tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32>
  func.return %result : tensor<2x8x8x8x1xf32>
}

// -----

// CHECK-LABEL: @conv2d_backprop_filter
func.func @conv2d_backprop_filter(
    %input: tensor<100x28x28x1xf32>,
    %out_backprop: tensor<100x26x26x32xf32>
  ) -> tensor<3x3x1x32xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  // CHECK-SAME:  batch_group_count = 1 : i64
  // CHECK-SAME:  feature_group_count = 1 : i64
  // CHECK: return %[[RESULT]]
  %filter_sizes = "tf.Const" () { value = dense<[3,3,1,32]> : tensor<4xi32> } : () -> tensor<4xi32>
  %result = "tf.Conv2DBackpropFilter"(%input, %filter_sizes, %out_backprop) {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1],
    use_cudnn_on_gpu = true
  } : (tensor<100x28x28x1xf32>, tensor<4xi32>, tensor<100x26x26x32xf32>) -> tensor<3x3x1x32xf32>
  func.return %result : tensor<3x3x1x32xf32>
}

// -----

// CHECK-LABEL: @conv2d_backprop_filter_grouped
func.func @conv2d_backprop_filter_grouped(
    %input: tensor<1x2x2x2xf32>,
    %out_backprop: tensor<1x1x1x2xf32>
  ) -> tensor<2x2x1x2xf32> {

  // CHECK: mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME:  batch_group_count = 2 : i64
  // CHECK-SAME:  feature_group_count = 1 : i64

  %filter_sizes = "tf.Const" () { value = dense<[2, 2, 1, 2]> : tensor<4xi32> } : () -> tensor<4xi32>
  %result = "tf.Conv2DBackpropFilter"(%input, %filter_sizes, %out_backprop) {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1],
    use_cudnn_on_gpu = true
  } : (tensor<1x2x2x2xf32>, tensor<4xi32>, tensor<1x1x1x2xf32>) -> tensor<2x2x1x2xf32>
  func.return %result : tensor<2x2x1x2xf32>
}


// CHECK-LABEL: @conv3d_backprop_filter
func.func @conv3d_backprop_filter(%input: tensor<2x8x8x8x1xf32>, %out_backprop: tensor<2x8x8x8x6xf32>) -> tensor<3x3x3x1x6xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.convolution(%arg0, %arg1)
  // CHECK-SAME: dim_numbers = [f, 0, 1, 2, b]x[i, 0, 1, 2, o]->[0, 1, 2, b, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 1, 1], pad = [[1, 1], [1, 1], [1, 1]], lhs_dilate = [1, 1, 1], rhs_dilate = [1, 1, 1]}
  // CHECK-SAME: batch_group_count = 1 : i64
  // CHECK-SAME: feature_group_count = 1 : i64
  // CHECK: return %[[RESULT]]
  %filter_sizes = "tf.Const"() {value = dense<[3, 3, 3, 1, 6]> : tensor<5xi32>} : () -> tensor<5xi32>
  %result = "tf.Conv3DBackpropFilterV2"(%input, %filter_sizes, %out_backprop) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1],  padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<2x8x8x8x1xf32>, tensor<5xi32>, tensor<2x8x8x8x6xf32>) -> tensor<3x3x3x1x6xf32>
  func.return %result : tensor<3x3x3x1x6xf32>
}

// -----

// CHECK-LABEL: @collective_permute
func.func @collective_permute(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %source_target_pairs = "tf.Const" () {
    value = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi32>
  } : () -> tensor<3x2xi32>

  // CHECK: "mhlo.collective_permute"
  // CHECK-SAME: source_target_pairs = dense<{{\[}}[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  %0 = "tf.CollectivePermute"(%arg0, %source_target_pairs) {
  } : (tensor<128x32xf32>, tensor<3x2xi32>) -> tensor<128x32xf32>

  func.return %0 : tensor<128x32xf32>
}

// -----

// CHECK-LABEL: @cross_replica_sum
func.func @cross_replica_sum(%input: tensor<10xf32>) -> tensor<10xf32> {
  %replica_groups = "tf.Const" () {
    value = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  } : () -> tensor<2x4xi32>

  // CHECK: mhlo.cross-replica-sum
  // CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  %result = "tf.CrossReplicaSum" (%input, %replica_groups) : (tensor<10xf32>, tensor<2x4xi32>) -> tensor<10xf32>
  func.return %result : tensor<10xf32>
}

// -----

// CHECK-LABEL: conv_dynamic
func.func @conv_dynamic(%arg0: tensor<?x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32> {
  // CHECK: "mhlo.dynamic_conv"
  // CHECK-SAME: {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 2 : i64, rhs_dilation = dense<[2, 3]> : tensor<2xi64>, window_strides = dense<[4, 5]> : tensor<2xi64>} : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<?x8x7x16xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<?x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<?x8x7x16xf32>
  func.return %0 : tensor<?x8x7x16xf32>
}

//===----------------------------------------------------------------------===//
// tf.Split legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @split_not_match_non_const_split_dim
func.func @split_not_match_non_const_split_dim(%input: tensor<4x4xf32>, %split_dim: tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK: tf.Split
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<i32>, tensor<4x4xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: @split_not_match_unknown_input_dim
func.func @split_not_match_unknown_input_dim(%input: tensor<4x?x4xf32>) -> (tensor<4x?x4xf32>, tensor<4x?x4xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tensor.dim {{.*}} : tensor<4x?x4xf32>
  // CHECK: arith.divsi {{.*}} : index
  // CHECK: tensor.from_elements {{.*}} : tensor<3xindex>
  // CHECK: "mhlo.real_dynamic_slice"({{.*}}) : (tensor<4x?x4xf32>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<4x?x4xf32>
  // CHECK: muli {{.*}} : index
  // CHECK: muli {{.*}} : index
  // CHECK: tensor.from_elements {{.*}} : tensor<3xindex>
  // CHECK: tensor.from_elements {{.*}} : tensor<3xindex>
  // CHECK: tensor.from_elements {{.*}} : tensor<3xindex>
  // CHECK: "mhlo.real_dynamic_slice"({{.*}}) : (tensor<4x?x4xf32>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>) -> tensor<4x?x4xf32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x?x4xf32>) -> (tensor<4x?x4xf32>, tensor<4x?x4xf32>)
  func.return %0#0, %0#1 : tensor<4x?x4xf32>, tensor<4x?x4xf32>
}

// -----

// CHECK-LABEL: @split_match_and_split_into_two
func.func @split_match_and_split_into_two(%input: tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>) {
  %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[2, 6]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<2x6xf32>
  // CHECK: %[[TWO:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<2x6xf32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
  // CHECK: return %[[ONE]], %[[TWO]]
  func.return %0#0, %0#1 : tensor<2x6xf32>, tensor<2x6xf32>
}

// -----

// CHECK-LABEL: @split_match_and_split_into_two_dynamic
func.func @split_match_and_split_into_two_dynamic(%input: tensor<4x?xf32>) -> (tensor<2x?xf32>, tensor<2x?xf32>) {
  %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[2, -1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x?xf32>) -> tensor<2x?xf32>
  // CHECK: %[[TWO:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[4, -1]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x?xf32>) -> tensor<2x?xf32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x?xf32>) -> (tensor<2x?xf32>, tensor<2x?xf32>)
  // CHECK: return %[[ONE]], %[[TWO]]
  func.return %0#0, %0#1 : tensor<2x?xf32>, tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @split_match_and_split_into_three
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x6xf32>)
func.func @split_match_and_split_into_three(%input: tensor<4x6xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[TWO:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<4> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[THREE:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  %0:3 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
  // CHECK: return %[[ONE]], %[[TWO]], %[[THREE]]
  func.return %0#0, %0#1, %0#2 : tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>
}

//===----------------------------------------------------------------------===//
// tf.TopKV2 legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: topk_v2_non_const_k
func.func @topk_v2_non_const_k(%input: tensor<16xf32>, %k: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // CHECK: tf.TopKV2
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
  func.return %0#0, %0#1: tensor<?xf32>, tensor<?xi32>
}

// -----

// CHECK-LABEL: topk_v2_unknown_input_last_dim
func.func @topk_v2_unknown_input_last_dim(%input: tensor<16x?xf32>) -> (tensor<16x?xf32>, tensor<16x?xi32>) {
  %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.TopKV2
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x?xf32>, tensor<i32>) -> (tensor<16x?xf32>, tensor<16x?xi32>)
  func.return %0#0, %0#1: tensor<16x?xf32>, tensor<16x?xi32>
}

// -----

// CHECK-LABEL: topk_v2
// CHECK-SAME: %[[INPUT:.*]]: tensor<16x16xf32>
func.func @topk_v2(%input: tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>

  // CHECK:     chlo.top_k(%[[INPUT]], k = 8)
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x16xf32>, tensor<i32>) -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %0#0, %0#1: tensor<16x8xf32>, tensor<16x8xi32>
}

//===----------------------------------------------------------------------===//
// tf.SplitV legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @splitv_match_and_split_into_three
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x6xf32>)
func.func @splitv_match_and_split_into_three(%input: tensor<4x6xf32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x1xf32>
  // CHECK: %[[TWO:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[THREE:.*]] = "mhlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x3xf32>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
  // CHECK: return %[[ONE]], %[[TWO]], %[[THREE]]
  func.return %0#0, %0#1, %0#2 : tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>
}

// -----

// CHECK-LABEL: @splitv_match_and_split_into_three_dynamic
func.func @splitv_match_and_split_into_three_dynamic(%input: tensor<?x6xf32>) -> (tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "mhlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x1xf32>
  // CHECK: "mhlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x2xf32>
  // CHECK: "mhlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x3xf32>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<?x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>
}

// -----

// CHECK-LABEL: @splitv_dynamic_dim_in_split_sizes
func.func @splitv_dynamic_dim_in_split_sizes(%input: tensor<4x6xf32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: limit_indices = dense<[4, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>
  // CHECK: limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>
  // CHECK: limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>
}

//===----------------------------------------------------------------------===//
// tf.Assert legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @assert
func.func @assert(%arg0: tensor<i1>, %arg1: tensor<*xf32>) {
  // CHECK-NOT: tf.Assert
  "tf.Assert"(%arg0, %arg1) {summarize = 1} : (tensor<i1>, tensor<*xf32>) -> ()
  func.return
}

//===----------------------------------------------------------------------===//
// tf.Unpack legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @unpack
func.func @unpack(%input: tensor<4x3x6xf32>) -> (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
  // CHECK: %[[SLICE1:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[4, 1, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
  // CHECK: %[[RES1:.*]] = "mhlo.reshape"(%[[SLICE1]]) : (tensor<4x1x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[SLICE2:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[4, 2, 6]> : tensor<3xi64>, start_indices = dense<[0, 1, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
  // CHECK: %[[RES2:.*]] = "mhlo.reshape"(%[[SLICE2]]) : (tensor<4x1x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[SLICE3:.*]] = "mhlo.slice"(%{{.*}}) {limit_indices = dense<[4, 3, 6]> : tensor<3xi64>, start_indices = dense<[0, 2, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
  // CHECK: %[[RES3:.*]] = "mhlo.reshape"(%[[SLICE3]]) : (tensor<4x1x6xf32>) -> tensor<4x6xf32>

  %0:3 = "tf.Unpack"(%input) {axis = 1} : (tensor<4x3x6xf32>) -> (tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>)
  // return %[[RES1]], %[[RES2]], %[[RES3]]
  func.return %0#0, %0#1, %0#2 : tensor<4x6xf32>, tensor<4x6xf32>, tensor<4x6xf32>
}

// -----

// CHECK-LABEL: func @unpack_dynamic
func.func @unpack_dynamic(%arg0: tensor<?x?x2xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK: "mhlo.real_dynamic_slice"({{.*}}) : (tensor<?x?x2xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x1xf32>
  // CHECK: tensor.from_elements {{.*}} : tensor<2xi32>
  // CHECK: "mhlo.dynamic_reshape"({{.*}}) : (tensor<?x?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // CHECK: tensor.from_elements {{.*}} : tensor<3xi32>
  // CHECK: "mhlo.real_dynamic_slice"({{.*}}) : (tensor<?x?x2xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x?x1xf32>
  // CHECK: tensor.from_elements {{.*}} : tensor<2xi32>
  // CHECK: "mhlo.dynamic_reshape"({{.*}}) : (tensor<?x?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  // CHECK: return {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>
  %0:2 = "tf.Unpack"(%arg0) {axis = -1 : i64} : (tensor<?x?x2xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @unpack_unranked
func.func @unpack_unranked(%input: tensor<*xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {

  // CHECK: tf.Unpack
  %0:2 = "tf.Unpack"(%input) {axis = -1} : (tensor<*xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

//===----------------------------------------------------------------------===//
// tf.UnsortedSegment{Max|Min|Prod|Sum} legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @unsorted_segment_sum
// CHECK-SAME: [[DATA:%.*]]: tensor<8x16x64xf32>
// CHECK-SAME: [[SI:%.*]]: tensor<8x16xi32>
func.func @unsorted_segment_sum(%data: tensor<8x16x64xf32>, %segment_ids : tensor<8x16xi32>) -> (tensor<4x64xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[ZERO:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[INIT:%.*]] = "mhlo.broadcast"([[ZERO]]) {broadcast_sizes = dense<[4, 64]> : tensor<2xi64>} : (tensor<f32>) -> tensor<4x64xf32>
  // CHECK: [[SCATTER:%.*]] = "mhlo.scatter"([[INIT]], [[SI]], [[DATA]]) ({
  // CHECK: ^{{.*}}([[LHS:%.*]]: tensor<f32>, [[RHS:%.*]]: tensor<f32>):
  // CHECK:   [[ADD:%.*]] = mhlo.add [[LHS]], [[RHS]] : tensor<f32>
  // CHECK:   "mhlo.return"([[ADD]])
  // CHECK: indices_are_sorted = false,
  // CHECK-SAME: scatter_dimension_numbers =
  // CHECK-SAME:   update_window_dims = [2]
  // CHECK-SAME:   inserted_window_dims = [0]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0]
  // CHECK-SAME:   index_vector_dim = 2
  // CHECK-SAME: unique_indices = false
  // CHECK-SAME: (tensor<4x64xf32>, tensor<8x16xi32>, tensor<8x16x64xf32>) -> tensor<4x64xf32>
  // CHECK: return [[SCATTER]]
  %0 = "tf.UnsortedSegmentSum"(%data, %segment_ids, %num_segments) : (tensor<8x16x64xf32>, tensor<8x16xi32>, tensor<i32>) -> (tensor<4x64xf32>)
  func.return %0: tensor<4x64xf32>
}

// -----

// CHECK-LABEL: @unsorted_segment_prod
// CHECK-SAME: [[DATA:%.*]]: tensor<8x?x64xf32>
// CHECK-SAME: [[SI:%.*]]: tensor<?x16xi32>
func.func @unsorted_segment_prod(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[ONE:%.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: [[INIT:%.*]] = "mhlo.broadcast"([[ONE]]) {broadcast_sizes = dense<[4, 64]> : tensor<2xi64>} : (tensor<f32>) -> tensor<4x64xf32>
  // CHECK: [[SCATTER:%.*]] = "mhlo.scatter"([[INIT]], [[SI]], [[DATA]]) ({
  // CHECK: ^{{.*}}([[LHS:%.*]]: tensor<f32>, [[RHS:%.*]]: tensor<f32>):
  // CHECK:   [[MUL:%.*]] = mhlo.multiply [[LHS]], [[RHS]] : tensor<f32>
  // CHECK:   "mhlo.return"([[MUL]])
  // CHECK: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers =
  // CHECK-SAME:   update_window_dims = [2]
  // CHECK-SAME:   inserted_window_dims = [0]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0]
  // CHECK-SAME:   index_vector_dim = 2
  // CHECK-SAME: unique_indices = false
  // CHECK-SAME: (tensor<4x64xf32>, tensor<?x16xi32>, tensor<8x?x64xf32>) -> tensor<4x?xf32>
  // CHECK: return [[SCATTER]]
  %0 = "tf.UnsortedSegmentProd"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  func.return %0: tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @unsorted_segment_min
func.func @unsorted_segment_min(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: mhlo.constant dense<3.40282347E+38> : tensor<f32>
  // CHECK: mhlo.scatter
  // CHECK: mhlo.minimum
  %0 = "tf.UnsortedSegmentMin"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  func.return %0: tensor<4x?xf32>
}

// -----

// CHECK-LABEL: @unsorted_segment_max
func.func @unsorted_segment_max(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  // CHECK: mhlo.scatter
  // CHECK: mhlo.maximum
  %0 = "tf.UnsortedSegmentMax"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  func.return %0: tensor<4x?xf32>
}

//===----------------------------------------------------------------------===//
// tf.GatherNd legalization
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @gatherNd_dynamic
func.func @gatherNd_dynamic(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?x6x2xi32>) -> tensor<?x6x?xi32> {
  // CHECK: tensor.dim
  // CHECK: index_cast
  // CHECK: tensor.from_elements
  // CHECK: mhlo.dynamic_gather
  // CHECK-SAME: dimension_numbers =
  // CHECK-SAME:   offset_dims = [2]
  // CHECK-SAME:   collapsed_slice_dims = [0, 1]
  // CHECK-SAME:   start_index_map = [0, 1]
  // CHECK-SAME:   index_vector_dim = 2
  // CHECK-SAME: indices_are_sorted = false
  %0 =  "tf.GatherNd"(%arg0, %arg1) {Tindices = i32, Tparams = i32, device = ""} : (tensor<?x?x?xi32>, tensor<?x6x2xi32>) -> tensor<?x6x?xi32>
  func.return %0 : tensor<?x6x?xi32>
}

// -----

// CHECK-LABEL: func @gatherNd_static
func.func @gatherNd_static(%arg0: tensor<2x4x128xf32>, %arg1: tensor<2x1xi32>) -> tensor<2x4x128xf32> {
  // CHECK:      "mhlo.gather"({{.*}}) {
  // CHECK-SAME:   dimension_numbers =
  // CHECK-SAME:     offset_dims = [1, 2]
  // CHECK-SAME:     collapsed_slice_dims = [0]
  // CHECK-SAME:     start_index_map = [0]
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   indices_are_sorted = false
  // CHECK-SAME:   slice_sizes = dense<[1, 4, 128]>
  // CHECK-SAME: (tensor<2x4x128xf32>, tensor<2x1xi32>) -> tensor<2x4x128xf32>
  %0 =  "tf.GatherNd"(%arg0, %arg1) {Tindices = i32, Tparams = i32, device = ""} : (tensor<2x4x128xf32>, tensor<2x1xi32>) -> tensor<2x4x128xf32>
  func.return %0 : tensor<2x4x128xf32>
}

//===----------------------------------------------------------------------===//
// tf.GatherV2 legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @gather_v2
//  CHECK-SAME: %[[PARAMS:[a-zA-Z0-9_]+]]
//  CHECK-SAME: %[[INDICES:[a-zA-Z0-9_]+]]
func.func @gather_v2(%params: tensor<16x2x3xf32>, %indices: tensor<16x5xi32>) -> tensor<16x2x5xf32> {
  //      CHECK: mhlo.torch_index_select
  // CHECK-SAME:   %[[PARAMS]], %[[INDICES]]
  // CHECK-SAME:   batch_dims = 1
  // CHECK-SAME:   dim = 2
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%params, %indices, %axis) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5xf32>
  func.return %1 : tensor<16x2x5xf32>
}

// -----

// CHECK-LABEL: @gather_v2_dynamic
//  CHECK-SAME: %[[PARAMS:[a-zA-Z0-9_]+]]
//  CHECK-SAME: %[[INDICES:[a-zA-Z0-9_]+]]
func.func @gather_v2_dynamic(%params: tensor<?x?x?xf32>, %indices: tensor<?x?xi32>) -> tensor<*xf32> {
  //      CHECK: mhlo.torch_index_select
  // CHECK-SAME:   %[[PARAMS]], %[[INDICES]]
  // CHECK-SAME:   batch_dims = 1
  // CHECK-SAME:   dim = 2
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%params, %indices, %axis) {batch_dims = -1 : i64} : (tensor<?x?x?xf32>, tensor<?x?xi32>, tensor<1xi32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @gather_v2_dynamic_index_i64
//  CHECK-SAME: %[[PARAMS:[a-zA-Z0-9_]+]]
//  CHECK-SAME: %[[INDICES:[a-zA-Z0-9_]+]]
func.func @gather_v2_dynamic_index_i64(%params: tensor<?x?x?xf32>, %indices: tensor<?x?xi64>) -> tensor<*xf32> {
  //      CHECK: mhlo.torch_index_select
  // CHECK-SAME:   %[[PARAMS]], %[[INDICES]]
  // CHECK-SAME:   batch_dims = 1
  // CHECK-SAME:   dim = 2
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%params, %indices, %axis) {batch_dims = -1 : i64} : (tensor<?x?x?xf32>, tensor<?x?xi64>, tensor<1xi32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @gather_v2_unranked
func.func @gather_v2_unranked(%params: tensor<*xf32>, %indices: tensor<*xi32>) -> tensor<*xf32> {
  // CHECK: tf.GatherV2
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%params, %indices, %axis) {batch_dims = -1 : i64} : (tensor<*xf32>, tensor<*xi32>, tensor<1xi32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @gather_v2_dynamic_shape
//  CHECK-SAME: %[[PARAMS:[a-zA-Z0-9_]+]]
//  CHECK-SAME: %[[INDICES:[a-zA-Z0-9_]+]]
func.func @gather_v2_dynamic_shape(%params: tensor<?x2x3xf32>, %indices: tensor<?x5xi32>) -> tensor<?x2x5xf32> {
  //      CHECK: mhlo.torch_index_select
  // CHECK-SAME:   %[[PARAMS]], %[[INDICES]]
  // CHECK-SAME:   batch_dims = 1
  // CHECK-SAME:   dim = 2
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%params, %indices, %axis) {batch_dims = -1 : i64} : (tensor<?x2x3xf32>, tensor<?x5xi32>, tensor<1xi32>) -> tensor<?x2x5xf32>
  func.return %1 : tensor<?x2x5xf32>
}

//===----------------------------------------------------------------------===//
// tf.StridedSliceGrad legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: strided_slice_grad
// CHECK-SAME: [[GRAD:%.*]]: tensor<4x16x1022xf32>
func.func @strided_slice_grad(%grad: tensor<4x16x1022xf32>) -> tensor<4x128x1024xf32> {

  // For StridedSlice
  // Dim #:        0,   1,    2
  // Input shape: [4, 128, 1024]
  // Begin:        1,   4,   -3
  // End:          8,  65,   42
  // Stride:       1,   4,   -1
  // Begin mask:   1,   0,    0  (= 1)
  // End mask:     0,   0,    1  (= 4)

  // So result shape:
  // Dim #0: begin mask (1) -> begin = 0; end 8 canonicalized to 4: so 4
  // Dim #1: 4 to 65 stride 4: so 16
  // Dim #2: begin -3 + 1024 = 1021; end mask (1) -> end = -1: so 1022
  // result shape: [4, 16, 1022]

  // To pad back:
  // Dim #:        0,   1,   2
  // Pad low:      0,   4,   0
  // Pad interm:   0,   3,   0
  // Pad high:     0,  63,   2

  %shape = "tf.Const"() {value = dense<[4, 128, 1024]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %begin = "tf.Const"() {value = dense<[1, 4, -3]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %end = "tf.Const"() {value = dense<[8, 65, 42]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %strides = "tf.Const"() {value = dense<[1, 4, -1]> : tensor<3xi32>} : () -> (tensor<3xi32>)

  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"(%arg0) : (tensor<4x16x1022xf32>) -> tensor<4x16x1022xf32>
  // CHECK: [[REVERSE:%.*]] = "mhlo.reverse"([[RESHAPE]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x16x1022xf32>) -> tensor<4x16x1022xf32>
  // CHECK: [[ZERO:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[REVERSE]], [[ZERO]]) {edge_padding_high = dense<[0, 63, 2]> : tensor<3xi64>, edge_padding_low = dense<[0, 4, 0]> : tensor<3xi64>, interior_padding = dense<[0, 3, 0]> : tensor<3xi64>} : (tensor<4x16x1022xf32>, tensor<f32>) -> tensor<4x128x1024xf32>

  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 1, end_mask = 4} : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<4x16x1022xf32>) -> tensor<4x128x1024xf32>
  // CHECK: return [[PAD]]
  func.return %0: tensor<4x128x1024xf32>
}

// -----

// CHECK-LABEL: strided_slice_grad_shrink_axis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<8xf32>
func.func @strided_slice_grad_shrink_axis_mask(%grad: tensor<8xf32>) -> tensor<4x8xf32> {
  // Input to StridedSlice was of shape 4x8xf32
  // Strided slice gets input[2:3, 0:8]
  // shrink_axis_mask is 1 denoting that dim#0 is shrunk. So the output is 8xf32
  // which is the shape of gradient.
  // StridedSliceGrad would reshape the gradient to 1x8xf32 and
  // then pad to match the shape of input 4x8xf32.

  %shape = "tf.Const"() {value = dense<[4, 8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %begin = "tf.Const"() {value = dense<[2, 0]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[GRAD]]) : (tensor<8xf32>) -> tensor<1x8xf32>
  // CHECK: [[ZEROS:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[2, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<[1, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<2xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, shrink_axis_mask = 1} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<8xf32>) -> tensor<4x8xf32>

  // CHECK: return [[PAD]] : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: strided_slice_grad_new_axis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<1x2xf32>
func.func @strided_slice_grad_new_axis_mask(%grad: tensor<1x2xf32>) -> tensor<8xf32> {
  // Input to StridedSlice was of shape 8xf32
  // Strided slice gets input[tf.new_axis, 2:4]
  // new_axis_mask is 1 denoting new axis is inserted at dim#0. So the output is
  // 1x2xf32 which is the shape of gradient.
  // StridedSliceGrad would reshape the gradient to 2xf32 and
  // then pad to match the shape of input 4x8xf32.

  %shape = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %begin = "tf.Const"() {value = dense<[0, 2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[0, 4]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[GRAD]]) : (tensor<1x2xf32>) -> tensor<2xf32>
  // CHECK: [[ZEROS:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<2> : tensor<1xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<4> : tensor<1xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<1xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, new_axis_mask = 1} : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<1x2xf32>) -> tensor<8xf32>

  // CHECK: return [[PAD]] : tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: strided_slice_grad_ellipsis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<2x4x8xf32>
func.func @strided_slice_grad_ellipsis_mask(%grad: tensor<2x4x8xf32>) -> tensor<4x4x8xf32> {
  // Input to StridedSlice was of shape 4x4x8xf32
  // Strided slice gets input[2:4, ...]
  // ellipsis_mask is 2 denoting that slice contains all elements in dim#1 and
  // dim#2, ignoring begin and end indices for these dimensions. So the output
  // is 2x4x8xf32 which is the shape of gradient.
  // StridedSliceGrad would pad the gradient to match the shape of
  // input 4x4x8xf32.

  %shape = "tf.Const"() {value = dense<[4, 4, 8]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %begin = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[4, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[GRAD]]) : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  // CHECK: [[ZEROS:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[2, 0, 0]> : tensor<3xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<0> : tensor<3xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<3xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, ellipsis_mask = 2} : (tensor<3xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x4x8xf32>) -> tensor<4x4x8xf32>

  // CHECK: return [[PAD]] : tensor<4x4x8xf32>
  func.return %0 : tensor<4x4x8xf32>
}


// CHECK-LABEL: strided_slice_grad_all_masks
// CHECK-SAME: [[GRAD:%.*]]: tensor<1x4x8x8x10x2x1xf32>
func.func @strided_slice_grad_all_masks(%grad: tensor<1x4x8x8x10x2x1xf32>) -> tensor<2x4x8x16x32x64xf32> {
  // For StridedSlice input[1, tf.new_axis, ..., 8:, :10, 2:6:2, tf.new_axis]
  // New axis mask is at index 1 and 6 of sparse spec, so
  // new_axis_mask = 2^1 + 2^6 = 66
  // The ellipsis mask is applied to dim #1, #2 of input i.e, we get
  // canonicalized slice input[1, :, :, 8:, :10, 2:6:2]
  // The StridedSliceGrad op would propogate the gradient for the sliced tensor
  // to the original input tensor by padding with zeroes.

  %shape = "tf.Const"() {value = dense<[2, 4, 8, 16, 32, 64]> : tensor<6xi32>} : () -> (tensor<6xi32>)
  %begin = "tf.Const"() {value = dense<[1, 0, 0, 8, 1, 2, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %end = "tf.Const"() {value = dense<[2, 0, 0, 10, 10, 6, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %strides = "tf.Const"() {value = dense<[1, 1, 1, 1, 1, 2, 1]> : tensor<7xi32>} : () -> (tensor<7xi32>)

  // Remove 2 new axes (at index 1 and 6) and 1 shrink axis (at index 0)
  // CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[GRAD]]) : (tensor<1x4x8x8x10x2x1xf32>) -> tensor<1x4x8x8x10x2xf32>
  // CHECK: [[ZERO:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // The edge_padding_low, edge_padding_high and interior_padding attributes of
  // mhlo.pad would reflect the padding required to get the shape of the
  // input of StridedSlice op.
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[RESHAPE]], [[ZERO]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<[0, 0, 0, 0, 22, 59]> : tensor<6xi64>
  // CHECK-DAG-SAME: interior_padding = dense<[0, 0, 0, 0, 0, 1]> : tensor<6xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 16, end_mask = 8, shrink_axis_mask = 1, ellipsis_mask = 4, new_axis_mask = 66} : (tensor<6xi32>, tensor<7xi32>, tensor<7xi32>, tensor<7xi32>, tensor<1x4x8x8x10x2x1xf32>) -> tensor<2x4x8x16x32x64xf32>

  // CHECK: return [[PAD]] : tensor<2x4x8x16x32x64xf32>
  func.return %0 : tensor<2x4x8x16x32x64xf32>
}

// -----

// CHECK-LABEL: @tensor_scatter_update
func.func @tensor_scatter_update(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    "mhlo.return"(%arg4) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   update_window_dims = [1]
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @tensor_scatter_add
func.func @tensor_scatter_add(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    %1 = mhlo.add %arg3, %arg4 : tensor<f32>
  // CHECK:    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   update_window_dims = [1]
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterAdd"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @tensor_scatter_sub
func.func @tensor_scatter_sub(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    %1 = mhlo.subtract %arg3, %arg4 : tensor<f32>
  // CHECK:    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   update_window_dims = [1]
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterSub"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @tensor_scatter_min
func.func @tensor_scatter_min(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    %1 = mhlo.minimum %arg3, %arg4 : tensor<f32>
  // CHECK:    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   update_window_dims = [1]
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterMin"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @tensor_scatter_max
func.func @tensor_scatter_max(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    %1 = mhlo.maximum %arg3, %arg4 : tensor<f32>
  // CHECK:    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   update_window_dims = [1]
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterMax"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

//===----------------------------------------------------------------------===//
// tf.RandomShuffle legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @random_shuffle_first_dim_1
// CHECK-SAME: [[INPUT:%.*]]: tensor<1x?xf32>
func.func @random_shuffle_first_dim_1(%input: tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = "tf.RandomShuffle"(%input) : (tensor<1x?xf32>) -> (tensor<1x?xf32>)
  // CHECK-NEXT: return [[INPUT]]
  func.return %0: tensor<1x?xf32>
}

// -----

// CHECK-LABEL: @random_shuffle_1D_16
// CHECK-SAME: [[INPUT:%.*]]: tensor<16xf32>
func.func @random_shuffle_1D_16(%input: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK-DAG: [[SHAPE:%.*]] = mhlo.constant dense<16> : tensor<1xi64>
  // CHECK-DAG: [[LOWER:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[UPPER:%.*]] = mhlo.constant dense<-1> : tensor<i32>
  // CHECK: [[RNG:%.*]] = "mhlo.rng"([[LOWER]], [[UPPER]], [[SHAPE]]) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}
  // CHECK: [[SORT:%.*]]:2 = "mhlo.sort"([[RNG]], [[INPUT]]) ({
  // CHECK: ^{{.*}}([[ARG1:%.*]]: tensor<i32>, [[ARG2:%.*]]: tensor<i32>, {{.*}}: tensor<f32>, {{.*}}: tensor<f32>):
  // CHECK:   "mhlo.compare"([[ARG1]], [[ARG2]]) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK: }) {dimension = -1 : i64, is_stable = {{.*}}} : (tensor<16xi32>, tensor<16xf32>) -> (tensor<16xi32>, tensor<16xf32>)
  // CHECK: return [[SORT]]#1
  %0 = "tf.RandomShuffle"(%input) : (tensor<16xf32>) -> (tensor<16xf32>)
  func.return %0: tensor<16xf32>
}

// -----

// CHECK-LABEL: @random_shuffle_1D_10240
func.func @random_shuffle_1D_10240(%input: tensor<10240xf32>) -> tensor<10240xf32> {
  // CHECK: mhlo.rng{{.*UNIFORM.*}}
  // CHECK: mhlo.sort
  // CHECK: mhlo.rng{{.*UNIFORM.*}}
  // CHECK: mhlo.sort
  %0 = "tf.RandomShuffle"(%input) : (tensor<10240xf32>) -> (tensor<10240xf32>)
  func.return %0: tensor<10240xf32>
}

// -----

// CHECK-LABEL: @random_shuffle_3D
// CHECK-SAME: [[INPUT:%.*]]: tensor<4x?x16xf32>
func.func @random_shuffle_3D(%input: tensor<4x?x16xf32>) -> tensor<4x?x16xf32> {
  // CHECK: [[INDICES:%.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>

  // CHECK-DAG: [[RNG_SHAPE:%.*]] = mhlo.constant dense<4> : tensor<1xi64>
  // CHECK-DAG: [[RNG_LOWER:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: [[RNG_UPPER:%.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK: [[SWAPS:%.*]] = "mhlo.rng"([[RNG_LOWER]], [[RNG_UPPER]], [[RNG_SHAPE]]) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}

  // CHECK: [[IV_INIT:%.*]] = mhlo.constant dense<0> : tensor<i32>

  // CHECK: [[WHILE_OUT:%.*]]:3 = mhlo.while([[ITER_ARG:.*]] = [[IV_INIT]], [[ITER_ARG1:.*]] = [[SWAPS]], [[ITER_ARG2:.*]] = [[INDICES]])
  // CHECK:   [[LIMIT:%.*]] = mhlo.constant dense<4> : tensor<i32>
  // CHECK:   [[CMP:%.*]] = "mhlo.compare"([[ITER_ARG]], [[LIMIT]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK:   "mhlo.return"([[CMP]])
  // CHECK: } do {
  // CHECK:   [[SRC_IDX:%.*]] = "mhlo.dynamic_slice"([[ITER_ARG2]], [[ITER_ARG]]) {slice_sizes = dense<1> : tensor<i64>} : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:   [[SWP_IDX:%.*]] = "mhlo.dynamic_slice"([[ITER_ARG1]], [[ITER_ARG]]) {slice_sizes = dense<1> : tensor<i64>} : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:   [[SWP:%.*]] = "mhlo.reshape"([[SWP_IDX]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK:   [[TGT_IDX:%.*]] = "mhlo.dynamic_slice"([[ITER_ARG2]], [[SWP]]) {slice_sizes = dense<1> : tensor<i64>}
  // CHECK:   [[INDICES1:%.*]] = "mhlo.dynamic_update_slice"([[ITER_ARG2]], [[TGT_IDX]], [[ITER_ARG]]) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK:   [[INDICES2:%.*]] = "mhlo.dynamic_update_slice"([[INDICES1]], [[SRC_IDX]], [[SWP]]) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK:   [[ONE:%.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK:   [[NEW_IV:%.*]] = chlo.broadcast_add [[ITER_ARG]], [[ONE]]
  // CHECK:   "mhlo.return"([[NEW_IV]], [[ITER_ARG1]], [[INDICES2]])

  // CHECK: [[GATHER:%.*]] = "mhlo.gather"([[INPUT]], [[WHILE_OUT]]#2)
  // CHECK-SAME:   dimension_numbers =
  // CHECK-SAME:     offset_dims = [1, 2]
  // CHECK-SAME:     collapsed_slice_dims = [0]
  // CHECK-SAME:     start_index_map = [0]
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: slice_sizes = dense<[1, -1, 16]>
  // CHECK: (tensor<4x?x16xf32>, tensor<4xi32>) -> tensor<4x?x16xf32>

  // CHECK: return [[GATHER]]

  %0 = "tf.RandomShuffle"(%input) : (tensor<4x?x16xf32>) -> (tensor<4x?x16xf32>)
  func.return %0: tensor<4x?x16xf32>
}

//===----------------------------------------------------------------------===//
// tf.AvgPool legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   @avgpool_valid_padding
// CHECK-SAME:      [[ARG:%.+]]: tensor<2x12x21x7xf16>
// CHECK:           [[CONV32:%.+]] = mhlo.convert(%arg0) : (tensor<2x12x21x7xf16>) -> tensor<2x12x21x7xf32>
// CHECK:           [[ZERO:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[DIVIDEND:%.+]] = "mhlo.reduce_window"([[CONV32]], [[ZERO]]) ({
// CHECK:           ^bb0([[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<f32>):
// CHECK:             [[ADD:%.+]] = mhlo.add [[ARG1]], [[ARG2]]
// CHECK:             "mhlo.return"([[ADD]])
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 2, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 4, 4, 1]>
// CHECK-SAME:        -> tensor<2x3x5x7xf32>
// CHECK:           [[COUNT:%.+]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           [[DIV_RESULT:%.+]] = chlo.broadcast_divide [[DIVIDEND]], [[COUNT]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<2x3x5x7xf32>
// CHECK:           [[CONV16:%.+]] = mhlo.convert([[DIV_RESULT]])
// CHECK-SAME:        -> tensor<2x3x5x7xf16>
// CHECK:           return [[CONV16]]
func.func @avgpool_valid_padding(%arg0: tensor<2x12x21x7xf16>) -> tensor<2x3x5x7xf16> {
  %0 = "tf.AvgPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 4, 4, 1]} : (tensor<2x12x21x7xf16>) -> tensor<2x3x5x7xf16>
  func.return %0 : tensor<2x3x5x7xf16>
}

// -----

// CHECK-LABEL:   @avgpool_3d_valid_padding
// CHECK-SAME:      [[ARG:%.+]]: tensor<2x4x12x21x7xf16>
// CHECK:           [[CONV32:%.+]] = mhlo.convert(%arg0) : (tensor<2x4x12x21x7xf16>) -> tensor<2x4x12x21x7xf32>
// CHECK:           [[ZERO:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[DIVIDEND:%.+]] = "mhlo.reduce_window"([[CONV32]], [[ZERO]]) ({
// CHECK:           ^bb0([[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<f32>):
// CHECK:           [[ADD:%.+]] = mhlo.add [[ARG1]], [[ARG2]]
// CHECK:             "mhlo.return"([[ADD]])
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 1, 4, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x3x5x7xf32>
// CHECK:           [[COUNT:%.+]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           [[DIV_RESULT:%.+]] = chlo.broadcast_divide [[DIVIDEND]], [[COUNT]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<2x4x3x5x7xf32>
// CHECK:           [[CONV16:%.+]] = mhlo.convert([[DIV_RESULT]])
// CHECK-SAME:        -> tensor<2x4x3x5x7xf16>
// CHECK:           return [[CONV16]]
func.func @avgpool_3d_valid_padding(%arg0: tensor<2x4x12x21x7xf16>) -> tensor<2x4x3x5x7xf16> {
  %0 = "tf.AvgPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 2, 2, 1], padding = "VALID", strides = [1, 1, 4, 4, 1]} : (tensor<2x4x12x21x7xf16>) -> tensor<2x4x3x5x7xf16>
  func.return %0 : tensor<2x4x3x5x7xf16>
}

// -----

// CHECK-LABEL:   @avgpool_nchw_format
// CHECK-SAME:      [[ARG:%.+]]: tensor<2x7x12x21xf16>
// CHECK:           [[CONV32:%.+]] = mhlo.convert(%arg0) : (tensor<2x7x12x21xf16>) -> tensor<2x7x12x21xf32>
// CHECK:           [[ZERO:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[DIVIDEND:%.+]] = "mhlo.reduce_window"([[CONV32]], [[ZERO]]) ({
// CHECK:           ^bb0([[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<f32>):
// CHECK:             [[ADD:%.+]] = mhlo.add [[ARG1]], [[ARG2]]
// CHECK:             "mhlo.return"([[ADD]])
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 2]>
// CHECK-SAME:        window_strides = dense<[1, 1, 4, 4]>
// CHECK-SAME:        -> tensor<2x7x3x5xf32>
// CHECK:           [[COUNT:%.+]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           [[DIV_RESULT:%.+]] = chlo.broadcast_divide [[DIVIDEND]], [[COUNT]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<2x7x3x5xf32>
// CHECK:           [[CONV16:%.+]] = mhlo.convert([[DIV_RESULT]])
// CHECK-SAME:        -> tensor<2x7x3x5xf16>
// CHECK:           return [[CONV16]]
func.func @avgpool_nchw_format(%arg0: tensor<2x7x12x21xf16>) -> tensor<2x7x3x5xf16> {
  %0 = "tf.AvgPool"(%arg0) {data_format = "NCHW", ksize = [1, 1, 2, 2], padding = "VALID", strides = [1, 1, 4, 4]} : (tensor<2x7x12x21xf16>) -> tensor<2x7x3x5xf16>
  func.return %0 : tensor<2x7x3x5xf16>
}

// -----

// CHECK-LABEL:   @avgpool_3d_ncdhw_format
// CHECK-SAME:      [[ARG:%.+]]: tensor<2x7x4x12x21xf16>
// CHECK:           [[CONV32:%.+]] = mhlo.convert(%arg0) : (tensor<2x7x4x12x21xf16>) -> tensor<2x7x4x12x21xf32>
// CHECK:           [[ZERO:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[DIVIDEND:%.+]] = "mhlo.reduce_window"([[CONV32]], [[ZERO]]) ({
// CHECK:           ^bb0([[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<f32>):
// CHECK:             [[ADD:%.+]] = mhlo.add [[ARG1]], [[ARG2]]
// CHECK:             "mhlo.return"([[ADD]])
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 1, 2, 2]>
// CHECK-SAME:        window_strides = dense<[1, 1, 1, 4, 4]>
// CHECK-SAME:        -> tensor<2x7x4x3x5xf32>
// CHECK:           [[COUNT:%.+]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           [[DIV_RESULT:%.+]] = chlo.broadcast_divide [[DIVIDEND]], [[COUNT]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<2x7x4x3x5xf32>
// CHECK:           [[CONV16:%.+]] = mhlo.convert([[DIV_RESULT]])
// CHECK-SAME:        -> tensor<2x7x4x3x5xf16>
// CHECK:           return [[CONV16]]
func.func @avgpool_3d_ncdhw_format(%arg0: tensor<2x7x4x12x21xf16>) -> tensor<2x7x4x3x5xf16> {
  %0 = "tf.AvgPool3D"(%arg0) {data_format = "NCDHW", ksize = [1, 1, 1, 2, 2], padding = "VALID", strides = [1, 1, 1, 4, 4]} : (tensor<2x7x4x12x21xf16>) -> tensor<2x7x4x3x5xf16>
  func.return %0 : tensor<2x7x4x3x5xf16>
}

// -----

// CHECK-LABEL:   @avgpool_same_padding(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x12x21x7xf32>) -> tensor<2x4x6x7xf32>
// CHECK:           %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[DIVIDEND:.*]] = "mhlo.reduce_window"(%[[ARG0]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [1, 1], [0, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 5, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 3, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x6x7xf32>
// CHECK:           %[[ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x12x21x7xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [1, 1], [0, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 5, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 3, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x6x7xf32>
// CHECK:           %[[RESULT:.*]] = mhlo.divide %[[DIVIDEND]], %[[DIVISOR]] : tensor<2x4x6x7xf32>
// CHECK:           return %[[RESULT]] : tensor<2x4x6x7xf32>
// CHECK:         }
func.func @avgpool_same_padding(%arg0: tensor<2x12x21x7xf32>) -> tensor<2x4x6x7xf32> {
  %0 = "tf.AvgPool"(%arg0) {data_format = "NHWC", ksize = [1, 5, 2, 1], padding = "SAME", strides = [1, 3, 4, 1]} : (tensor<2x12x21x7xf32>) -> tensor<2x4x6x7xf32>
  func.return %0 : tensor<2x4x6x7xf32>
}

// -----

// CHECK-LABEL:   @avgpool_3d_same_padding(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x4x12x21x7xf32>) -> tensor<2x4x4x6x7xf32>
// CHECK:           %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[DIVIDEND:.*]] = "mhlo.reduce_window"(%[[ARG0]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [0, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 1, 5, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 1, 3, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x4x6x7xf32>
// CHECK:           %[[ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x4x12x21x7xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 0], [1, 1], [0, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 1, 5, 2, 1]>
// CHECK-SAME:        window_strides = dense<[1, 1, 3, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x4x6x7xf32>
// CHECK:           %[[RESULT:.*]] = mhlo.divide %[[DIVIDEND]], %[[DIVISOR]]
// CHECK:           return %[[RESULT]] : tensor<2x4x4x6x7xf32>
// CHECK:         }
func.func @avgpool_3d_same_padding(%arg0: tensor<2x4x12x21x7xf32>) -> tensor<2x4x4x6x7xf32> {
  %0 = "tf.AvgPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 5, 2, 1], padding = "SAME", strides = [1, 1, 3, 4, 1]} : (tensor<2x4x12x21x7xf32>) -> tensor<2x4x4x6x7xf32>
  func.return %0 : tensor<2x4x4x6x7xf32>
}

//===----------------------------------------------------------------------===//
// AvgPoolGrad op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL:   @avgpool_grad_valid_padding(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<10x12x16x64xf32>) -> tensor<10x24x32x64xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[DIVISOR:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = chlo.broadcast_divide %[[OUT_GRAD]], %[[DIVISOR]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<10x12x16x64xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 1, 1, 0]>
// CHECK-SAME:        edge_padding_low = dense<[0, 1, 1, 0]>
// CHECK-SAME:        interior_padding = dense<[0, 1, 1, 0]>
// CHECK-SAME:        -> tensor<10x25x33x64xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 2, 2, 1]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<10x24x32x64xf32>
// CHECK:           return %[[RESULT]] : tensor<10x24x32x64xf32>
func.func @avgpool_grad_valid_padding(%grad: tensor<10x12x16x64xf32>) -> tensor<10x24x32x64xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[10, 24, 32, 64]> : tensor<4xi32>} : () -> (tensor<4xi32>)
  %result = "tf.AvgPoolGrad"(%orig_input_shape, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<4xi32>, tensor<10x12x16x64xf32>) -> tensor<10x24x32x64xf32>
  func.return %result : tensor<10x24x32x64xf32>
}

// -----

// CHECK-LABEL:   @avgpool_3d_grad_valid_padding(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<10x8x12x16x64xf32>) -> tensor<10x8x24x32x64xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[DIVISOR:.*]] = mhlo.constant dense<4.000000e+00> : tensor<f32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = chlo.broadcast_divide %[[OUT_GRAD]], %[[DIVISOR]] {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<10x8x12x16x64xf32>, tensor<f32>) -> tensor<10x8x12x16x64xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 0, 1, 1, 0]>
// CHECK-SAME:        edge_padding_low = dense<[0, 0, 1, 1, 0]>
// CHECK-SAME:        interior_padding = dense<[0, 0, 1, 1, 0]>
// CHECK-SAME:        -> tensor<10x8x25x33x64xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 2, 1]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<10x8x24x32x64xf32>
// CHECK:           return %[[RESULT]] : tensor<10x8x24x32x64xf32>
func.func @avgpool_3d_grad_valid_padding(%grad: tensor<10x8x12x16x64xf32>) -> tensor<10x8x24x32x64xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[10, 8, 24, 32, 64]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %result = "tf.AvgPool3DGrad"(%orig_input_shape, %grad) {
    data_format = "NDHWC",
    ksize = [1, 1, 2, 2, 1],
    padding = "VALID",
    strides = [1, 1, 2, 2, 1]} : (tensor<5xi32>, tensor<10x8x12x16x64xf32>) -> tensor<10x8x24x32x64xf32>
  func.return %result : tensor<10x8x24x32x64xf32>
}

// -----

// CHECK-LABEL:   @avgpool_grad_same_padding(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<2x4x7x9xf32>) -> tensor<2x13x25x9xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[ALL_ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x13x25x9xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ALL_ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 2, 3, 1]>
// CHECK-SAME:        window_strides = dense<[1, 4, 4, 1]>
// CHECK-SAME:        -> tensor<2x4x7x9xf32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = mhlo.divide %[[OUT_GRAD]], %[[DIVISOR]] : tensor<2x4x7x9xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 0, 1, 0]>
// CHECK-SAME:        edge_padding_low = dense<[0, 1, 1, 0]>
// CHECK-SAME:        interior_padding = dense<[0, 3, 3, 0]>
// CHECK-SAME:        -> tensor<2x14x27x9xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 2, 3, 1]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<2x13x25x9xf32>
// CHECK:           return %[[RESULT]] : tensor<2x13x25x9xf32>
func.func @avgpool_grad_same_padding(%grad: tensor<2x4x7x9xf32>) -> tensor<2x13x25x9xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[2, 13, 25, 9]> : tensor<4xi32>} : () -> (tensor<4xi32>)
  %result = "tf.AvgPoolGrad"(%orig_input_shape, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 3, 1],
     padding = "SAME",
     strides = [1, 4, 4, 1]
  } : (tensor<4xi32>, tensor<2x4x7x9xf32>) -> tensor<2x13x25x9xf32>
  func.return %result : tensor<2x13x25x9xf32>
}

// -----

// CHECK-LABEL:   @avgpool_3d_grad_same_padding(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<2x8x4x7x9xf32>) -> tensor<2x8x13x25x9xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[ALL_ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x8x13x25x9xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ALL_ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1], [0, 0]]>
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 3, 1]>
// CHECK-SAME:        window_strides = dense<[1, 1, 4, 4, 1]>
// CHECK-SAME:        -> tensor<2x8x4x7x9xf32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = mhlo.divide %[[OUT_GRAD]], %[[DIVISOR]] : tensor<2x8x4x7x9xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 0, 0, 1, 0]>
// CHECK-SAME:        edge_padding_low = dense<[0, 0, 1, 1, 0]>
// CHECK-SAME:        interior_padding = dense<[0, 0, 3, 3, 0]>
// CHECK-SAME:        -> tensor<2x8x14x27x9xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 3, 1]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<2x8x13x25x9xf32>
// CHECK:           return %[[RESULT]] : tensor<2x8x13x25x9xf32>
func.func @avgpool_3d_grad_same_padding(%grad: tensor<2x8x4x7x9xf32>) -> tensor<2x8x13x25x9xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[2, 8, 13, 25, 9]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %result = "tf.AvgPool3DGrad"(%orig_input_shape, %grad) {
    data_format = "NDHWC",
    ksize = [1, 1, 2, 3, 1],
    padding = "SAME",
    strides = [1, 1, 4, 4, 1]} : (tensor<5xi32>, tensor<2x8x4x7x9xf32>) -> tensor<2x8x13x25x9xf32>
  func.return %result : tensor<2x8x13x25x9xf32>
}

// -----

// CHECK-LABEL:   @avgpool_grad_nchw_format(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<2x9x4x7xf32>) -> tensor<2x9x13x25xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[ALL_ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x9x13x25xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ALL_ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1]]>
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 3]>
// CHECK-SAME:        window_strides = dense<[1, 1, 4, 4]>
// CHECK-SAME:        -> tensor<2x9x4x7xf32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = mhlo.divide %[[OUT_GRAD]], %[[DIVISOR]] : tensor<2x9x4x7xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 0, 0, 1]>
// CHECK-SAME:        edge_padding_low = dense<[0, 0, 1, 1]>
// CHECK-SAME:        interior_padding = dense<[0, 0, 3, 3]>
// CHECK-SAME:        -> tensor<2x9x14x27xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 2, 3]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<2x9x13x25xf32>
// CHECK:           return %[[RESULT]] : tensor<2x9x13x25xf32>
func.func @avgpool_grad_nchw_format(%grad: tensor<2x9x4x7xf32>) -> tensor<2x9x13x25xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[2, 9, 13, 25]> : tensor<4xi32>} : () -> (tensor<4xi32>)
  %result = "tf.AvgPoolGrad"(%orig_input_shape, %grad) {
     data_format = "NCHW",
     ksize = [1, 1, 2, 3],
     padding = "SAME",
     strides = [1, 1, 4, 4]
  } : (tensor<4xi32>, tensor<2x9x4x7xf32>) -> tensor<2x9x13x25xf32>
  func.return %result : tensor<2x9x13x25xf32>
}

// -----

// CHECK-LABEL:   @avgpool_3d_grad_ncdwh_format(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<2x9x8x4x7xf32>) -> tensor<2x9x8x13x25xf32> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[ALL_ONES:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2x9x8x13x25xf32>
// CHECK:           %[[DIVISOR:.*]] = "mhlo.reduce_window"(%[[ALL_ONES]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM1:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM1]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        padding = dense<{{\[\[}}0, 0], [0, 0], [0, 0], [0, 1], [1, 1]]>
// CHECK-SAME:        window_dimensions = dense<[1, 1, 1, 2, 3]>
// CHECK-SAME:        window_strides = dense<[1, 1, 1, 4, 4]>
// CHECK-SAME:        -> tensor<2x9x8x4x7xf32>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = mhlo.divide %[[OUT_GRAD]], %[[DIVISOR]] : tensor<2x9x8x4x7xf32>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 0, 0, 0, 1]>
// CHECK-SAME:        edge_padding_low = dense<[0, 0, 0, 1, 1]>
// CHECK-SAME:        interior_padding = dense<[0, 0, 0, 3, 3]>
// CHECK-SAME:        -> tensor<2x9x8x14x27xf32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT]], %[[ZERO]]) ({
// CHECK:           ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:             %[[SUM2:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM2]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 1, 1, 2, 3]>
// CHECK-SAME:        window_strides = dense<1> : tensor<5xi64>
// CHECK-SAME:        -> tensor<2x9x8x13x25xf32>
// CHECK:           return %[[RESULT]] : tensor<2x9x8x13x25xf32>
func.func @avgpool_3d_grad_ncdwh_format(%grad: tensor<2x9x8x4x7xf32>) -> tensor<2x9x8x13x25xf32> {
  %orig_input_shape = "tf.Const"() {value = dense<[2, 9, 8, 13, 25]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %result = "tf.AvgPool3DGrad"(%orig_input_shape, %grad) {
    data_format = "NCDHW",
    ksize = [1, 1, 1, 2, 3],
    padding = "SAME",
    strides = [1, 1, 1, 4, 4]} : (tensor<5xi32>, tensor<2x9x8x4x7xf32>) -> tensor<2x9x8x13x25xf32>
  func.return %result : tensor<2x9x8x13x25xf32>
}

// -----

// CHECK-LABEL:   @avgpool_grad_bf16(
// CHECK-SAME:      %[[OUT_GRAD:.*]]: tensor<10x12x16x64xbf16>) -> tensor<10x24x32x64xbf16> {
// CHECK-DAG:       %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-DAG:       %[[DIVISOR:.*]] = mhlo.constant dense<4.000000e+00> : tensor<bf16>
// CHECK:           %[[OUT_GRAD_DIVIDED:.*]] = chlo.broadcast_divide %[[OUT_GRAD]], %[[DIVISOR]]
// CHECK-SAME:        broadcast_dimensions = dense<>
// CHECK-SAME:        -> tensor<10x12x16x64xbf16>
// CHECK:           %[[REDUCE_WINDOW_INPUT:.*]] = "mhlo.pad"(%[[OUT_GRAD_DIVIDED]], %[[ZERO]])
// CHECK-SAME:        edge_padding_high = dense<[0, 1, 1, 0]>
// CHECK-SAME:        edge_padding_low = dense<[0, 1, 1, 0]>
// CHECK-SAME:        interior_padding = dense<[0, 1, 1, 0]>
// CHECK-SAME:        -> tensor<10x25x33x64xbf16>
// CHECK:           %[[REDUCE_WINDOW_INPUT_CONVERTED:.*]] = mhlo.convert(%[[REDUCE_WINDOW_INPUT]]) : (tensor<10x25x33x64xbf16>) -> tensor<10x25x33x64xf32>
// CHECK:           %[[ZERO_F32:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[RESULT:.*]] = "mhlo.reduce_window"(%[[REDUCE_WINDOW_INPUT_CONVERTED]], %[[ZERO_F32]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[SUM:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             "mhlo.return"(%[[SUM]]) : (tensor<f32>) -> ()
// CHECK:           })
// CHECK-SAME:        window_dimensions = dense<[1, 2, 2, 1]>
// CHECK-SAME:        window_strides = dense<1>
// CHECK-SAME:        -> tensor<10x24x32x64xf32>
// CHECK:           %[[RESULT_CONVERTED:.*]] = mhlo.convert(%[[RESULT]]) : (tensor<10x24x32x64xf32>) -> tensor<10x24x32x64xbf16>
// CHECK:           return %[[RESULT_CONVERTED]] : tensor<10x24x32x64xbf16>
func.func @avgpool_grad_bf16(%grad: tensor<10x12x16x64xbf16>) -> tensor<10x24x32x64xbf16> {
  %orig_input_shape = "tf.Const"() {value = dense<[10, 24, 32, 64]> : tensor<4xi32>} : () -> (tensor<4xi32>)
  %result = "tf.AvgPoolGrad"(%orig_input_shape, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<4xi32>, tensor<10x12x16x64xbf16>) -> tensor<10x24x32x64xbf16>
  func.return %result : tensor<10x24x32x64xbf16>
}

// -----

// CHECK-LABEL: xla_sharding
func.func @xla_sharding(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // CHECK-NEXT: "mhlo.custom_call"(%arg0) {call_target_name = "Sharding", mhlo.sharding = ""}
  %0 = "tf.XlaSharding"(%arg0) {_XlaSharding = "", sharding = ""} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: inplace_update_one
func.func @inplace_update_one(%arg0: tensor<8x4xf32>, %arg1: tensor<1x4xf32>, %arg2: tensor<1xi32>) -> tensor<8x4xf32> {
  // CHECK-DAG: [[CST:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[SLICE1:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE2:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-DAG: [[RESHAPE1:%.+]] = "mhlo.reshape"([[SLICE1]])
  // CHECK-DAG: [[UPDATE:%.+]] = "mhlo.dynamic_update_slice"(%arg0, [[SLICE2]], [[RESHAPE1]], [[CST]])
  %0 = "tf.InplaceUpdate"(%arg0, %arg2, %arg1) : (tensor<8x4xf32>, tensor<1xi32>, tensor<1x4xf32>) -> tensor<8x4xf32>

  // CHECK: return [[UPDATE]]
  func.return %0 : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: inplace_update_three
func.func @inplace_update_three(%arg0: tensor<8x8x4xf32>, %arg1: tensor<3x8x4xf32>, %arg2: tensor<3xi32>) -> tensor<8x8x4xf32> {
  // CHECK-DAG: [[CST:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[SLICE1:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE2:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE3:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE4:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 8, 4]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[SLICE5:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[2, 8, 4]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[SLICE6:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[3, 8, 4]> : tensor<3xi64>, start_indices = dense<[2, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[RESHAPE1:%.+]] = "mhlo.reshape"([[SLICE1]])
  // CHECK-DAG: [[RESHAPE2:%.+]] = "mhlo.reshape"([[SLICE2]])
  // CHECK-DAG: [[RESHAPE3:%.+]] = "mhlo.reshape"([[SLICE3]])
  // CHECK-DAG: [[UPDATE1:%.+]] = "mhlo.dynamic_update_slice"(%arg0, [[SLICE4]], [[RESHAPE1]], [[CST]], [[CST]])
  // CHECK-DAG: [[UPDATE2:%.+]] = "mhlo.dynamic_update_slice"([[UPDATE1]], [[SLICE5]], [[RESHAPE2]], [[CST]], [[CST]])
  // CHECK-DAG: [[UPDATE3:%.+]] = "mhlo.dynamic_update_slice"([[UPDATE2]], [[SLICE6]], [[RESHAPE3]], [[CST]], [[CST]])
  %0 = "tf.InplaceUpdate"(%arg0, %arg2, %arg1) : (tensor<8x8x4xf32>, tensor<3xi32>, tensor<3x8x4xf32>) -> tensor<8x8x4xf32>

  // CHECK:  return [[UPDATE3]] : tensor<8x8x4xf32>
  func.return %0 : tensor<8x8x4xf32>
}

// -----

// CHECK-LABEL: xla_dynamic_update_slice
func.func @xla_dynamic_update_slice(%arg0: tensor<4x16xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2xi32>) -> tensor<4x16xf32> {
  // CHECK: [[SLICE0:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE0:%.+]] = "mhlo.reshape"([[SLICE0]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[SLICE1:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE1:%.+]] = "mhlo.reshape"([[SLICE1]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[DUS:%.+]] = "mhlo.dynamic_update_slice"(%arg0, %arg1, [[RESHAPE0]], [[RESHAPE1]]) : (tensor<4x16xf32>, tensor<2x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x16xf32>
  // CHECK: return [[DUS]]
  %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<4x16xf32>, tensor<2x4xf32>, tensor<2xi32>) -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: xla_dynamic_update_slice2
func.func @xla_dynamic_update_slice2(%arg0: tensor<4xf32>, %arg1: tensor<2xf32>, %arg2: tensor<1xi32>) -> tensor<4xf32> {
  // CHECK: [[SLICE0:%.+]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE0:%.+]] = "mhlo.reshape"([[SLICE0]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[DUS:%.+]] = "mhlo.dynamic_update_slice"(%arg0, %arg1, [[RESHAPE0]]) : (tensor<4xf32>, tensor<2xf32>, tensor<i32>) -> tensor<4xf32>
  // CHECK: return [[DUS]]
  %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<2xf32>, tensor<1xi32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

//===----------------------------------------------------------------------===//
// AllToAll op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @alltoall_basic
// See https://www.tensorflow.org/api_docs/python/tf/raw_ops/AllToAll
func.func @alltoall_basic(%input: tensor<1x2xf32>) -> tensor<2x1xf32> {
  %group_assignment = "tf.Const" () {
    value = dense<[[0, 1]]> : tensor<1x2xi32>
  } : () -> tensor<1x2xi32>
  %result = "tf.AllToAll"(%input, %group_assignment) {T = f32, concat_dimension = 0 : i64, split_count = 2 : i64, split_dimension = 1 : i64} :  (tensor<1x2xf32>, tensor<1x2xi32>)  -> tensor<2x1xf32>
  // CHECK: mhlo.all_to_all
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  func.return %result : tensor<2x1xf32>
}

//===----------------------------------------------------------------------===//
// Cumsum op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @cumsum_static
// CHECK-SAME: [[X:%.*]]: tensor<4xf32>
func.func @cumsum_static(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[AXIS:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert [[X]] : tensor<4xf32>
  // CHECK: [[INIT:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.*]] = "mhlo.reduce_window"([[CONVERT_X]], [[INIT]]) ({
  // CHECK: ^bb0([[A:%.*]]: tensor<f32>, [[B:%.*]]: tensor<f32>):
  // CHECK:   [[SUM:%.*]] = mhlo.add [[A]], [[B]] : tensor<f32>
  // CHECK:   "mhlo.return"([[SUM]]) : (tensor<f32>) -> ()
  // CHECK: }) {padding = dense<{{\[\[}}3, 0]]> : tensor<1x2xi64>, window_dimensions = dense<4> : tensor<1xi64>, window_strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_REDUCE:%.*]] = mhlo.convert [[REDUCE]] : tensor<4xf32>
  // CHECK: return [[CONVERT_REDUCE]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = false, reverse = false} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @cumsum_exclusive
// CHECK-SAME: [[X:%.*]]: tensor<4xf32>
func.func @cumsum_exclusive(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[AXIS:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert [[X]] : tensor<4xf32>
  // CHECK: [[INIT:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.*]] = "mhlo.reduce_window"([[CONVERT_X]], [[INIT]]) ({
  // CHECK: ^bb0([[A:%.*]]: tensor<f32>, [[B:%.*]]: tensor<f32>):
  // CHECK:   [[SUM:%.*]] = mhlo.add [[A]], [[B]] : tensor<f32>
  // CHECK:   "mhlo.return"([[SUM]]) : (tensor<f32>) -> ()
  // CHECK: }) {padding = dense<{{\[\[}}3, 0]]> : tensor<1x2xi64>, window_dimensions = dense<4> : tensor<1xi64>, window_strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[REDUCE]], %{{.*}}) {edge_padding_high = dense<-1> : tensor<1xi64>, edge_padding_low = dense<1> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_REDUCE:%.*]] = mhlo.convert [[PAD]] : tensor<4xf32>
  // CHECK: return [[CONVERT_REDUCE]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = true, reverse = false} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @cumsum_reverse
// CHECK-SAME: [[X:%.*]]: tensor<4xf32>
func.func @cumsum_reverse(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[AXIS:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[REVERSE1:%.*]] = "mhlo.reverse"([[X]]) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert [[REVERSE1]] : tensor<4xf32>
  // CHECK: [[INIT:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.*]] = "mhlo.reduce_window"([[CONVERT_X]], [[INIT]]) ({
  // CHECK: ^bb0([[A:%.*]]: tensor<f32>, [[B:%.*]]: tensor<f32>):
  // CHECK:   [[SUM:%.*]] = mhlo.add [[A]], [[B]] : tensor<f32>
  // CHECK:   "mhlo.return"([[SUM]]) : (tensor<f32>) -> ()
  // CHECK: }) {padding = dense<{{\[\[}}3, 0]]> : tensor<1x2xi64>, window_dimensions = dense<4> : tensor<1xi64>, window_strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_REDUCE:%.*]] = mhlo.convert [[REDUCE]] : tensor<4xf32>
  // CHECK: [[REVERSE_BACK:%.*]] = "mhlo.reverse"([[CONVERT_REDUCE]]) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return [[REVERSE_BACK]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = false, reverse = true} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @cumsum_exclusive_reverse
// CHECK-SAME: [[X:%.*]]: tensor<4xf32>
func.func @cumsum_exclusive_reverse(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[AXIS:%.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: [[REVERSE1:%.*]] = "mhlo.reverse"([[X]]) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_X:%.*]] = mhlo.convert [[REVERSE1]] : tensor<4xf32>
  // CHECK: [[INIT:%.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.*]] = "mhlo.reduce_window"([[CONVERT_X]], [[INIT]]) ({
  // CHECK: ^bb0([[A:%.*]]: tensor<f32>, [[B:%.*]]: tensor<f32>):
  // CHECK:   [[SUM:%.*]] = mhlo.add [[A]], [[B]] : tensor<f32>
  // CHECK:   "mhlo.return"([[SUM]]) : (tensor<f32>) -> ()
  // CHECK: }) {padding = dense<{{\[\[}}3, 0]]> : tensor<1x2xi64>, window_dimensions = dense<4> : tensor<1xi64>, window_strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[PAD:%.*]] = "mhlo.pad"([[REDUCE]], %{{.*}}) {edge_padding_high = dense<-1> : tensor<1xi64>, edge_padding_low = dense<1> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_REDUCE:%.*]] = mhlo.convert [[PAD]] : tensor<4xf32>
  // CHECK: [[REVERSE_BACK:%.*]] = "mhlo.reverse"([[CONVERT_REDUCE]]) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return [[REVERSE_BACK]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = true, reverse = true} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @cumsum_empty
func.func @cumsum_empty(%arg0: tensor<0xf32>) -> tensor<0xf32> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // CHECK: mhlo.constant dense<> : tensor<0xf32>
  %1 = "tf.Cumsum"(%arg0, %0) : (tensor<0xf32>, tensor<i32>) -> tensor<0xf32>
  func.return %1 : tensor<0xf32>
}

// -----

// CHECK-LABEL: func @cumsum_dynamic
func.func @cumsum_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
  // CHECK: "tf.Cumsum"
  %0 = "tf.Cumsum"(%arg0, %arg1) : (tensor<?xf32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// Cumprod op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @cumprod
func.func @cumprod(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[INIT:%.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: "mhlo.reduce_window"({{.*}}, [[INIT]]) ({
  // CHECK:   mhlo.mul
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumprod"(%arg0, %0) {exclusive = false, reverse = false} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

//===----------------------------------------------------------------------===//
// Qr op legalization
//===----------------------------------------------------------------------===//

// CHECK:  func @qr([[VAL_0:%.*]]: tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>)
func.func @qr(%arg0: tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>) {
  // The tf.Qr lowering is a full algorithm that is not effective to verify with
  // FileCheck. Just verify that it converted.
  // TODO(laurenzo): Move this out of the mainline tf2xla conversion as it is
  // really only applicable to certain legacy uses.
  // CHECK-NOT: "tf.Qr"
  %0:2 = "tf.Qr"(%arg0) {full_matrices = false} : (tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>)
  func.return %0#0, %0#1 : tensor<500x100x75xf32>, tensor<500x75x75xf32>
}

//===----------------------------------------------------------------------===//
// tf.Softplus legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @softplus_f16
// CHECK-SAME: ([[FEATURES:%.*]]: tensor<8x16xf16>)
func.func @softplus_f16(%arg0: tensor<8x16xf16>) -> tensor<8x16xf16> {
  // CHECK-DAG: [[FEATURES_EXP:%.*]] = mhlo.exponential [[FEATURES]]
  // CHECK-DAG: [[EPSILON:%.*]] = mhlo.constant dense<1.220700e-04> : tensor<f16>
  // CHECK-DAG: [[EPSILON_LOG:%.*]] = mhlo.log [[EPSILON]]
  // CHECK-DAG: [[TWO:%.*]] = mhlo.constant dense<2.000000e+00> : tensor<f16>
  // CHECK:     [[THRESHOLD:%.*]] = chlo.broadcast_add [[EPSILON_LOG]], [[TWO]]
  // CHECK:     [[NEG_THRESHOLD:%.*]] = mhlo.negate [[THRESHOLD]]
  // CHECK-DAG: [[COMPARE_GT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[NEG_THRESHOLD]] {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: [[COMPARE_LT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[THRESHOLD]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[FEATURES_EXP_LOG:%.*]] = mhlo.log_plus_one [[FEATURES_EXP]]
  // CHECK:     [[ELSE_SELECT:%.*]] = "mhlo.select"([[COMPARE_LT]], [[FEATURES_EXP]], [[FEATURES_EXP_LOG]])
  // CHECK:     [[ENTRY_SELECT:%.*]] = "mhlo.select"([[COMPARE_GT]], [[FEATURES]], [[ELSE_SELECT]])
  %0 = "tf.Softplus"(%arg0) : (tensor<8x16xf16>) -> tensor<8x16xf16>

  // CHECK:     return [[ENTRY_SELECT]] : tensor<8x16xf16>
  func.return %0 : tensor<8x16xf16>
}

// -----

// CHECK-LABEL: func @softplus_bf16
// CHECK-SAME: ([[FEATURES:%.*]]: tensor<8x16xbf16>)
func.func @softplus_bf16(%arg0: tensor<8x16xbf16>) -> tensor<8x16xbf16> {
  // CHECK-DAG: [[FEATURES_EXP:%.*]] = mhlo.exponential [[FEATURES]]
  // CHECK-DAG: [[EPSILON:%.*]] = mhlo.constant dense<7.812500e-03> : tensor<bf16>
  // CHECK-DAG: [[EPSILON_LOG:%.*]] = mhlo.log [[EPSILON]]
  // CHECK-DAG: [[TWO:%.*]] = mhlo.constant dense<2.000000e+00> : tensor<bf16>
  // CHECK:     [[THRESHOLD:%.*]] = chlo.broadcast_add [[EPSILON_LOG]], [[TWO]]
  // CHECK:     [[NEG_THRESHOLD:%.*]] = mhlo.negate [[THRESHOLD]]
  // CHECK-DAG: [[COMPARE_GT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[NEG_THRESHOLD]] {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: [[COMPARE_LT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[THRESHOLD]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[FEATURES_EXP_LOG:%.*]] = mhlo.log_plus_one [[FEATURES_EXP]]
  // CHECK:     [[ELSE_SELECT:%.*]] = "mhlo.select"([[COMPARE_LT]], [[FEATURES_EXP]], [[FEATURES_EXP_LOG]])
  // CHECK:     [[ENTRY_SELECT:%.*]] = "mhlo.select"([[COMPARE_GT]], [[FEATURES]], [[ELSE_SELECT]])
  %0 = "tf.Softplus"(%arg0) : (tensor<8x16xbf16>) -> tensor<8x16xbf16>

  // CHECK:     return [[ENTRY_SELECT]] : tensor<8x16xbf16>
  func.return %0 : tensor<8x16xbf16>
}

// -----

// CHECK-LABEL: func @softplus_f32
// CHECK-SAME: ([[FEATURES:%.*]]: tensor<8x16xf32>)
func.func @softplus_f32(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-DAG: [[FEATURES_EXP:%.*]] = mhlo.exponential [[FEATURES]]
  // CHECK-DAG: [[EPSILON:%.*]] = mhlo.constant dense<1.1920929E-7> : tensor<f32>
  // CHECK-DAG: [[EPSILON_LOG:%.*]] = mhlo.log [[EPSILON]]
  // CHECK-DAG: [[TWO:%.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK:     [[THRESHOLD:%.*]] = chlo.broadcast_add [[EPSILON_LOG]], [[TWO]]
  // CHECK:     [[NEG_THRESHOLD:%.*]] = mhlo.negate [[THRESHOLD]]
  // CHECK-DAG: [[COMPARE_GT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[NEG_THRESHOLD]] {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: [[COMPARE_LT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[THRESHOLD]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[FEATURES_EXP_LOG:%.*]] = mhlo.log_plus_one [[FEATURES_EXP]]
  // CHECK:     [[ELSE_SELECT:%.*]] = "mhlo.select"([[COMPARE_LT]], [[FEATURES_EXP]], [[FEATURES_EXP_LOG]])
  // CHECK:     [[ENTRY_SELECT:%.*]] = "mhlo.select"([[COMPARE_GT]], [[FEATURES]], [[ELSE_SELECT]])
  %0 = "tf.Softplus"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK:     return [[ENTRY_SELECT]] : tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func @softplus_f64
// CHECK-SAME: ([[FEATURES:%.*]]: tensor<8x16xf64>)
func.func @softplus_f64(%arg0: tensor<8x16xf64>) -> tensor<8x16xf64> {
  // CHECK-DAG: [[FEATURES_EXP:%.*]] = mhlo.exponential [[FEATURES]]
  // CHECK-DAG: [[EPSILON:%.*]] = mhlo.constant dense<2.2204460492503131E-16> : tensor<f64>
  // CHECK-DAG: [[EPSILON_LOG:%.*]] = mhlo.log [[EPSILON]]
  // CHECK-DAG: [[TWO:%.*]] = mhlo.constant dense<2.000000e+00> : tensor<f64>
  // CHECK:     [[THRESHOLD:%.*]] = chlo.broadcast_add [[EPSILON_LOG]], [[TWO]]
  // CHECK:     [[NEG_THRESHOLD:%.*]] = mhlo.negate [[THRESHOLD]]
  // CHECK-DAG: [[COMPARE_GT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[NEG_THRESHOLD]] {comparison_direction = #mhlo<comparison_direction GT>}
  // CHECK-DAG: [[COMPARE_LT:%.*]] = chlo.broadcast_compare [[FEATURES]], [[THRESHOLD]] {comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-DAG: [[FEATURES_EXP_LOG:%.*]] = mhlo.log_plus_one [[FEATURES_EXP]]
  // CHECK:     [[ELSE_SELECT:%.*]] = "mhlo.select"([[COMPARE_LT]], [[FEATURES_EXP]], [[FEATURES_EXP_LOG]])
  // CHECK:     [[ENTRY_SELECT:%.*]] = "mhlo.select"([[COMPARE_GT]], [[FEATURES]], [[ELSE_SELECT]])
  %0 = "tf.Softplus"(%arg0) : (tensor<8x16xf64>) -> tensor<8x16xf64>

  // CHECK:     return [[ENTRY_SELECT]] : tensor<8x16xf64>
  func.return %0 : tensor<8x16xf64>
}

// -----

// CHECK-LABEL: @xla_gather
func.func @xla_gather(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<1x300x10xf32> {
  %cst = "tf.Const"() { value = dense<[1, 1, 300]> : tensor<3xi64> } : () -> tensor<3xi64>

  // CHECK: "mhlo.gather"
  // CHECK-SAME: dimension_numbers =
  // CHECK-SAME:   offset_dims = [0, 1]
  // CHECK-SAME:   collapsed_slice_dims = [0]
  // CHECK-SAME:   start_index_map = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: indices_are_sorted = true
  // CHECK-SAME: slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>

  %0 = "tf.XlaGather"(%arg0, %arg1, %cst) {dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01\20\01", indices_are_sorted = true} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
  func.return %0 : tensor<1x300x10xf32>
}

// -----

// CHECK-LABEL: @xla_gather_i32
func.func @xla_gather_i32(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<1x300x10xf32> {
  %cst = "tf.Const"() { value = dense<[1, 1, 300]> : tensor<3xi32> } : () -> tensor<3xi32>

  // CHECK: "mhlo.gather"
  // CHECK-SAME: dimension_numbers =
  // CHECK-SAME:   offset_dims = [0, 1]
  // CHECK-SAME:   collapsed_slice_dims = [0]
  // CHECK-SAME:   start_index_map = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: indices_are_sorted = true
  // CHECK-SAME: slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>

  %0 = "tf.XlaGather"(%arg0, %arg1, %cst) {dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01\20\01", indices_are_sorted = true} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi32>) -> tensor<1x300x10xf32>
  func.return %0 : tensor<1x300x10xf32>
}


// CHECK: func @stridedslice_with_i32
func.func @stridedslice_with_i32(%arg0: tensor<i32>) -> tensor<4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "const_0_arg", outputs = "identity_0_retval_RetVal"}} {
// CHECK-NOT: tf.StridedSlice
// CHECK: [[DYNSLICE:%.*]] = "mhlo.dynamic_slice
// CHECK: [[RESHAPE:%.*]] = "mhlo.reshape"([[DYNSLICE]])
// CHECK: return [[RESHAPE]]
  %0 = "tf.Const"() {value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.AddV2"(%arg0, %1) {_xla_inferred_shapes = [#tf_type.shape<>], device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.Pack"(%3) {_xla_inferred_shapes = [#tf_type.shape<1>], axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %5 = "tf.Pack"(%arg0) {_xla_inferred_shapes = [#tf_type.shape<1>], axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %6 = "tf.StridedSlice"(%0, %5, %4, %2) {_xla_inferred_shapes = [#tf_type.shape<4>], begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2x4xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xf32>
  func.return %6 : tensor<4xf32>
}

func.func @replica_id() -> tensor<i32> {
  // CHECK: %[[ID:.*]] = "mhlo.replica_id"() : () -> tensor<ui32>
  // CHECK: %[[RESULT:.*]] = mhlo.convert(%0) : (tensor<ui32>) -> tensor<i32>
  %0 = "tf.XlaReplicaId"() : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK: func @angle_c64
// CHECK-SAME: ([[ARG0:%.*]]: tensor<complex<f32>>)
func.func @angle_c64(%arg0: tensor<complex<f32>>) -> tensor<f32> {
// CHECK: [[IMAG:%.*]] = mhlo.imag([[ARG0]])
// CHECK: [[REAL:%.*]] = mhlo.real([[ARG0]])
// CHECK: [[ATAN2:%.*]] = mhlo.atan2 [[IMAG]], [[REAL]]
  %0 = "tf.Angle"(%arg0): (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

//===----------------------------------------------------------------------===//
// tf.ApproximateEqual legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @approximateequal_f64
func.func @approximateequal_f64(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xi1> {
  // CHECK: %[[SUB:.*]] = mhlo.subtract %arg0, %arg1 : tensor<?xf64>
  // CHECK: %[[ABS:.*]] = mhlo.abs %[[SUB]] : tensor<?xf64>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK: %[[CONVERT:.*]] = mhlo.convert(%[[CST]]) : (tensor<f32>) -> tensor<f64>
  // CHECK: %[[LE:.*]] = chlo.broadcast_compare %[[ABS]], %[[CONVERT]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?xf64>, tensor<f64>) -> tensor<?xi1>
  // CHECK: return %[[LE]] : tensor<?xi1>
  %equal = "tf.ApproximateEqual"(%arg0, %arg1) { tolerance = 2. : f32 } : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xi1>
  func.return %equal : tensor<?xi1>
}

// CHECK-LABEL: func @approximateequal_i32
func.func @approximateequal_i32(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  // CHECK: %[[SUB:.*]] = mhlo.subtract %arg0, %arg1 : tensor<?xi32>
  // CHECK: %[[ABS:.*]] = mhlo.abs %[[SUB]] : tensor<?xi32>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK: %[[CONVERT:.*]] = mhlo.convert(%[[CST]]) : (tensor<f32>) -> tensor<i32>
  // CHECK: %[[LE:.*]] = chlo.broadcast_compare %[[ABS]], %[[CONVERT]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
  // CHECK: return %[[LE]] : tensor<?xi1>
  %equal = "tf.ApproximateEqual"(%arg0, %arg1) { tolerance = 2. : f32 } : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  func.return %equal : tensor<?xi1>
}

// CHECK-LABEL: func @approximateequal_complex64
func.func @approximateequal_complex64(%arg0: tensor<?xcomplex<f32>>, %arg1: tensor<?xcomplex<f32>>) -> tensor<?xi1> {
  // CHECK: %[[SUB:.*]] = mhlo.subtract %arg0, %arg1 : tensor<?xcomplex<f32>>
  // CHECK: %[[ABS:.*]] = mhlo.abs(%[[SUB]]) : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK: %[[CONVERT:.*]] = mhlo.convert %[[CST]] : tensor<f32>
  // CHECK: %[[LE:.*]] = chlo.broadcast_compare %[[ABS]], %[[CONVERT]] {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>
  // CHECK: return %[[LE]] : tensor<?xi1>
  %equal = "tf.ApproximateEqual"(%arg0, %arg1) { tolerance = 2. : f32 } : (tensor<?xcomplex<f32>>, tensor<?xcomplex<f32>>) -> tensor<?xi1>
  func.return %equal : tensor<?xi1>
}

//===----------------------------------------------------------------------===//
// tf.XlaDot legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @xladot_matmul(
// CHECK-SAME:    %[[LHS:.*]]: tensor<64x32xi8>, %[[RHS:.*]]: tensor<32x16xi8>) -> tensor<64x16xi32>
func.func @xladot_matmul(%lhs : tensor<64x32xi8>, %rhs : tensor<32x16xi8>) -> tensor<64x16xi32> {
  // CHECK: "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {
  // CHECK-SAME:  dot_dimension_numbers = #mhlo.dot<
  // CHECK-NOT:     lhs_batching_dimensions =
  // CHECK-NOT:     rhs_batching_dimensions =
  // CHECK-SAME:    lhs_contracting_dimensions = [1]
  // CHECK-SAME:    rhs_contracting_dimensions = [0]
  // CHECK-SAME:  precision_config = []
  %res = "tf.XlaDot"(%lhs, %rhs) {dimension_numbers = "\0A\01\01\12\01\00", precision_config = ""} : (tensor<64x32xi8>, tensor<32x16xi8>) -> tensor<64x16xi32>
  func.return %res : tensor<64x16xi32>
}

//===----------------------------------------------------------------------===//
// tf.XlaDotV2 legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @xladotv2_matmul(
// CHECK-SAME:    %[[LHS:.*]]: tensor<64x32xi8>, %[[RHS:.*]]: tensor<32x16xi8>) -> tensor<64x16xi32>
func.func @xladotv2_matmul(%lhs : tensor<64x32xi8>, %rhs : tensor<32x16xi8>) -> tensor<64x16xi32> {
  // CHECK: "mhlo.dot_general"(%[[LHS]], %[[RHS]]) {
  // CHECK-SAME:  dot_dimension_numbers = #mhlo.dot<
  // CHECK-NOT:     lhs_batching_dimensions =
  // CHECK-NOT:     rhs_batching_dimensions =
  // CHECK-SAME:    lhs_contracting_dimensions = [1]
  // CHECK-SAME:    rhs_contracting_dimensions = [0]
  // CHECK-SAME:  precision_config = []
  %res = "tf.XlaDotV2"(%lhs, %rhs) {dimension_numbers = "\0A\01\01\12\01\00", precision_config = ""} : (tensor<64x32xi8>, tensor<32x16xi8>) -> tensor<64x16xi32>
  func.return %res : tensor<64x16xi32>
}

//===----------------------------------------------------------------------===//
// tf.XlaDynamicSlice legalization
//===----------------------------------------------------------------------===//
// -----

// CHECK-LABEL: xla_dynamic_slice_constant_start
func.func @xla_dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = mhlo.constant dense<1> : tensor<i64>
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[RESULT:.*]] = "mhlo.dynamic_slice"(%arg0, %[[START]])
  // CHECK-DAG-SAME: {slice_sizes = dense<2> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.XlaDynamicSlice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: xla_dynamic_slice_i32_consts
func.func @xla_dynamic_slice_i32_consts(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK: "mhlo.dynamic_slice"(%arg0, %[[START]]) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i32>) -> tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %0 = "tf.XlaDynamicSlice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: xla_dynamic_slice_constant_start_dynamic_shape
func.func @xla_dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK-DAG: %[[START1:.*]] = mhlo.constant dense<1> : tensor<i64>
  // CHECK-DAG: %[[START2:.*]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"
  // CHECK-DAG-SAME: (%arg0, %[[START1]], %[[START2]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 4]> : tensor<2xi64>} :
  // CHECK-DAG-SAME: (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %starts = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.XlaDynamicSlice"(%arg0, %starts, %sizes) : (tensor<?x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: xla_dynamic_slice_variable_start
func.func @xla_dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[SLICED_START1:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START1:.*]] = "mhlo.reshape"(%[[SLICED_START1]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[SLICED_START2:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<2> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START2:.*]] = "mhlo.reshape"(%[[SLICED_START2]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"(%arg0, %[[RESHAPED_START1]], %[[RESHAPED_START2]]) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.XlaDynamicSlice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: xla_dynamic_slice_mhlo_sizes
func.func @xla_dynamic_slice_mhlo_sizes(%arg0: tensor<1x1024x4xf32>, %arg1: tensor<3xi32>) -> tensor<1x512x4xf32> {
  // CHECK-NOT: "tf.XlaDynamicSlice"
  %0 = "mhlo.constant"() {value = dense<[1, 512, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.XlaDynamicSlice"(%arg0, %arg1, %0) : (tensor<1x1024x4xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x512x4xf32>
  func.return %1 : tensor<1x512x4xf32>
}

//===----------------------------------------------------------------------===//
// tf.XlaEinsum legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @xlaeinsum
func.func @xlaeinsum(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK-NEXT:  mhlo.einsum
  %0 = "tf.XlaEinsum"(%arg0, %arg1) {equation = "ab,bc->ac"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return %0: tensor<2x4xf32>
}

//===----------------------------------------------------------------------===//
// tf.XlaSort legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @xlasort_int
// CHECK-SAME: %[[INPUT:.*]]: tensor<16xi32>
func.func @xlasort_int(%input: tensor<16xi32>) -> (tensor<16xi32>) {
  // CHECK-NEXT: %[[SORT:.*]] = "mhlo.sort"(%[[INPUT]]) ({
  // CHECK-NEXT: ^{{.*}}(%[[LHS:.*]]: tensor<i32>, %[[RHS:.*]]: tensor<i32>)
  // CHECK-NEXT:   %[[CMP:.*]] = "mhlo.compare"(%[[LHS]], %[[RHS]]) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-NEXT:   "mhlo.return"(%[[CMP]])
  // CHECK-NEXT: }) {dimension = -1 : i64, is_stable = false} : (tensor<16xi32>) -> tensor<16xi32>
  // CHECK-NEXT: return %[[SORT]]
  %output = "tf.XlaSort"(%input) : (tensor<16xi32>) -> (tensor<16xi32>)
  func.return %output : tensor<16xi32>
}

// -----

// CHECK-LABEL: @xlasort_float
// CHECK-SAME: %[[INPUT:.*]]: tensor<8xf64>
func.func @xlasort_float(%input: tensor<8xf64>) -> (tensor<8xf64>) {
  // CHECK-NEXT: %[[SORT:.*]] = "mhlo.sort"(%[[INPUT]]) ({
  // CHECK-NEXT: ^{{.*}}(%[[LHS:.*]]: tensor<f64>, %[[RHS:.*]]: tensor<f64>)
  // CHECK-NEXT:   %[[CMP:.*]] = "mhlo.compare"(%[[LHS]], %[[RHS]]) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction LT>}
  // CHECK-NEXT:   "mhlo.return"(%[[CMP]])
  // CHECK-NEXT: }) {dimension = -1 : i64, is_stable = false} : (tensor<8xf64>) -> tensor<8xf64>
  // CHECK-NEXT: return %[[SORT]]
  %output = "tf.XlaSort"(%input) : (tensor<8xf64>) -> (tensor<8xf64>)
  func.return %output : tensor<8xf64>
}

// -----

// CHECK-LABEL: @xlasort_const
func.func @xlasort_const() -> (tensor<2x3xi64>) {
  // CHECK: [2, 4, 3], [6, 5, 1]
  %input = "tf.Const"() {value = dense<[[2, 4, 3], [6, 5, 1]]> : tensor<2x3xi64>} : () -> (tensor<2x3xi64>)
  // CHECK-NEXT: [2, 3, 4], [1, 5, 6]
  %output = "tf.XlaSort"(%input): (tensor<2x3xi64>) -> (tensor<2x3xi64>)
  func.return %output : tensor<2x3xi64>
}

//===----------------------------------------------------------------------===//
// tf.NextAfter legalization
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nextafter
func.func @nextafter(%arg0: tensor<2xf32>, %arg1 : tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:  %0 = chlo.broadcast_next_after %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT:  return %0 : tensor<2xf32>
  %0 = "tf.NextAfter"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0: tensor<2xf32>
}

//===----------------------------------------------------------------------===//
// tf.XlaReduceScatter legalization
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @xla_reduce_scatter
func.func @xla_reduce_scatter(%arg0: tensor<128x128xf32>) -> tensor<64x128xf32> {
    %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    // CHECK:          "mhlo.reduce_scatter"(%arg0)
    // CHECK{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
    // CHECK-SAME:     scatter_dimension = 0
    //
    %1 = "tf.XlaReduceScatter"(%arg0, %cst_0, %cst) {reduce_op = "Add"} : (tensor<128x128xf32>, tensor<4x2xi32>, tensor<i32>) -> tensor<64x128xf32>
    func.return %1 : tensor<64x128xf32>
}
