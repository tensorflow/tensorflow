// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion legalize-chlo=false" %s | FileCheck %s --dump-input-on-failure
// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion legalize-chlo=true" -verify-diagnostics %s
// This test runs twice:
//   1. Through FileCheck with chlo legalization disabled since verifying
//      that the chlo ops emit produces more useful tests.
//   2. With chlo legalization enabled, verifying diagnostics to pick up any
//      issues with the full lowering (can catch some broadcasting corner
//      cases which emit with a warning).

//===----------------------------------------------------------------------===//
// BatchNorm op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: fusedBatchNorm_notraining
func @fusedBatchNorm_notraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "xla_hlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNorm_training
func @fusedBatchNorm_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // TODO(riverriddle) Support training.
  // CHECK: "tf.FusedBatchNorm"
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true}  : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormV3_noTraining
func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "xla_hlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormV3_noTraining_mixedPrecision
// CHECK-SAME:  ([[X:%.*]]: tensor<8x8x8x8xbf16>, [[SCALE:%.*]]: tensor<8xf32>, [[OFFSET:%.*]]: tensor<8xf32>, [[MEAN:%.*]]: tensor<8xf32>, [[VARIANCE:%.*]]: tensor<8xf32>)
func @fusedBatchNormV3_noTraining_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) {
  // CHECK: [[CONVERT_X:%.*]] = "xla_hlo.convert"([[X]]) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK: [[Y:%.*]] = "xla_hlo.batch_norm_inference"([[CONVERT_X]], [[SCALE]], [[OFFSET]], [[MEAN]], [[VARIANCE]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)
  // CHECK: [[Y_CONVERT:%.*]] = "xla_hlo.convert"([[Y]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK: [[DUMMY:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<0xf32>
  // CHECK: [[DUMMY_CAST:%.*]] = tensor_cast [[DUMMY]] : tensor<0xf32> to tensor<*xf32>
  // CHECK: return [[Y_CONVERT]], [[MEAN]], [[VARIANCE]], [[MEAN]], [[VARIANCE]], [[DUMMY_CAST]]
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>
}

// CHECK-LABEL: fusedBatchNormV3_training
func @fusedBatchNormV3_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[RESULT0:.*]] = "xla_hlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK: "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK: %[[VAR:.*]] = "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK: xla_hlo.constant
  // CHECK: xla_chlo.broadcast_multiply %[[VAR]], {{.*}} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @fusedBatchNormV3_training_batchVariance
func @fusedBatchNormV3_training_batchVariance(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[RESULT0:.*]] = "xla_hlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: %[[VAR:.*]] = "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK: return %[[VAR]]
  return %0#4 : tensor<8xf32>
}

// CHECK-LABEL: fusedBatchNormV3_training_exponentialAvgFactor
func @fusedBatchNormV3_training_exponentialAvgFactor(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  // CHECK: %[[RESULT0:.*]] = "xla_hlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 0.8 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-DAG: %[[BATCH_MEAN:.*]] = "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 1 : i32}
  // CHECK-DAG: %[[BATCH_VAR:.*]] = "xla_hlo.get_tuple_element"(%[[RESULT0]]) {index = 2 : i32}

  // CHECK: %[[FACTOR:.*]] = xla_hlo.constant dense<1.00195694>
  // CHECK: %[[CORRECTED_VAR:.*]] = xla_chlo.broadcast_multiply %[[BATCH_VAR]], %[[FACTOR]]

  // CHECK-DAG: %[[ALPHA:.*]] = xla_hlo.constant dense<0.199999988>
  // CHECK-DAG: %[[BETA:.*]] = xla_hlo.constant dense<8.000000e-01>

  // CHECK: %[[ALPHA_MUL_OLD_MEAN:.*]] = xla_chlo.broadcast_multiply %[[ALPHA]], %arg3
  // CHECK: %[[BETA_MUL_BATCH_MEAN:.*]] = xla_chlo.broadcast_multiply %[[BETA]], %[[BATCH_MEAN]]
  // CHECK: %[[NEW_BATCH_MEAN:.*]] = xla_chlo.broadcast_add %[[ALPHA_MUL_OLD_MEAN]], %[[BETA_MUL_BATCH_MEAN]]

  // CHECK: %[[ALPHA_MUL_OLD_VAR:.*]] = xla_chlo.broadcast_multiply %[[ALPHA]], %arg4
  // CHECK: %[[BETA_MUL_CORRECTED_VAR:.*]] = xla_chlo.broadcast_multiply %[[BETA]], %[[CORRECTED_VAR]]
  // CHECK: %[[NEW_BATCH_VAR:.*]] = xla_chlo.broadcast_add %[[ALPHA_MUL_OLD_VAR]], %[[BETA_MUL_CORRECTED_VAR]]

  // CHECK: return %[[NEW_BATCH_MEAN]], %[[NEW_BATCH_VAR]], %[[BATCH_MEAN]], %[[BATCH_VAR]]
  return %0#1, %0#2, %0#3, %0#4 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: fusedBatchNormV3_training_mixedPrecision
func @fusedBatchNormV3_training_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK: "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: "xla_hlo.convert"({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  return %0#0 : tensor<8x8x8x8xbf16>
}

// CHECK-LABEL: fusedBatchNormV3_NCHW
func @fusedBatchNormV3_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "xla_hlo.batch_norm_training"({{.*}}, %arg1, %arg2) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormV3_noTraining_dynamic_supported
func @fusedBatchNormV3_noTraining_dynamic_supported(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>) -> (tensor<?x?x?x?xf32>) {
  // CHECK: "xla_hlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
  return %0#0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: fusedBatchNormV3_training_dynamic_unsupported1
func @fusedBatchNormV3_training_dynamic_unsupported1(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: tensor<?xf32>, %arg4: tensor<?xf32>) -> (tensor<?x?x?x?xf32>) {
  // CHECK: tf.FusedBatchNormV3
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
  return %0#0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: fusedBatchNormV3_training_dynamic_unsupported2
func @fusedBatchNormV3_training_dynamic_unsupported2(%arg0: tensor<?x6x?x?xf32>, %arg1: tensor<6xf32>, %arg2: tensor<6xf32>, %arg3: tensor<6xf32>, %arg4: tensor<6xf32>) -> (tensor<?x6x?x?xf32>) {
  // CHECK: tf.FusedBatchNormV3
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<?x6x?x?xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> (tensor<?x6x?x?xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>)
  return %0#0 : tensor<?x6x?x?xf32>
}

// CHECK-LABEL: fusedBatchNormGrad_noTraining
func @fusedBatchNormGrad_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = xla_hlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = xla_chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = "xla_hlo.rsqrt"(%[[add]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = xla_hlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = xla_hlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = "xla_hlo.convert"(%[[mul]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = "xla_hlo.reduce"(%[[cmul]], %[[init]]) ( {
  // CHECK-NEXT: ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced:.*]] = xla_hlo.add %arg5, %arg6 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = "xla_hlo.convert"(%[[red1]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = xla_hlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = xla_hlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = xla_hlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = "xla_hlo.convert"(%[[grad]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = "xla_hlo.reduce"(%[[cgrad]], %[[init2]]) ( {
  // CHECK-NEXT: ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced1:.*]] = xla_hlo.add %arg5, %arg6 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.convert"(%[[red2]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[mul3]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGrad_Training
func @fusedBatchNormGrad_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[training:.*]] = "xla_hlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  // CHECK-NEXT: %[[tact:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[tact]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV2_noTraining
func @fusedBatchNormGradV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = xla_hlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = xla_chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = "xla_hlo.rsqrt"(%[[add]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = xla_hlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = xla_hlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = "xla_hlo.convert"(%[[mul]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = "xla_hlo.reduce"(%[[cmul]], %[[init]]) ( {
  // CHECK-NEXT: ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced:.*]] = xla_hlo.add %arg5, %arg6 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = "xla_hlo.convert"(%[[red1]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = xla_hlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = xla_hlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = xla_hlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = "xla_hlo.convert"(%[[grad]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = "xla_hlo.reduce"(%[[cgrad]], %[[init2]]) ( {
  // CHECK-NEXT: ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced1:.*]] = xla_hlo.add %arg5, %arg6 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.convert"(%[[red2]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[mul3]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV2_Training
func @fusedBatchNormGradV2_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[training:.*]] = "xla_hlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  // CHECK-NEXT: %[[tact:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[tact]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV2_noTraining_mixed_precision
func @fusedBatchNormGradV2_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = "xla_hlo.convert"({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xbf16>
}

// CHECK-LABEL: fusedBatchNormGradV2_Training_mixed_precision
func @fusedBatchNormGradV2_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[training:.*]] = "xla_hlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  // CHECK-NEXT: %[[tact:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[tact]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xbf16>
}

// CHECK-LABEL: fusedBatchNormGradV3_noTraining
func @fusedBatchNormGradV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = xla_hlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = xla_chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = "xla_hlo.rsqrt"(%[[add]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = xla_hlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = xla_hlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = "xla_hlo.convert"(%[[mul]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = "xla_hlo.reduce"(%[[cmul]], %[[init]]) ( {
  // CHECK-NEXT: ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced:.*]] = xla_hlo.add %arg6, %arg7 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = "xla_hlo.convert"(%[[red1]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = xla_hlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = xla_hlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = xla_hlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: xla_hlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = "xla_hlo.convert"(%[[grad]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = "xla_hlo.reduce"(%[[cgrad]], %[[init2]]) ( {
  // CHECK-NEXT: ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced1:.*]] = xla_hlo.add %arg6, %arg7 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.convert"(%[[red2]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[mul3]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV3_Training
func @fusedBatchNormGradV3_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[training:.*]] = "xla_hlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  // CHECK-NEXT: %[[tact:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[tact]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_mixed_precision
func @fusedBatchNormGradV3_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>

  // CHECK: %[[x_backprop:.*]] = "xla_hlo.convert"({{.*}}) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xbf16>
}

// CHECK-LABEL: fusedBatchNormGradV3_Training_mixed_precision
func @fusedBatchNormGradV3_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[training:.*]] = "xla_hlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  // CHECK-NEXT: %[[tact:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 0 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[scale_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 1 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.get_tuple_element"(%[[training]]) {index = 2 : i32} : (tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>) -> tensor<8xf32>
  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[tact]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xbf16>
}

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_NCHW
func @fusedBatchNormGradV3_noTraining_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[grad:.*]] = "xla_hlo.convert"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[act:.*]] = "xla_hlo.convert"(%arg1) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[eps:.*]] = xla_hlo.constant dense<1.000000e-03> : tensor<f32>

  // CHECK-NEXT: %[[add:.*]] = xla_chlo.broadcast_add %arg4, %[[eps]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr1:.*]] = "xla_hlo.rsqrt"(%[[add]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK:      %[[bcast_arg3:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[sub:.*]] = xla_hlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul:.*]] = xla_hlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: xla_hlo.constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-NEXT: %[[cmul:.*]] = "xla_hlo.convert"(%[[mul]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red1:.*]] = "xla_hlo.reduce"(%[[cmul]], %[[init]]) ( {
  // CHECK-NEXT: ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced:.*]] = xla_hlo.add %arg6, %arg7 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[scr2:.*]] = "xla_hlo.convert"(%[[red1]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[mul2:.*]] = xla_hlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
  // CHECK:      %[[bcast_mul2:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[mul3:.*]] = xla_hlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

  // CHECK-NEXT: %[[scale_backprop:.*]] = xla_hlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

  // CHECK-NEXT: xla_hlo.constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-NEXT: %[[cgrad:.*]] = "xla_hlo.convert"(%[[grad]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[init2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[red2:.*]] = "xla_hlo.reduce"(%[[cgrad]], %[[init2]]) ( {
  // CHECK-NEXT: ^bb0(%arg6: tensor<f32>, %arg7: tensor<f32>):	// no predecessors
  // CHECK-NEXT:   %[[reduced1:.*]] = xla_hlo.add %arg6, %arg7 : tensor<f32>
  // CHECK-NEXT:   "xla_hlo.return"(%[[reduced1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[offset_backprop:.*]] = "xla_hlo.convert"(%[[red2]]) : (tensor<8xf32>) -> tensor<8xf32>

  // CHECK-NEXT: %[[x_backprop:.*]] = "xla_hlo.convert"(%[[mul3]]) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[x_backprop]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNormGradV3_Training_NCHW
func @fusedBatchNormGradV3_Training_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %{{.*}} = "xla_hlo.batch_norm_grad"(%{{.*}}, %arg2, %arg3, %arg4, %[[grad]]) {epsilon = 1.000000e-03 : f32, feature_index = 1 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> tuple<tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>>
  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

//===----------------------------------------------------------------------===//
// Bias op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @biasAdd_NHWC
func @biasAdd_NHWC(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[ARG0_SHAPE]])
  // CHECK: %[[ARG1_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<3> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = xla_hlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

// CHECK-LABEL: func @biasAdd_NCHW
func @biasAdd_NCHW(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[ARG0_SHAPE]])
  // CHECK: %[[ARG1_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = xla_hlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

// CHECK-LABEL: func @biasAdd_dynamic
func @biasAdd_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?x?xi32> {
  // CHECK: %[[ARG0_SHAPE:.+]] = shape.shape_of %arg0
  // CHECK: %[[ARG0_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[ARG0_SHAPE]])
  // CHECK: %[[ARG1_BCAST:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%arg1, %[[ARG0_EXTENTS]])
  // CHECK-SAME:   {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[RESULT:.+]] = xla_hlo.add %arg0, %[[ARG1_BCAST]]
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW"} : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}

//===----------------------------------------------------------------------===//
// DiagPart
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @diag_part
// CHECK-SAME: %[[ARG:.*]]: tensor<4x3x4x3xf32>
func @diag_part(%arg0: tensor<4x3x4x3xf32>) -> tensor<4x3xf32> {
  // CHECK: %[[RS:.*]] = "xla_hlo.reshape"(%[[ARG]]) : (tensor<4x3x4x3xf32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[IOTA0:.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[IOTA1:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<12x12xi32>
  // CHECK-DAG: %[[COMP:.*]] = "xla_hlo.compare"(%[[IOTA0]], %[[IOTA1]]) {comparison_direction = "EQ"} : (tensor<12x12xi32>, tensor<12x12xi32>) -> tensor<12x12xi1>
  // CHECK-DAG: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZERO_MAT:.*]] = "xla_hlo.broadcast"(%[[ZERO]]) {broadcast_sizes = dense<12> : tensor<2xi64>} : (tensor<f32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[SEL:.*]] = "xla_hlo.select"(%[[COMP]], %[[RS]], %[[ZERO_MAT]]) : (tensor<12x12xi1>, tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
  // CHECK-DAG: %[[RED:.*]] = "xla_hlo.reduce"(%[[SEL]], %[[ZERO]])
  // CHECK-DAG:  xla_hlo.add
  // CHECK-DAG: {dimensions = dense<0> : tensor<1xi64>} : (tensor<12x12xf32>, tensor<f32>) -> tensor<12xf32>
  // CHECK-DAG:  %[[RES:.*]] = "xla_hlo.reshape"(%[[RED]]) : (tensor<12xf32>) -> tensor<4x3xf32>
  // CHECK-DAG:  return %[[RES]] : tensor<4x3xf32>
  %0 = "tf.DiagPart"(%arg0) : (tensor<4x3x4x3xf32>) -> tensor<4x3xf32>
  return %0: tensor<4x3xf32>
}

//===----------------------------------------------------------------------===//
// Einsum.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @einsum
func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK:  xla_hlo.einsum
  %0 = "tf.Einsum"(%arg0, %arg1) {equation = "ab,bc->ac"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0: tensor<2x4xf32>
}

// CHECK-LABEL: func @unary_einsum
func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK:  xla_hlo.unary_einsum
  %0 = "tf.Einsum"(%arg0) {equation = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// FloorDiv and FloorMod.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @floordiv_broadcast_i32
func @floordiv_broadcast_i32(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[ZEROS1:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = xla_chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = "LT"}
  // CHECK-DAG: [[ZEROS2:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = xla_chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = "LT"}
  // CHECK-DAG: [[CMP3:%.+]] = xla_chlo.broadcast_compare [[CMP1]], [[CMP2]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"}
  // CHECK-DAG: [[DIV1:%.+]] = xla_chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ABS1:%.+]] = "xla_hlo.abs"(%arg0)
  // CHECK-DAG: [[ABS2:%.+]] = "xla_hlo.abs"(%arg1)
  // CHECK-DAG: [[ZEROS3:%.+]] = xla_hlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = xla_chlo.broadcast_subtract [[ABS2]], [[ZEROS3]]
  // CHECK-DAG: [[ADD:%.+]] = xla_chlo.broadcast_add [[ABS1]], [[SUB]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[NEG:%.+]] = "xla_hlo.negate"([[ADD]])
  // CHECK-DAG: [[ABS3:%.+]] = "xla_hlo.abs"(%arg1)
  // CHECK-DAG: [[DIV2:%.+]] = xla_chlo.broadcast_divide [[NEG]], [[ABS3]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SELECT:%.+]] = "xla_hlo.select"([[CMP3]], [[DIV1]], [[DIV2]])
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  return %0: tensor<2x3xi32>
}

// CHECK-LABEL: func @floordiv_reverse_broadcast_i32
func @floordiv_reverse_broadcast_i32(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[ZEROS1:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = xla_chlo.broadcast_compare %arg0, [[ZEROS1]] {comparison_direction = "LT"}
  // CHECK-DAG: [[ZEROS2:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = xla_chlo.broadcast_compare %arg1, [[ZEROS2]] {comparison_direction = "LT"}
  // CHECK-DAG: [[CMP3:%.+]] = xla_chlo.broadcast_compare [[CMP1]], [[CMP2]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"}
  // CHECK-DAG: [[DIV1:%.+]] = xla_chlo.broadcast_divide %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ABS1:%.+]] = "xla_hlo.abs"(%arg0)
  // CHECK-DAG: [[ABS2:%.+]] = "xla_hlo.abs"(%arg1)
  // CHECK-DAG: [[ZEROS3:%.+]] = xla_hlo.constant dense<1>
  // CHECK-DAG: [[SUB:%.+]] = xla_chlo.broadcast_subtract [[ABS2]], [[ZEROS3]]
  // CHECK-DAG: [[ADD:%.+]] = xla_chlo.broadcast_add [[ABS1]], [[SUB]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[NEG:%.+]] = "xla_hlo.negate"([[ADD]])
  // CHECK-DAG: [[ABS3:%.+]] = "xla_hlo.abs"(%arg1)
  // CHECK-DAG: [[DIV2:%.+]] = xla_chlo.broadcast_divide [[NEG]], [[ABS3]]
  // CHECK-DAG: [[SELECT:%.+]] = "xla_hlo.select"([[CMP3]], [[DIV1]], [[DIV2]])
  // CHECK: return [[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0: tensor<2x3xi32>
}

// CHECK-LABEL: func @floordiv_f32
func @floordiv_f32(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT:  %[[DIV:.*]] = xla_chlo.broadcast_divide %arg0, %arg0
  // CHECK-NEXT:  %[[FLOOR:.*]] = "xla_hlo.floor"(%[[DIV]])
  // CHECK-NEXT:  return %[[FLOOR]] : tensor<2xf32>
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0: tensor<2xf32>
}

// CHECK-LABEL: func @floordiv_bf16
func @floordiv_bf16(%arg0: tensor<2xbf16>) -> tensor<2xbf16> {
  // CHECK-NEXT:  xla_hlo.convert
  // CHECK-NEXT:  xla_hlo.convert
  // CHECK-NEXT:  xla_chlo.broadcast_divide
  // CHECK-NEXT:  xla_hlo.floor
  // CHECK-NEXT:  xla_hlo.convert
  // CHECK-NEXT:  return
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  return %0: tensor<2xbf16>
}

// CHECK-LABEL: func @floordiv_f16_broadcast
func @floordiv_f16_broadcast(%arg0: tensor<2x3xf16>, %arg1: tensor<3xf16>) -> tensor<2x3xf16> {
  // CHECK-NEXT:  xla_chlo.broadcast_divide
  // CHECK-NEXT:  xla_hlo.floor
  // CHECK-NEXT:  return
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
  return %0: tensor<2x3xf16>
}

// CHECK-LABEL: func @floordiv_dynamic
func @floordiv_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK: tf.FloorDiv
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @floordiv_unranked
func @floordiv_unranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: tf.FloorDiv
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0: tensor<*xi32>
}

// CHECK-LABEL: func @floormod_broadcast_numerator
func @floormod_broadcast_numerator(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = xla_chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ZL:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = xla_chlo.broadcast_compare [[REM]], [[ZL]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"}
  // CHECK-DAG: [[ZR:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = xla_chlo.broadcast_compare %arg1, [[ZR:%.+]] {comparison_direction = "LT"}
  // CHECK-DAG: [[CMP3:%.+]] = xla_chlo.broadcast_compare [[REM:%.+]], [[ZR]] {comparison_direction = "LT"}
  // CHECK-DAG: [[CMP4:%.+]] = xla_chlo.broadcast_compare [[CMP2]], [[CMP3]] {comparison_direction = "NE"}
  // CHECK-DAG: [[AND:%.+]] = xla_chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = xla_chlo.broadcast_add %arg1, [[REM]]
  // CHECK-DAG: [[SELECT:%.+]] = "xla_hlo.select"([[AND]], [[ADD]], [[REM]])
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0: tensor<2x3xi32>
}

// CHECK-LABEL: func @floormod_broadcast_denominator
func @floormod_broadcast_denominator(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: [[REM:%.+]] = xla_chlo.broadcast_remainder %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[ZL:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP1:%.+]] = xla_chlo.broadcast_compare [[REM]], [[ZL]] {comparison_direction = "NE"}
  // CHECK-DAG: [[ZR:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[CMP2:%.+]] = xla_chlo.broadcast_compare %arg1, [[ZR:%.+]] {comparison_direction = "LT"}
  // CHECK-DAG: [[CMP3:%.+]] = xla_chlo.broadcast_compare [[REM:%.+]], [[ZR]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "LT"}
  // CHECK-DAG: [[CMP4:%.+]] = xla_chlo.broadcast_compare [[CMP2]], [[CMP3]] {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"}
  // CHECK-DAG: [[AND:%.+]] = xla_chlo.broadcast_and [[CMP1]], [[CMP4]]
  // CHECK-DAG: [[ADD:%.+]] = xla_chlo.broadcast_add %arg1, [[REM]] {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SELECT:%.+]] = "xla_hlo.select"([[AND]], [[ADD]], [[REM]])
  // CHECK-NEXT: return [[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  return %0: tensor<2x3xi32>
}

// CHECK-LABEL: func @floormod_dynamic
func @floormod_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK: tf.FloorMod
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  return %0: tensor<?x?xi32>
}

// CHECK-LABEL: func @floormod_unranked
func @floormod_unranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: tf.FloorMod
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0: tensor<*xi32>
}

//===----------------------------------------------------------------------===//
// BroadcastTo.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @broadcast_to
func @broadcast_to(%arg0: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %cst = "tf.Const"() { value = dense<16> : tensor<4xi32> } : () -> tensor<4xi32>

  // CHECK: [[CST:%.+]] = xla_hlo.constant
  // CHECK: [[CAST:%.+]] = tensor_cast [[CST]] : tensor<4xi32> to tensor<4xi32>
  // CHECK: "xla_hlo.dynamic_broadcast_in_dim"(%arg0, [[CAST]])
  // CHECK-SAME: {broadcast_dimensions = dense<3> : tensor<1xi64>}
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  return %0 : tensor<16x16x16x16xf32>
}

//===----------------------------------------------------------------------===//
// Complex op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @complex
func @complex(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xcomplex<f32>> {
  // CHECK: "xla_hlo.complex"
  %1 = "tf.Complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
  return %1 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: func @imag
func @imag(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: "xla_hlo.imag"
  %1 = "tf.Imag"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @real
func @real(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: "xla_hlo.real"
  %1 = "tf.Real"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @conj
func @conj(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-DAG: [[R1:%.*]] = "xla_hlo.real"(%arg0)
  // CHECK-DAG: [[R2:%.*]] = "xla_hlo.imag"(%arg0)
  // CHECK-DAG: [[R3:%.*]] = "xla_hlo.negate"([[R2]])
  // CHECK: [[R4:%.*]] = "xla_hlo.complex"([[R1]], [[R3]])
  %1 = "tf.Conj"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  return %1 : tensor<3xcomplex<f32>>
}

//===----------------------------------------------------------------------===//
// Concat op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @concat_v2
func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  return %1 : tensor<6x3xf32>
}

// CHECK-LABEL: func @concat_v2_neg_axis
func @concat_v2_neg_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>

  %axis = "tf.Const"() { value = dense<-2> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  return %1 : tensor<6x3xf32>
}

// CHECK-LABEL: func @concat_v2_1d_axis
func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 1 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x6xf32>

  %axis = "tf.Const"() { value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi64>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

// CHECK-LABEL: func @concat_v2_non_const_axis
func @concat_v2_non_const_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %axis: tensor<i64>) -> tensor<3x6xf32> {
  // CHECK: "tf.ConcatV2"
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

// CHECK-LABEL: func @concat_v2_unranked
func @concat_v2_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  // CHECK: "tf.ConcatV2"
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<*xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// Pad op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @padv2_1D
func @padv2_1D(%arg0: tensor<3xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK: "xla_hlo.pad"(%arg0, %arg1) {
  // CHECK-SAME: edge_padding_high = dense<2> : tensor<1xi64>,
  // CHECK-SAME: edge_padding_low = dense<1> : tensor<1xi64>,
  // CHECK-SAME: interior_padding = dense<0> : tensor<1xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

// CHECK-LABEL: func @padv2_2D
func @padv2_2D(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi64> } : () -> tensor<2x2xi64>
  // CHECK: "xla_hlo.pad"(%arg0, %arg1) {
  // CHECK-SAME:    edge_padding_high = dense<[2, 4]> : tensor<2xi64>,
  // CHECK-SAME:    edge_padding_low = dense<[1, 3]> : tensor<2xi64>,
  // CHECK-SAME:    interior_padding = dense<0> : tensor<2xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<6x9xf32>
  return %1 : tensor<6x9xf32>
}

// CHECK-LABEL: func @padv2_i32_paddings
func @padv2_i32_paddings(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi32> } : () -> tensor<2x2xi32>
  // CHECK: "xla_hlo.pad"(%arg0, %arg1) {
  // CHECK-SAME:    edge_padding_high = dense<[2, 4]> : tensor<2xi64>,
  // CHECK-SAME:    edge_padding_low = dense<[1, 3]> : tensor<2xi64>,
  // CHECK-SAME:    interior_padding = dense<0> : tensor<2xi64>
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<6x9xf32>
  return %1 : tensor<6x9xf32>
}

//===----------------------------------------------------------------------===//
// Identity op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @identity
func @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.Identity"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// CHECK-LABEL: func @stopgradient
func @stopgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.StopGradient"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// CHECK-LABEL: func @preventgradient
func @preventgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.PreventGradient"(%arg0) {message = "fin gradients"} : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// InfeedDequeueTuple legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @infeed_dequeue_tuple
func @infeed_dequeue_tuple() -> (tensor<3xi32>, tensor<4xf32>) {
// CHECK: [[TOKEN:%.*]] = "xla_hlo.create_token"() : () -> !xla_hlo.token
// CHECK: [[INFEED:%.*]] = "xla_hlo.infeed"([[TOKEN]]) {infeed_config = ""} : (!xla_hlo.token) -> tuple<tuple<tensor<3xi32>, tensor<4xf32>>, !xla_hlo.token>
// CHECK: [[INFEED_VAL:%.*]] = "xla_hlo.get_tuple_element"([[INFEED]]) {index = 0 : i32} : (tuple<tuple<tensor<3xi32>, tensor<4xf32>>, !xla_hlo.token>) -> tuple<tensor<3xi32>, tensor<4xf32>>
// CHECK: [[RES_1:%.*]] = "xla_hlo.get_tuple_element"([[INFEED_VAL]]) {index = 0 : i32} : (tuple<tensor<3xi32>, tensor<4xf32>>) -> tensor<3xi32>
// CHECK: [[RES_2:%.*]] = "xla_hlo.get_tuple_element"([[INFEED_VAL]]) {index = 1 : i32} : (tuple<tensor<3xi32>, tensor<4xf32>>) -> tensor<4xf32>
// CHECK: return [[RES_1]], [[RES_2]]
  %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<3xi32>, tensor<4xf32>
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
func @infeed_dequeue_tuple_sharding() -> tensor<8xi32> {
  // CHECK: "xla_hlo.infeed"
  // An additional sharding is added at the end to account for token result.
  // CHECK-SAME: xla_hlo.sharding = "type: TUPLE\0Atuple_shardings {\0A type: MAXIMAL\0A tile_assignment_dimensions: 1\0A tile_assignment_devices: 0\0A}\0Atuple_shardings {\0A type: MAXIMAL\0A tile_assignment_dimensions: 1\0A tile_assignment_devices: 0\0A}\0A"
  %0 = "tf.InfeedDequeueTuple"() {_XlaSharding = "\08\02*\08\08\01\1A\01\01\22\01\00"} : () -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

//===----------------------------------------------------------------------===//
// Nullary op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @const
func @const() -> tensor<2xi32> {
  // CHECK: xla_hlo.constant dense<0> : tensor<2xi32>
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
  return %0: tensor<2xi32>
}

// CHECK-LABEL: @const_dynamic_output
func @const_dynamic_output() -> tensor<*xi32> {
  // CHECK: [[CONST:%.*]] = xla_hlo.constant dense<0> : tensor<2xi32>
  // CHECK: [[CAST:%.*]] = tensor_cast [[CONST]] : tensor<2xi32> to tensor<*xi32>
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> (tensor<*xi32>)
  // CHECK: return [[CAST]]
  return %0: tensor<*xi32>
}

// CHECK-LABEL: @opaque_const
func @opaque_const() -> tensor<!tf.variant<tensor<2xi32>>> {
  // CHECK-NOT: xla_hlo.constant
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<!tf.variant>} : () -> tensor<!tf.variant<tensor<2xi32>>>
  return %0 : tensor<!tf.variant<tensor<2xi32>>>
}

//===----------------------------------------------------------------------===//
// Matmul op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: matmul_notranspose
// CHECK-SAME: (%[[A:.*]]: tensor<5x7xf32>, %[[B:.*]]: tensor<7x11xf32>)
func @matmul_notranspose(%a: tensor<5x7xf32>, %b: tensor<7x11xf32>) -> tensor<5x11xf32> {
  // CHECK: "xla_hlo.dot"(%[[A]], %[[B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<5x7xf32>, tensor<7x11xf32>) -> tensor<5x11xf32>

  return %0 : tensor<5x11xf32>
}

// CHECK-LABEL: matmul_transpose_b
// CHECK-SAME: (%[[A:.*]]: tensor<5x7xf32>, %[[B:.*]]: tensor<11x7xf32>)
func @matmul_transpose_b(%a: tensor<5x7xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_B:.*]] = "xla_hlo.transpose"(%[[B]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK: "xla_hlo.dot"(%[[A]], %[[UPDATED_B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = true} : (tensor<5x7xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  return %0 : tensor<5x11xf32>
}

// CHECK-LABEL: matmul_transpose_both
// CHECK-SAME: (%[[A:.*]]: tensor<7x5xf32>, %[[B:.*]]: tensor<11x7xf32>)
func @matmul_transpose_both(%a: tensor<7x5xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_A:.*]] = "xla_hlo.transpose"(%[[A]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK: %[[UPDATED_B:.*]] = "xla_hlo.transpose"(%[[B]]) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK: "xla_hlo.dot"(%[[UPDATED_A]], %[[UPDATED_B]])
  %0 = "tf.MatMul"(%a, %b) {transpose_a = true, transpose_b = true} : (tensor<7x5xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  return %0 : tensor<5x11xf32>
}

// Verify that MatMul with ranked inputs are lowered to HLO.
// CHECK-LABEL: matmul_ranked
func @matmul_ranked(%a: tensor<?x7xf32>, %b: tensor<7x?xf32>) -> tensor<?x?xf32> {
  // CHECK: "xla_hlo.dot"
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<?x7xf32>, tensor<7x?xf32>) -> tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
}

// Verify that MatMul with unranked inputs are lowered to HLO.
// CHECK-LABEL: matmul_unranked
func @matmul_unranked(%a: tensor<*xf32>, %b: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "xla_hlo.dot"
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  return %0 : tensor<*xf32>
}

// Verify SparseMatMul is legalized to dot.
// CHECK-LABEL: test_sparse_mat_mul
func @test_sparse_mat_mul(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  // CHECK: "xla_hlo.dot"
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = false} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  return %0: tensor<3x5xf32>
}

//===----------------------------------------------------------------------===//
// MatrixBandPart op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: matrix_band_part
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func @matrix_band_part(%arg0: tensor<64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<64x64xbf16> {
  // CHECK: %[[M:.*]] = xla_hlo.constant dense<64> : tensor<i64>
  // CHECK: %[[N:.*]] = xla_hlo.constant dense<64> : tensor<i64>

  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i64>
  // CHECK: %[[A:.*]] = "xla_hlo.compare"(%[[LOWER]], %[[ZERO]]) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: %[[B:.*]] = "xla_hlo.select"(%[[A]], %[[M]], %[[LOWER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>

  // CHECK: %[[C:.*]] = "xla_hlo.compare"(%[[UPPER]], %[[ZERO]]) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK: %[[D:.*]] = "xla_hlo.select"(%[[C]], %[[N]], %[[UPPER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>

  // CHECK: %[[E:.*]] = "xla_hlo.convert"(%[[B]]) : (tensor<i64>) -> tensor<bf16>
  // CHECK: %[[F:.*]] = "xla_hlo.negate"(%[[E]]) : (tensor<bf16>) -> tensor<bf16>

  // CHECK: %[[X:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<64x64xbf16>
  // CHECK: %[[Y:.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<64x64xbf16>
  // CHECK: %[[OFFSET:.*]] = xla_hlo.subtract %[[X]], %[[Y]] : tensor<64x64xbf16>
  // CHECK: %[[G:.*]] = xla_chlo.broadcast_compare %[[F]], %[[OFFSET]] {comparison_direction = "LE"} : (tensor<bf16>, tensor<64x64xbf16>) -> tensor<64x64xi1>

  // CHECK: %[[H:.*]] = "xla_hlo.convert"(%[[D]]) : (tensor<i64>) -> tensor<bf16>
  // CHECK: %[[I:.*]] = xla_chlo.broadcast_compare %[[OFFSET]], %[[H]] {comparison_direction = "LE"} : (tensor<64x64xbf16>, tensor<bf16>) -> tensor<64x64xi1>

  // CHECK: %[[J:.*]] = xla_hlo.and %[[G]], %[[I]] : tensor<64x64xi1>

  // CHECK: %[[ZERO2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<64x64xbf16>
  // CHECK: %[[R:.*]] = "xla_hlo.select"(%[[J]], %[[INPUT]], %[[ZERO2]])
  // CHECK: return %[[R]]
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

// CHECK-LABEL: matrix_band_part_2
// CHECK-SAME: (%[[INPUT:.*]]: tensor<12x24x48xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func @matrix_band_part_2(%arg0: tensor<12x24x48xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<12x24x48xbf16> {
  // CHECK: %[[X:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<24x48xbf16>
  // CHECK: %[[Y:.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<24x48xbf16>
  // CHECK: %[[OFFSET:.*]] = xla_hlo.subtract %[[X]], %[[Y]] : tensor<24x48xbf16>

  // CHECK: %[[G:.*]] = xla_chlo.broadcast_compare %[[F]], %[[OFFSET]] {comparison_direction = "LE"} : (tensor<bf16>, tensor<24x48xbf16>) -> tensor<24x48xi1>

  // CHECK: %[[H:.*]] = "xla_hlo.convert"(%[[D]]) : (tensor<i64>) -> tensor<bf16>
  // CHECK: %[[I:.*]] = xla_chlo.broadcast_compare %[[OFFSET]], %[[H]] {comparison_direction = "LE"} : (tensor<24x48xbf16>, tensor<bf16>) -> tensor<24x48xi1>
  // CHECK: %[[J:.*]] = xla_hlo.and %[[G]], %[[I]] : tensor<24x48xi1>

  // CHECK: %[[ZERO2:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<12x24x48xbf16>
  // CHECK: %[[R:.*]] = "xla_hlo.select"(%[[J]], %[[INPUT]], %[[ZERO2]])
  // CHECK: return %[[R]]
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<12x24x48xbf16>, tensor<i64>, tensor<i64>) -> tensor<12x24x48xbf16>
  return %0 : tensor<12x24x48xbf16>
}

// CHECK-LABEL: matrix_band_part_3
// CHECK-SAME: (%[[INPUT:.*]]: tensor<*xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func @matrix_band_part_3(%arg0: tensor<*xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<*xbf16> {
  // CHECK: "tf.MatrixBandPart"
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<*xbf16>, tensor<i64>, tensor<i64>) -> tensor<*xbf16>
  return %0 : tensor<*xbf16>
}

// CHECK-LABEL: matrix_band_part_4
// CHECK-SAME: (%[[INPUT:.*]]: tensor<24x48xbf16>, %[[LOWER:.*]]: tensor<i64>, %[[UPPER:.*]]: tensor<i64>)
func @matrix_band_part_4(%arg0: tensor<24x48xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<24x48xbf16> {
  // This one should lower.
  // CHECK-NOT: "tf.MatrixBandPart"
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<24x48xbf16>, tensor<i64>, tensor<i64>) -> tensor<24x48xbf16>
  return %0 : tensor<24x48xbf16>
}

//===----------------------------------------------------------------------===//
// MaxPool op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: maxpool_valid_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func @maxpool_valid_padding(%arg0: tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<-2147483648> : tensor<i32>
  // CHECK: "xla_hlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK: xla_hlo.maximum
  // CHECK: xla_hlo.return
  // CHECK: {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>}

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 4, 4, 1]} : (tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32>
  return %0 : tensor<2x3x5x7xi32>
}

// CHECK-LABEL: maxpool_same_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func @maxpool_same_padding(%arg0: tensor<2x13x25x7xi32>) -> tensor<2x4x7x7xi32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 4, 1]} : (tensor<2x13x25x7xi32>) -> tensor<2x4x7x7xi32>
  return %0 : tensor<2x4x7x7xi32>
}

// CHECK-LABEL: maxpool_3d_valid_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func @maxpool_3d_valid_padding(%arg0: tensor<2x8x12x20x7xf32>) -> tensor<2x8x3x5x7xf32> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: "xla_hlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK: xla_hlo.maximum
  // CHECK: xla_hlo.return
  // CHECK: {window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 4, 4, 1]> : tensor<5xi64>}

  %0 = "tf.MaxPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 2, 2, 1], padding = "VALID", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x12x20x7xf32>) -> tensor<2x8x3x5x7xf32>
  return %0 : tensor<2x8x3x5x7xf32>
}

// CHECK-LABEL: maxpool_3d_same_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func @maxpool_3d_same_padding(%arg0: tensor<2x8x13x25x7xf32>) -> tensor<2x8x4x7x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>

  %0 = "tf.MaxPool3D"(%arg0) {data_format = "NDHWC", ksize = [1, 1, 2, 3, 1], padding = "SAME", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x13x25x7xf32>) -> tensor<2x8x4x7x7xf32>
  return %0 : tensor<2x8x4x7x7xf32>
}

//===----------------------------------------------------------------------===//
// MaxPoolGrad op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @max_pool_grad_valid
// CHECK-SAME: %[[INPUT:.*]]: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %[[GRAD:.*]]: tensor<10x12x12x64xf32>
func @max_pool_grad_valid(%orig_input: tensor<10x24x24x64xf32>, %orig_output: tensor<10x12x12x64xf32>, %grad: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) ( {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = "xla_hlo.compare"(%[[VALUE_A]], %[[VALUE_B]]) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: "xla_hlo.return"(%[[SELECT_RESULT]]) : (tensor<i1>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = xla_hlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: "xla_hlo.return"(%[[SELECT_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  // CHECK: return %[[RESULT]] : tensor<10x24x24x64xf32>
  // CHECK: }
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
  return %result : tensor<10x24x24x64xf32>
}

// CHECK-LABEL: @max_pool_3d_grad_valid
// CHECK-SAME: %[[INPUT:.*]]: tensor<10x8x24x24x64xf32>, %arg1: tensor<10x8x12x12x64xf32>, %[[GRAD:.*]]: tensor<10x8x12x12x64xf32>
func @max_pool_3d_grad_valid(%orig_input: tensor<10x8x24x24x64xf32>, %orig_output: tensor<10x8x12x12x64xf32>, %grad: tensor<10x8x12x12x64xf32>) -> tensor<10x8x24x24x64xf32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) ( {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = "xla_hlo.compare"(%[[VALUE_A]], %[[VALUE_B]]) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: "xla_hlo.return"(%[[SELECT_RESULT]]) : (tensor<i1>) -> ()
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = xla_hlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: "xla_hlo.return"(%[[SELECT_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>} : (tensor<10x8x24x24x64xf32>, tensor<10x8x12x12x64xf32>, tensor<f32>) -> tensor<10x8x24x24x64xf32>
  // CHECK: return %[[RESULT]] : tensor<10x8x24x24x64xf32>
  // CHECK: }
  %result = "tf.MaxPool3DGrad"(%orig_input, %orig_output, %grad) {data_format = "NDHWC", ksize = [1, 1, 2, 2, 1], padding = "VALID", strides = [1, 1, 2, 2, 1]} : (tensor<10x8x24x24x64xf32>, tensor<10x8x12x12x64xf32>, tensor<10x8x12x12x64xf32>) -> tensor<10x8x24x24x64xf32>
  return %result : tensor<10x8x24x24x64xf32>
}

// CHECK-LABEL: @max_pool_grad_same
func @max_pool_grad_same(%orig_input: tensor<2x13x25x7xf32>, %orig_output: tensor<2x4x7x7xf32>, %grad: tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 3, 1],
     padding = "SAME",
     strides = [1, 4, 4, 1]
  } : (tensor<2x13x25x7xf32>, tensor<2x4x7x7xf32>, tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32>
  return %result : tensor<2x13x25x7xf32>
}

// CHECK-LABEL: @max_pool_3d_grad_same
func @max_pool_3d_grad_same(%orig_input: tensor<2x8x13x25x7xf32>, %orig_output: tensor<2x8x4x7x7xf32>, %grad: tensor<2x8x4x7x7xf32>) -> tensor<2x8x13x25x7xf32> {
  // CHECK: padding = dense<{{\[\[}}0, 0], [0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>
  %result = "tf.MaxPool3DGrad"(%orig_input, %orig_output, %grad) {data_format = "NDHWC", ksize = [1, 1, 2, 3, 1], padding = "SAME", strides = [1, 1, 4, 4, 1]} : (tensor<2x8x13x25x7xf32>, tensor<2x8x4x7x7xf32>, tensor<2x8x4x7x7xf32>) -> tensor<2x8x13x25x7xf32>
  return %result : tensor<2x8x13x25x7xf32>
}

//===----------------------------------------------------------------------===//
// OneHot op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL:one_hot
func @one_hot(%indices: tensor<3xi32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  // CHECK: %[[IOTA:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<3x5xi32>
  // CHECK: %[[BCAST_ARG0:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<3x5xi32>
  // CHECK: %[[COMPARE:.*]] = "xla_hlo.compare"(%[[BCAST_ARG0]], %[[IOTA]]) {comparison_direction = "EQ"} : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<3x5xi1>
  // CHECK: %[[ON_VALUE:.*]] = "xla_hlo.broadcast"(%arg1) {broadcast_sizes = dense<[3, 5]> : tensor<2xi64>} : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[OFF_VALUE:.*]] = "xla_hlo.broadcast"(%arg2) {broadcast_sizes = dense<[3, 5]> : tensor<2xi64>} : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.select"(%[[COMPARE]], %[[ON_VALUE]], %[[OFF_VALUE]]) : (tensor<3x5xi1>, tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK: return %[[RESULT]] : tensor<3x5xf32>
  %depth = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  return %result : tensor<3x5xf32>
}

//===----------------------------------------------------------------------===//
// tf.OutfeedEnqueueTuple legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @outfeed_enqueue_tuple
// CHECK-SAME: [[VAL_0:%.*]]: tensor<3xi32>, [[VAL_1:%.*]]: tensor<4xf32>)
func @outfeed_enqueue_tuple(%data_1: tensor<3xi32>, %data_2: tensor<4xf32>) -> () {
// CHECK: [[TUPLE:%.*]] = "xla_hlo.tuple"([[VAL_0]], [[VAL_1]]) : (tensor<3xi32>, tensor<4xf32>) -> tuple<tensor<3xi32>, tensor<4xf32>>
// CHECK: [[TOKEN:%.*]] = "xla_hlo.create_token"() : () -> !xla_hlo.token
// CHECK: "xla_hlo.outfeed"([[TUPLE]], [[TOKEN]]) {outfeed_config = ""} : (tuple<tensor<3xi32>, tensor<4xf32>>, !xla_hlo.token) -> !xla_hlo.token
  "tf.OutfeedEnqueueTuple"(%data_1, %data_2) : (tensor<3xi32>, tensor<4xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Pack op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @pack
func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "xla_hlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "xla_hlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  %0 = "tf.Pack"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
// PartitionedCall op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @partitioned_call
func @partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: call @pcall_func(%arg0) : (tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func} : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

func @pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

// CHECK-LABEL: func @partitioned_call_multi_input
func @partitioned_call_multi_input(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: call @pcall_multi_input(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_input} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

func @pcall_multi_input(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

// CHECK-LABEL: func @partitioned_call_multi_in_out
func @partitioned_call_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: call @pcall_multi_in_out(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}

func @pcall_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  return %arg1, %arg0 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @unhandled_partitioned_call
func @unhandled_partitioned_call(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<i32>, tensor<i32>) {
  // The argument types don't match the parameter types for the
  // pcall_multi_in_out function. That's fine for a PartitionedCallOp but not
  // for a standard CallOp, so this op can't be lowered.
  // CHECK: "tf.PartitionedCall"
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @unhandled_partitioned_call_2
func @unhandled_partitioned_call_2(%arg0: tensor<i32>, %arg1: tensor<*xi32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: "tf.PartitionedCall"
  %0, %1 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_multi_in_out} : (tensor<i32>, tensor<*xi32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}


//===----------------------------------------------------------------------===//
// ReverseV2 op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reverse_func_32
func @reverse_func_32(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "xla_hlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi32>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  return %reversed : tensor<5xi32>
}

// CHECK-LABEL: @reverse_func_64
func @reverse_func_64(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> (tensor<1xi64>)

  // CHECK: [[VAL:%.+]] = "xla_hlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi64>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  return %reversed : tensor<5xi32>
}

// CHECK-LABEL: @reverse_func_neg
func @reverse_func_neg(%arg0: tensor<5x5xi32>) -> tensor<5x5xi32> {
  %axis = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = "xla_hlo.reverse"(%arg0) {dimensions = dense<1> : tensor<1xi64>}
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5x5xi32>, tensor<1xi32>) -> tensor<5x5xi32>

  // CHECK: return [[VAL]] : tensor<5x5xi32>
  return %reversed : tensor<5x5xi32>
}

//===----------------------------------------------------------------------===//
// StatefulPartitionedCall op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @stateful_partitioned_call
// CHECK-SAME: [[ARG:%.+]]: tensor<i32>
func @stateful_partitioned_call(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: call @stateful_pcall_func([[ARG]]) : (tensor<i32>) -> tensor<i32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

func @stateful_pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}

// CHECK-LABEL: func @stateful_partitioned_call_multi_in_out
// CHECK-SAME: ([[ARG0:%.+]]: tensor<i32>, [[ARG1:%.+]]: tensor<i32>)
func @stateful_partitioned_call_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: call @stateful_pcall_multi_in_out([[ARG0]], [[ARG1]]) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %0, %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @stateful_pcall_multi_in_out} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}

func @stateful_pcall_multi_in_out(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  return %arg1, %arg0 : tensor<i32>, tensor<i32>
}

//===----------------------------------------------------------------------===//
// Relu op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @relu
func @relu(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: xla_chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// CHECK-LABEL: func @relu_unranked
func @relu_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: xla_chlo.broadcast_maximum %[[ZERO]], %arg0 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  return %0: tensor<?xi32>
}

// CHECK-LABEL: func @relu6
func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SIX:.*]] = xla_hlo.constant dense<6> : tensor<i32>
  // CHECK: "xla_hlo.clamp"(%[[ZERO]], %arg0, %[[SIX]]) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// CHECK-LABEL: func @relu6_unranked
func @relu6_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: %[[SIX:.*]] = xla_hlo.constant dense<6> : tensor<i32>
  // CHECK: "xla_hlo.clamp"(%[[ZERO]], %arg0, %[[SIX]]) : (tensor<i32>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  return %0: tensor<?xi32>
}

// CHECK-LABEL: func @relu_grad
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<4x8xf32>, %[[FEATURES:.*]]: tensor<?x?xf32>)
func @relu_grad(%gradients: tensor<4x8xf32>, %features: tensor<?x?xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[ZERO_SCALAR:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  // CHECK-DAG: %[[PRED:.*]] = xla_chlo.broadcast_compare %[[FEATURES]], %[[ZERO_SCALAR]] {comparison_direction = "GT"} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
  // CHECK-DAG: %[[RESULT:.*]] = "xla_hlo.select"(%[[PRED]], %[[GRADIENTS]], %[[ZERO]]) : (tensor<?x?xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-DAG: return %[[RESULT]] : tensor<4x8xf32>
  %2 = "tf.ReluGrad"(%gradients, %features) : (tensor<4x8xf32>, tensor<?x?xf32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Select op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @selectv2
func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_pred_scalar
func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_then
func @selectv2_broadcast_then(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BROADCAST:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x1xi32>) -> tensor<2x8x8xi32>
  // CHECK: "xla_hlo.select"(%arg0, %[[BROADCAST]], %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_else
func @selectv2_broadcast_else(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BROADCAST:.*]] = "xla_hlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x1xi32>) -> tensor<2x8x8xi32>
  // CHECK: "xla_hlo.select"(%arg0, %arg1, %[[BROADCAST]])
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_pred
func @selectv2_broadcast_pred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // CHECK: %[[BROADCAST:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<2x8x8xi1>
  // CHECK: "xla_hlo.select"(%[[BROADCAST]], %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_tensor_pred
func @selectv2_broadcast_tensor_pred(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf16>, %arg2: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // CHECK: %[[BROADCAST:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK: "xla_hlo.select"(%[[BROADCAST]], %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %0: tensor<2x3xf16>
}

// CHECK-LABEL: func @selectv2_broadcast_all
func @selectv2_broadcast_all(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  // CHECK-DAG: %[[BROADCAST_0:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x1xi1>) -> tensor<8x8x8xi1>
  // CHECK-DAG: %[[BROADCAST_1:.*]] = "xla_hlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x8x1xi32>) -> tensor<8x8x8xi32>
  // CHECK-DAG: %[[BROADCAST_2:.*]] = "xla_hlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  // CHECK: "xla_hlo.select"(%[[BROADCAST_0]], %[[BROADCAST_1]], %[[BROADCAST_2]])
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  return %0: tensor<8x8x8xi32>
}

// CHECK-LABEL: func @selectv2_dynamic_ranked
func @selectv2_dynamic_ranked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  // CHECK: tf.SelectV2
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  return %0: tensor<2x?x8xi32>
}

// CHECK-LABEL: func @selectv2_unranked
func @selectv2_unranked(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: tf.SelectV2
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0: tensor<*xi32>
}

//===----------------------------------------------------------------------===//
// Softmax op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @simple_softmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>)
func @simple_softmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {

  // Verify reduce op for max computation and its body.
  // CHECK-DAG: %[[NEG_INF:.*]] = xla_hlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK-DAG: %[[CASTED_INP:.*]] = "xla_hlo.convert"(%[[ARG0]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %[[MAX:.*]] = "xla_hlo.reduce"(%[[CASTED_INP]], %[[NEG_INF]])
  // CHECK:  xla_hlo.maximum
  // CHECK: "xla_hlo.return"
  // CHECK: {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: %[[CASTED_MAX:.*]] = "xla_hlo.convert"(%[[MAX]]) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: %[[RESULT_SHAPE:.+]] = shape.shape_of %[[ARG0]]
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_SHAPE]]) : (!shape.shape) -> tensor<2xindex>
  // CHECK: %[[BCAST_MAX:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[CASTED_MAX]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: %[[SHIFTED_INP:.*]] = xla_hlo.subtract %[[ARG0]], %[[BCAST_MAX]]
  // CHECK: %[[EXP:.*]] = "xla_hlo.exponential"(%[[SHIFTED_INP]])

  // Verify reduce op for summation and its body.
  // CHECK-DAG: %[[CASTED_EXP:.*]] = "xla_hlo.convert"(%[[EXP]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SUM:.*]] = "xla_hlo.reduce"(%[[CASTED_EXP]], %[[ZERO]])
  // CHECK:  xla_hlo.add
  // CHECK: "xla_hlo.return"
  // CHECK: {dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[CASTED_SUM:.*]] = "xla_hlo.convert"(%[[SUM]]) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: %[[RESULT_SHAPE:.+]] = shape.shape_of %[[ARG0]]
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_SHAPE]]) : (!shape.shape) -> tensor<2xindex>
  // CHECK: %[[BCAST_SUM:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[CASTED_SUM]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: %[[RESULT:.*]] = xla_hlo.divide %[[EXP]], %[[BCAST_SUM]]
  // CHECK: return %[[RESULT]]

  %0 = "tf.Softmax"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>
}

// Verify intermediate and final shape are correct with dynamic shapes.
// CHECK-LABEL: func @dynamic_softmax
func @dynamic_softmax(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: xla_hlo.divide {{.*}}  : tensor<?x?xf32>
  %0 = "tf.Softmax"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}

// CHECK-LABEL: bf16_softmax
func @bf16_softmax(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  // Verify that conversion to f32 and then back to bf16 are introduced.

  // CHECK: "xla_hlo.convert"({{.*}}) : (tensor<2x3xbf16>) -> tensor<2x3xf32>
  // CHECK: "xla_hlo.convert"({{.*}}) : (tensor<2xf32>) -> tensor<2xbf16>

  %0 = "tf.Softmax"(%arg0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %0: tensor<2x3xbf16>
}

// CHECK-LABEL: rank4_softmax
func @rank4_softmax(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x3x4x5xf16> {
  // Verify that reduce op dimensions and broadcast dimensions are correct.

  // CHECK: "xla_hlo.reduce"
  // CHECK: dimensions = dense<3>

  // CHECK: "xla_hlo.reduce"
  // CHECK: dimensions = dense<3>

  // CHECK: {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
  // CHECK: xla_hlo.divide {{.*}}
  %0 = "tf.Softmax"(%arg0) : (tensor<2x3x4x5xf16>) -> tensor<2x3x4x5xf16>
  return %0: tensor<2x3x4x5xf16>
}

//===----------------------------------------------------------------------===//
// LogSoftmax op legalizations.
// This just changes the tail of the regular Softmax legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @simple_logsoftmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>)
func @simple_logsoftmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %{{.*}} = "xla_hlo.reduce"({{.*}})
  // CHECK: %[[SUM:.*]] = "xla_hlo.reduce"({{.*}})
  // CHECK: %[[CASTED_SUM:.*]] = "xla_hlo.convert"(%[[SUM]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[LOG:.*]] = "xla_hlo.log"(%[[CASTED_SUM]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[RESULT_SHAPE:.+]] = shape.shape_of %[[ARG0]]
  // CHECK: %[[RESULT_EXTENTS:.+]] = "shape.to_extent_tensor"(%[[RESULT_SHAPE]]) : (!shape.shape) -> tensor<2xindex>
  // CHECK: %[[BCAST_SUM:.+]] = "xla_hlo.dynamic_broadcast_in_dim"(%[[LOG]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: %[[RESULT:.*]] = xla_hlo.subtract {{.*}}, %[[BCAST_SUM]]
  // CHECK: return %[[RESULT]]

  %0 = "tf.LogSoftmax"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Fast Fourier Transform op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @rfft_1D
func @rfft_1D(%arg0: tensor<8xf32>) -> tensor<8xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: "xla_hlo.fft"(%arg0) {fft_length = dense<8> : tensor<1xi64>, fft_type = "RFFT"} : (tensor<8xf32>
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<8xf32>, tensor<1xi32>) -> tensor<8xcomplex<f32>>
  return %0 : tensor<8xcomplex<f32>>
}

//===----------------------------------------------------------------------===//
// Transpose op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transpose_noop
func @transpose_noop(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %permutation = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: return %arg0
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: @transpose_2d
func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_3d_int32
func @transpose_3d_int32(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
  return %0 : tensor<3x2x1xf32>
}

// CHECK-LABEL: @transpose_3d
func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> (tensor<3xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
  return %0 : tensor<3x2x1xf32>
}

// CHECK-LABEL: @transpose_dynamic_2d
func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// CHECK-LABEL: @transpose_unranked_2d
func @transpose_unranked_2d(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}


//===----------------------------------------------------------------------===//
// Unary op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @abs
func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @abs_dynamic
func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @abs_unranked
func @abs_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @cast_dynamic_i2f
func @cast_dynamic_i2f(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK: "xla_hlo.convert"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @cast_i2f
func @cast_i2f(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @cast_c2f
func @cast_c2f(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  //CHECK: "xla_hlo.convert"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @ceil
func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @ceil_dynamic
func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @ceil_unranked
func @ceil_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @complex_abs
func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.ComplexAbs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @cos
func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @cos_dynamic
func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.cosine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @cos_unranked
func @cos_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.cosine"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @exp
func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @exp_dynamic
func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.exponential"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @exp_unranked
func @exp_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.exponential"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @floor
func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @floor_dynamic
func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @floor_unranked
func @floor_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @is_finite
func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  // CHECK:  "xla_hlo.is_finite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: func @is_finite_dynamic
func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK:  "xla_hlo.is_finite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

// CHECK-LABEL: func @is_finite_unranked
func @is_finite_unranked(%arg0: tensor<*xf32>) -> tensor<*xi1> {
  // CHECK:  "xla_hlo.is_finite"(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// CHECK-LABEL: @log
func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @log_dynamic
func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @log_unranked
func @log_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @log1p
func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @log1p_dynamic
func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.log_plus_one"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @log1p_unranked
func @log1p_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.log_plus_one"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @not_op_unranked
func @not_op_unranked(%arg0: tensor<*xi1>) -> tensor<*xi1> {
  // CHECK:  "xla_hlo.not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  %0 = "tf.LogicalNot"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// CHECK-LABEL: @neg
func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @neg_dynamic
func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.negate"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @neg_unranked
func @neg_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.negate"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @sigmoid
func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: [[SCALAR:%.+]] = xla_hlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[SHAPE:%.+]] = shape.shape_of %arg0 : tensor<2xf32>
  // CHECK-DAG: [[SHAPE_VAL:%.+]] = "shape.to_extent_tensor"([[SHAPE]]) : (!shape.shape) -> tensor<1xindex>
  // CHECK-DAG: [[HALF:%.+]] = "xla_hlo.dynamic_broadcast_in_dim"([[SCALAR]], [[SHAPE_VAL]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<2xf32>
  // CHECK-DAG: [[R1:%.+]] =  xla_hlo.multiply %arg0, [[HALF]] : tensor<2xf32>
  // CHECK-DAG: [[R2:%.+]] =  "xla_hlo.tanh"([[R1]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK-DAG: [[R3:%.+]] =  xla_hlo.multiply [[R2]], [[HALF]] : tensor<2xf32>
  // CHECK-DAG: [[R4:%.+]] =  xla_hlo.add [[R3]], [[HALF]] : tensor<2xf32>
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @sigmoid_complex
func @sigmoid_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: [[R0:%.+]] = xla_hlo.constant dense<(5.000000e-01,0.000000e+00)> : tensor<complex<f32>>
  // CHECK-NOT: tf.Sigmoid
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @sigmoid_unranked
func @sigmoid_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-DAG: [[SCALAR:%.+]] = xla_hlo.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[SHAPE:%.+]] = shape.shape_of %arg0 : tensor<*xf32>
  // CHECK-DAG: [[SHAPE_VAL:%.+]] = "shape.to_extent_tensor"([[SHAPE]]) : (!shape.shape) -> tensor<?xindex>
  // CHECK-DAG: [[HALF:%.+]] = "xla_hlo.dynamic_broadcast_in_dim"([[SCALAR]], [[SHAPE_VAL]]) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK-DAG: [[R1:%.+]] =  xla_hlo.multiply %arg0, [[HALF]] : tensor<*xf32>
  // CHECK-DAG: [[R2:%.+]] =  "xla_hlo.tanh"([[R1]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: [[R3:%.+]] =  xla_hlo.multiply [[R2]], [[HALF]] : tensor<*xf32>
  // CHECK-DAG: [[R4:%.+]] =  xla_hlo.add [[R3]], [[HALF]] : tensor<*xf32>
  %0 = "tf.Sigmoid"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}


// CHECK-LABEL: @sigmoid_grad
func @sigmoid_grad(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: [[MUL0:%.+]] =  xla_hlo.multiply %arg1, %arg0 : tensor<2xf32>
  // CHECK-DAG: [[ONE:%.+]] = xla_hlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK-DAG: [[SUB:%.+]] =  xla_hlo.subtract [[ONE]], %arg0 : tensor<2xf32>
  // CHECK-DAG: [[MUL1:%.+]] =  xla_hlo.multiply [[MUL0]], [[SUB]] : tensor<2xf32>
  // CHECK: return [[MUL1]]
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @sigmoid_grad_complex
func @sigmoid_grad_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK-DAG: [[MUL0:%.+]] =  xla_hlo.multiply %arg1, %arg0 : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[ONE:%.+]] = xla_hlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[SUB:%.+]] =  xla_hlo.subtract [[ONE]], %arg0 : tensor<2xcomplex<f32>>
  // CHECK-DAG: [[MUL1:%.+]] =  xla_hlo.multiply [[MUL0]], [[SUB]] : tensor<2xcomplex<f32>>
  // CHECK: return [[MUL1]]
  %0 = "tf.SigmoidGrad"(%arg0, %arg1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @sin
func @sin(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.sine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @sin_dynamic
func @sin_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.sine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @sin_unranked
func @sin_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.sine"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Sin"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @rsqrt
func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @rsqrt_dynamic
func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rsqrt_unranked
func @rsqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @sqrt
func @sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @sqrt_dynamic
func @sqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @sqrt_unranked
func @sqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.sqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Sqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @tanh
func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @tanh_dynamic
func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @tanh_unranked
func @tanh_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @bitcast
func @bitcast(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @bitcast_dynamic
func @bitcast_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "xla_hlo.bitcast_convert"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @bitcast_unranked
func @bitcast_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "xla_hlo.bitcast_convert"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @bitcast_same_widths
func @bitcast_same_widths(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  // CHECK:  "xla_hlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func @bitcast_smaller_input_width
func @bitcast_smaller_input_width(%arg0: tensor<2xi8>) -> tensor<2xi64> {
  // CHECK:  "tf.Bitcast"(%arg0) : (tensor<2xi8>) -> tensor<2xi64>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xi8>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// CHECK-LABEL: func @bitcast_smaller_output_width
func @bitcast_smaller_output_width(%arg0: tensor<2xf32>) -> tensor<2xf16> {
  // CHECK:  "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xf16>
  %0 = "tf.Bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2xf16>
  return %0 : tensor<2xf16>
}

// CHECK-LABEL: reshape
func @reshape(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) -> tensor<2x1xf32> {
  // CHECK:  "xla_hlo.reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: reshape_dynamic
func @reshape_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<1x1xf32> {
  // CHECK:  "xla_hlo.reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: reshape_unranked
func @reshape_unranked(%arg0: tensor<*xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK:  "tf.Reshape"
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: squeeze
func @squeeze(%arg0: tensor<1x1x10xf32>) -> tensor<1x10xf32> {
  // CHECK: "xla_hlo.reshape"
  %0 = "tf.Squeeze"(%arg0) : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: squeeze_dynamic
func @squeeze_dynamic(%arg0: tensor<?x10xf32>) -> tensor<*xf32> {
  // CHECK: "tf.Squeeze"
  %0 = "tf.Squeeze"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: expand_dims
func @expand_dims(%arg0: tensor<2xf32>, %axis: tensor<i32>) -> tensor<1x2xf32> {
  // CHECK: "xla_hlo.reshape"
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<2xf32>, tensor<i32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: func @sign
// CHECK-SAME: [[ARG:%arg.*]]: tensor<1x2x3x4xf32>
func @sign(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: [[PRED:%.*]] = "xla_hlo.compare"([[ARG]], [[ARG]])
  // CHECK: [[ZEROS:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<1x2x3x4xf32>
  // CHECK: [[SIGN:%.*]] = "xla_hlo.sign"([[ARG]])
  // CHECK: [[SELECT:%.*]] = "xla_hlo.select"([[PRED]], [[ZEROS]], [[SIGN]])
  // CHECK: return [[SELECT]] : tensor<1x2x3x4xf32>
  %0 = "tf.Sign"(%arg0) : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
  return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: slice_constant_start
func @slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = xla_hlo.constant dense<1> : tensor<1xi64>
  // CHECK: %[[CAST:.*]] = tensor_cast %[[START]] : tensor<1xi64> to tensor<1xi64>
  // CHECK: %[[START_I64:.*]] = "xla_hlo.convert"(%[[CAST]]) : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK: %[[SLICED_START:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START:.*]] = "xla_hlo.reshape"(%[[SLICED_START:.*]]) :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.dynamic-slice"(%arg0, %[[RESHAPED_START]])
  // CHECK-DAG-SAME: {slice_sizes = dense<2> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: slice_i32_consts
func @slice_i32_consts(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[START:.*]] = xla_hlo.constant dense<1> : tensor<1xi32>
  // CHECK: %[[START_CAST:.*]] = tensor_cast %[[START]] : tensor<1xi32> to tensor<1xi32>
  // CHECK: %[[START_I64:.*]] = "xla_hlo.convert"(%[[START_CAST]]) : (tensor<1xi32>) -> tensor<1xi64>
  // CHECK: %[[SLICED_START:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START:.*]] = "xla_hlo.reshape"(%[[SLICED_START]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: "xla_hlo.dynamic-slice"(%arg0, %[[RESHAPED_START]]) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: slice_constant_start_negative_one_size
func @slice_constant_start_negative_one_size(%arg0: tensor<4xi32>) -> tensor<3xi32> {
  // CHECK: %[[START:.*]] = xla_hlo.constant dense<1> : tensor<1xi64>
  // CHECK: %[[START_CAST:.*]] = tensor_cast %[[START]] : tensor<1xi64> to tensor<1xi64>
  // CHECK: %[[START_I64:.*]] = "xla_hlo.convert"(%[[START_CAST]]) : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK: %[[SLICED_START:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START:.*]] = "xla_hlo.reshape"(%[[SLICED_START]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] =  "xla_hlo.dynamic-slice"(%arg0, %[[RESHAPED_START]]) {slice_sizes = dense<3> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<3xi32>
  // CHECK: return %[[RESULT]] : tensor<3xi32>
  %starts = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[-1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: slice_constant_start_dynamic_shape
func @slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[START:.*]] = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  // CHECK: %[[START_CAST:.*]] = tensor_cast %[[START]] : tensor<2xi64> to tensor<2xi64>
  // CHECK: %[[START_I64:.*]] = "xla_hlo.convert"(%[[START_CAST]]) : (tensor<2xi64>) -> tensor<2xi64>
  // CHECK: %[[SLICED_START1:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START1:.*]] = "xla_hlo.reshape"(%[[SLICED_START1]]) :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[SLICED_START2:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<2> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} :
  // CHECK-DAG-SAME: (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START2:.*]] = "xla_hlo.reshape"(%[[SLICED_START2]]) :
  // CHECK-DAG-SAME: (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.dynamic-slice"
  // CHECK-DAG-SAME: (%arg0, %[[RESHAPED_START1]], %[[RESHAPED_START2]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 4]> : tensor<2xi64>} :
  // CHECK-DAG-SAME: (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %starts = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %starts, %sizes) : (tensor<?x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: slice_variable_start
func @slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[START_I64:.*]] = "xla_hlo.convert"(%arg1) : (tensor<2xi64>) -> tensor<2xi64>
  // CHECK: %[[SLICED_START1:.*]] = "xla_hlo.slice"(%[[START_I64]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START1:.*]] = "xla_hlo.reshape"(%[[SLICED_START1]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[SLICED_START2:.*]] = "xla_hlo.slice"(%[[START_I64]]) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START2:.*]] = "xla_hlo.reshape"(%[[SLICED_START2]]) : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.dynamic-slice"(%arg0, %[[RESHAPED_START1]], %[[RESHAPED_START2]]) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: slice_variable_start_negative_one_size
func @slice_variable_start_negative_one_size(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[RESULT:.*]] = "tf.Slice"
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, -1]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

//===----------------------------------------------------------------------===//
// StridedSlice op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: simple_strided_slice
func @simple_strided_slice(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %output : tensor<3x2xf32>
}

// CHECK-LABEL: strided_slice_negative_indices
func @strided_slice_negative_indices(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[-1, -2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-4, -8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, -3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "xla_hlo.reverse"(%arg0) {dimensions = dense<[0, 1]> : tensor<2xi64>}

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %output : tensor<3x2xf32>
}

// CHECK-LABEL: strided_slice_range_clamping
func @strided_slice_range_clamping(%input: tensor<4x8xf32>) -> tensor<0x3xf32> {
  %begin = "tf.Const"() {value = dense<[-4, -10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-1, 10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "xla_hlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[3, 0]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 8]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<0x3xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<0x3xf32>
  return %output : tensor<0x3xf32>
}

// CHECK-LABEL: strided_slice_begin_end_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<4x128x1024xf32>
func @strided_slice_begin_end_mask(%input: tensor<4x128x1024xf32>) {

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

  // CHECK: %[[REVERSE:.*]] = "xla_hlo.reverse"(%[[INPUT]])

  // CHECK: %[[SLICE:.*]] = "xla_hlo.slice"(%[[REVERSE]])
  // CHECK-DAG-SAME: limit_indices = dense<[4, 65, 1024]>
  // CHECK-DAG-SAME: start_indices = dense<[0, 4, 2]>
  // CHECK-DAG-SAME: strides = dense<[1, 4, 1]>
  // CHECK-SAME: -> tensor<4x16x1022xf32>

  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 1, end_mask = 4} : (tensor<4x128x1024xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x16x1022xf32>

  // CHECK: "xla_hlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<4x16x1022xf32>

  return
}

// CHECK-LABEL: strided_slice_shrink_axis_mask
// CHECK-SAME: %[[INPUT:.+]]: tensor<4x128x1024xf32>
func @strided_slice_shrink_axis_mask(%input: tensor<4x128x1024xf32>) {

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

  // CHECK: %[[SLICE:.*]] = "xla_hlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[1, 65, 1022]>
  // CHECK-DAG-SAME: start_indices = dense<[0, 4, 1021]>
  // CHECK-DAG-SAME: strides = dense<[1, 4, 1]>
  // CHECK-SAME: -> tensor<1x16x1xf32>

  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 1, end_mask = 4, shrink_axis_mask = 5} : (tensor<4x128x1024xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<16xf32>

  // CHECK: "xla_hlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<16xf32>

  return
}

// CHECK-LABEL: strided_slice_ellipsis_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<2x4x8x16x32x64xf32>
func @strided_slice_ellipsis_mask(%input: tensor<2x4x8x16x32x64xf32>) {
  // For StridedSlice input[1, ..., 8:, :10, 2:6:2]
  // The ellipsis mask is applied to dim #1, #2, i.e, we get canonicalized
  // slice input[1, :, :, 8:, :10, 2:6:2]

  // The start, limit indices and strides attributes of xla_hlo.slice would
  // reflect the canonicalized slice.
  // As output shape of StridedSlice differs, a reshape will follow.

  %begin = "tf.Const"() {value = dense<[1, 0, 8, 1, 2]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %end = "tf.Const"() {value = dense<[2, 0, 10, 10, 6]> : tensor<5xi32>} : () -> (tensor<5xi32>)
  %strides = "tf.Const"() {value = dense<[1, 1, 1, 1, 2]> : tensor<5xi32>} : () -> (tensor<5xi32>)

  // CHECK: %[[SLICE:.*]] = "xla_hlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4, 8, 16, 10, 6]> : tensor<6xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: strides = dense<[1, 1, 1, 1, 1, 2]> : tensoe<6xi64>
  // CHECK-SAME: -> tensor<1x4x8x8x10x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 8, end_mask = 4, shrink_axis_mask = 1, ellipsis_mask = 2} : (tensor<2x4x8x16x32x64xf32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<4x8x8x10x2xf32>

  // CHECK: "xla_hlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<4x8x8x10x2xf32>

  return
}

// CHECK-LABEL: strided_slice_new_axis_mask
// CHECK-SAME: %[[INPUT:[a-z0-9]+]]: tensor<2x4x8x16x32x64xf32>
func @strided_slice_new_axis_mask(%input: tensor<2x4x8x16x32x64xf32>) {
  // For StridedSlice input[1, tf.new_axis, ..., 8:, :10, 2:6:2, tf.new_axis]
  // New axis mask is at index 1 and 6 of sparse spec, so
  // new_axis_mask = 2^1 + 2^6 = 66
  // The ellipsis mask is applied to dim #1, #2 of input i.e, we get
  // canonicalized slice input[1, :, :, 8:, :10, 2:6:2]
  // This is then reshaped to add the new axes.

  // The start, limit indices and strides attributes of xla_hlo.slice would
  // reflect the canonicalized slice.
  // As output shape of StridedSlice differs, a reshape will follow to reflect
  // new axes added.

  %begin = "tf.Const"() {value = dense<[1, 0, 0, 8, 1, 2, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %end = "tf.Const"() {value = dense<[2, 0, 0, 10, 10, 6, 0]> : tensor<7xi32>} : () -> (tensor<7xi32>)
  %strides = "tf.Const"() {value = dense<[1, 1, 1, 1, 1, 2, 1]> : tensor<7xi32>} : () -> (tensor<7xi32>)

  // CHECK: %[[SLICE:.*]] = "xla_hlo.slice"(%[[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4, 8, 16, 10, 6]> : tensor<6xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: strides = dense<[1, 1, 1, 1, 1, 2]> : tensoe<6xi64>
  // CHECK-SAME: -> tensor<1x4x8x8x10x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {begin_mask = 16, end_mask = 8, shrink_axis_mask = 1, ellipsis_mask = 4, new_axis_mask = 66} : (tensor<2x4x8x16x32x64xf32>, tensor<7xi32>, tensor<7xi32>, tensor<7xi32>) -> tensor<1x4x8x8x10x2x1xf32>

  // CHECK: "xla_hlo.reshape"(%[[SLICE]])
  // CHECK-SAME: -> tensor<1x4x8x8x10x2x1xf32>

  return
}

// CHECK-LABEL: strided_slice_implicit_ellipsis_mask(
// CHECK-SAME: [[INPUT:%.*]]: tensor<10x16x2xf32>
func @strided_slice_implicit_ellipsis_mask(%input: tensor<10x16x2xf32>) -> tensor<2x16x2xf32> {
  // StridedSlice gets input[8:10], which is same as input[8:10, ...]
  // The start_indices, limit_indices, and strides attribute of xla_hlo.slice
  // reflect the canonicalized slice.
  %begin = "tf.Const"() {value = dense<8> : tensor<1xi32>} : () -> tensor<1xi32>
  %end = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: [[SLICE:%.*]] = "xla_hlo.slice"([[INPUT]])
  // CHECK-DAG-SAME: limit_indices = dense<[10, 16, 2]> : tensor<3xi64>
  // CHECK-DAG-SAME: start_indices = dense<[8, 0, 0]> : tensor<3xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<3xi64>
  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"([[SLICE]]) : (tensor<2x16x2xf32>) -> tensor<2x16x2xf32>
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = f32} : (tensor<10x16x2xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2x16x2xf32>
  // CHECK: return [[RESHAPE]] : tensor<2x16x2xf32>
  return %0 : tensor<2x16x2xf32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end
func @strided_slice_nonconstant_begin_end(%arg0: tensor<i32>, %arg1: tensor<32x1x97xi32>) -> (tensor<1x97xi32>) {
  // In this case, the `begin` and `end` inputs are unknown at compile time --
  // so the StridedSlice needs to slice these vectors and use that as input to
  // an HLO dynamic slice.
  %begin = "tf.Pack"(%arg0) {N = 1 : i64, T = i32, axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.AddV2"(%arg0, %0) {T = i32, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %end = "tf.Pack"(%2) {N = 1 : i64, T = i32, axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  // CHECK: %[[A:.*]] = "xla_hlo.reshape"(%arg0) : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[BEGIN:.*]] = "xla_hlo.concatenate"(%[[A]])
  // CHECK-DAG-SAME: {dimension = 0 : i64} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[INDEX:.*]] = "xla_hlo.slice"(%[[BEGIN]])
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INDEX2:.*]] = "xla_hlo.reshape"(%[[INDEX]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[CMP:.*]] = xla_chlo.broadcast_compare %[[INDEX2]], %[[ZERO]]
  // CHECK-DAG-SAME: {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DIM:.*]] = xla_hlo.constant dense<32> : tensor<i32>
  // CHECK-NEXT: %[[WRAP:.*]] = xla_chlo.broadcast_add %[[DIM]], %[[INDEX2]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[INDEX3:.*]] = "xla_hlo.select"(%[[CMP]], %[[WRAP]], %[[INDEX2]]) :
  // CHECK-DAG-SAME: (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[SLICED:.*]] = "xla_hlo.dynamic-slice"
  // CHECK-DAG-SAME: (%arg1, %[[INDEX3]], %[[ZERO]], %[[ZERO]])
  // CHECK-DAG-SAME: {slice_sizes = dense<[1, 1, 97]> : tensor<3xi64>} :
  // CHECK-DAG-SAME: (tensor<32x1x97xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x97xi32>
  // CHECK-NEXT: %[[FINAL:.*]] = "xla_hlo.reshape"(%[[SLICED]]) : (tensor<1x97xi32>) -> tensor<1x97xi32>
  %result = "tf.StridedSlice"(%arg1, %begin, %end, %1) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  // CHECK-NEXT: return %[[FINAL]] : tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_stride_1
func @strided_slice_nonconstant_begin_end_stride_1(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>, %strides: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // Dynamic stride: when `begin` and `end` inputs are unknown at compile time,
  // `strides` must be known.
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 4 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_stride_2
func @strided_slice_nonconstant_begin_end_stride_2(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // Invalid stride (not equal to 1): when `begin` and `end` inputs are unknown
  // at compile time, `strides` must be known to have all 1 values.
  %strides = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 4 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_invalid_elem_count
func @strided_slice_nonconstant_begin_end_invalid_elem_count(%input: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>) -> tensor<6x10xf32> {
  %strides = "tf.Const"() { value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  // When begin/end are dynamic, the number of output elements must be equal to
  // the number of input elements sliced.
  // CHECK: tf.StridedSlice
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<6x10xf32>
  return %0 : tensor<6x10xf32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_begin_mask
func @strided_slice_nonconstant_begin_end_and_begin_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // Begin mask: When `begin` and `end` inputs are unknown at compile time, we
  // can't support a begin mask.
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 4 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_end_mask
func @strided_slice_nonconstant_begin_end_and_end_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // End mask: When `begin` and `end` inputs are unknown at compile time, we
  // can't support an end mask.
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_new_axis_mask
func @strided_slice_nonconstant_begin_end_and_new_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // New axis mask: When `begin` and `end` inputs are unknown at compile time,
  // we can't support a new_axis mask.
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 15 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_ellipsis_mask
func @strided_slice_nonconstant_begin_end_and_ellipsis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This ellipsis mask is not supported because it does not refer to the last
  // dimension.
  // [0, 1, 0] = 2
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 2 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_valid_ellipsis_mask
func @strided_slice_nonconstant_begin_end_and_valid_ellipsis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This ellipsis mask is supported because it refers to the last dimension.
  // [1, 0, 0] = 4
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: xla_hlo.dynamic-slice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 4 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_valid_shrink_axis_mask
func @strided_slice_nonconstant_begin_end_and_valid_shrink_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This shrink_axis mask is supported because it refers to a major dimension.
  // [1, 1, 1] = 7
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: xla_hlo.dynamic-slice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 7 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}

// CHECK-LABEL: strided_slice_nonconstant_begin_end_and_invalid_shrink_axis_mask
func @strided_slice_nonconstant_begin_end_and_invalid_shrink_axis_mask(%input: tensor<32x1x97xi32>, %begin: tensor<1xi32>, %end: tensor<1xi32>) -> (tensor<1x97xi32>) {
  // This shrink_axis mask is unsupported because it does not refer to a major
  // dimension.
  // [0, 1, 0] = 2
  %strides = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: tf.StridedSlice
  %result = "tf.StridedSlice"(%input, %begin, %end, %strides) {Index = i32, T = i32, begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64} : (tensor<32x1x97xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x97xi32>
  return %result : tensor<1x97xi32>
}


//===----------------------------------------------------------------------===//
// Reduction op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @mean
func @mean(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
  // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
  // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.add %[[ARGA]], %[[ARGB]] : tensor<f32>
  // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[DIVISOR:.*]] = xla_hlo.constant dense<8.000000e+00> : tensor<f32>
  // CHECK: %[[MEAN:.*]] = xla_chlo.broadcast_divide %[[REDUCED]], %[[DIVISOR]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[MEAN]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @mean_scalar_dim
func @mean_scalar_dim(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // Verify that tf.Mean op with scalar attributes are lowered successfully.

  // CHECK-NOT: tf.Mean
  %dimension = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<i64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @mean_dynamic
func @mean_dynamic(%arg0: tensor<4x?xf16>) -> tensor<4x1xf16> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  // CHECK: "tf.Mean"
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x?xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @sum
func @sum(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
  // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
  // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.add %[[ARGA]], %[[ARGB]] : tensor<f32>
  // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Sum"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @sum_dynamic
func @sum_dynamic(%arg0: tensor<4x?xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x?xf16>) -> tensor<4x?xf32>
    // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
    // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
    // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.add %[[ARGA]], %[[ARGB]] : tensor<f32>
    // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x?xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Sum"(%arg0, %dimension) { keep_dims = true }: (tensor<4x?xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @max
func @max(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf16>
  // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0xFC00> : tensor<f16>
  // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
  // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f16>, %[[ARGB:.*]]: tensor<f16>):
  // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.maximum %[[ARGA]], %[[ARGB]] : tensor<f16>
  // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f16>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf16>, tensor<f16>) -> tensor<4xf16>
  // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf16>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @max_dynamic
func @max_dynamic(%arg0: tensor<4x?xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x?xf16>) -> tensor<4x?xf16>
    // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0xFC00> : tensor<f16>
    // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
    // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f16>, %[[ARGB:.*]]: tensor<f16>):
    // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.maximum %[[ARGA]], %[[ARGB]] : tensor<f16>
    // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f16>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x?xf16>, tensor<f16>) -> tensor<4xf16>
    // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf16>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x?xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @min
func @min(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf16>
  // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<0x7C00> : tensor<f16>
  // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
  // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f16>, %[[ARGB:.*]]: tensor<f16>):
  // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.minimum %[[ARGA]], %[[ARGB]] : tensor<f16>
  // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f16>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf16>, tensor<f16>) -> tensor<4xf16>
  // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf16>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Min"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @prod
func @prod(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
  // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
  // CHECK: %[[INITIAL:.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
  // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
  // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.multiply %[[ARGA]], %[[ARGB]] : tensor<f32>
  // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
  // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Prod"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: @all
func @all(%input: tensor<4x8xi1>) -> tensor<4xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<true> : tensor<i1>
  // CHECK: "xla_hlo.reduce"(%{{.*}}, %[[INIT]]) ( {
  // CHECK: ^{{.*}}(%[[ARGA:.*]]: tensor<i1>, %[[ARGB:.*]]: tensor<i1>):
  // CHECK:  %[[AND:.*]] = xla_hlo.and %[[ARGA]], %[[ARGB]] : tensor<i1>
  // CHECK:  "xla_hlo.return"(%[[AND]]) : (tensor<i1>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  %0 = "tf.All"(%input, %dims) : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: @all_keep_dim
func @all_keep_dim(%input: tensor<4x8xi1>) -> tensor<4x1xi1> {
  // CHECK: "xla_hlo.reshape"(%{{.*}}) : (tensor<4xi1>) -> tensor<4x1xi1>
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.All"(%input, %dims) {keep_dims = true} : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  return %0 : tensor<4x1xi1>
}

// CHECk-LABEL: @all_dynamic
func @all_dynamic(%input: tensor<4x?xi1>) -> tensor<4x1xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[ARG:.*]] = "xla_hlo.convert"(%{{.*}}) : (tensor<4x?xi1>) -> tensor<4x?xi1>
  // CHECK: "xla_hlo.reduce"(%[[ARG]]
  %0 = "tf.All"(%input, %dims) {keep_dims = true} : (tensor<4x?xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  return %0 : tensor<4x1xi1>
}

// CHECK-LABEL: @any
func @any(%input: tensor<4x8xi1>) -> tensor<4xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<false> : tensor<i1>
  // CHECK: "xla_hlo.reduce"(%{{.*}}, %[[INIT]]) ( {
  // CHECK: ^{{.*}}(%[[ARGA:.*]]: tensor<i1>, %[[ARGB:.*]]: tensor<i1>):
  // CHECK:  %[[AND:.*]] = xla_hlo.or %[[ARGA]], %[[ARGB]] : tensor<i1>
  // CHECK:  "xla_hlo.return"(%[[AND]]) : (tensor<i1>) -> ()
  // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  %0 = "tf.Any"(%input, %dims) : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// CHECK-LABEL: @any_keep_dim
func @any_keep_dim(%input: tensor<4x8xi1>) -> tensor<4x1xi1> {
  // CHECK: "xla_hlo.reshape"(%{{.*}}) : (tensor<4xi1>) -> tensor<4x1xi1>
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Any"(%input, %dims) {keep_dims = true} : (tensor<4x8xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  return %0 : tensor<4x1xi1>
}

// CHECk-LABEL: @any_dynamic
func @any_dynamic(%input: tensor<4x?xi1>) -> tensor<4x1xi1> {
  %dims = "tf.Const"() { value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[ARG:.*]] = "xla_hlo.convert"(%{{.*}}) : (tensor<4x?xi1>) -> tensor<4x?xi1>
  // CHECK: "xla_hlo.reduce"(%[[ARG]]
  %0 = "tf.Any"(%input, %dims) {keep_dims = true} : (tensor<4x?xi1>, tensor<1xi32>) -> tensor<4x1xi1>
  return %0 : tensor<4x1xi1>
}

//===----------------------------------------------------------------------===//
// Tile op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tile_by_reshape
func @tile_by_reshape(%arg0: tensor<4x8xf32>) -> tensor<28x24xf32> {
  // CHECK: %[[BROADCASTED:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<4x8xf32>) -> tensor<4x7x8x3xf32>
  // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[BROADCASTED]]) : (tensor<4x7x8x3xf32>) -> tensor<28x24xf32>
  // CHECK: return %[[RESULT]] : tensor<28x24xf32>
  %multiples = "tf.Const"() { value = dense<[7,3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %0 = "tf.Tile"(%arg0, %multiples) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<28x24xf32>
  return %0 : tensor<28x24xf32>
}

// CHECK-LABEL: func @tile_just_broadcast
func @tile_just_broadcast(%arg0: tensor<1x1xf32>) -> tensor<7x3xf32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xf32>) -> tensor<7x3xf32>
  // CHECK: return %[[RESULT]] : tensor<7x3xf32>
  %multiples = "tf.Const"() { value = dense<[7,3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %0 = "tf.Tile"(%arg0, %multiples) : (tensor<1x1xf32>, tensor<2xi64>) -> tensor<7x3xf32>
  return %0 : tensor<7x3xf32>
}

//===----------------------------------------------------------------------===//
// ArgMax op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @argmax_i64_input_i32_output_axis_0
func @argmax_i64_input_i32_output_axis_0(%arg0: tensor<3x7xi64>) -> tensor<7xi32> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<-9223372036854775808> : tensor<i64>
  // CHECK: %[[INDEX_INIT:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: %[[INDEX:.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<3x7xi32>
  // CHECK: %[[REDUCE:.*]]:2 = "xla_hlo.reduce"(%arg0, %[[INDEX]], %[[INIT]], %[[INDEX_INIT]])
  // CHECK: ^bb0(%[[ARG1:.*]]: tensor<i64>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i64>, %[[ARG4:.*]]: tensor<i32>):
  // CHECK: %[[COMPARE:.*]] = "xla_hlo.compare"(%[[ARG1]], %[[ARG3]]) {comparison_direction = "GT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK:  %[[RESULT1:.*]] = "xla_hlo.select"(%[[COMPARE]], %[[ARG1]], %[[ARG3]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK:  %[[RESULT2:.*]] = "xla_hlo.select"(%[[COMPARE]], %[[ARG2]], %[[ARG4]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: "xla_hlo.return"(%[[RESULT1]], %[[RESULT2]]) : (tensor<i64>, tensor<i32>) -> ()
  // CHECK: return %[[REDUCE]]#1 : tensor<7xi32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x7xi64>, tensor<i32>) -> tensor<7xi32>
  return %0 : tensor<7xi32>
}

// CHECK-LABEL: func @argmax_f32_input_i64_output_axis_1
func @argmax_f32_input_i64_output_axis_1(%arg0: tensor<3x7xf32>) -> tensor<3xi64> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[INDEX_INIT:.*]] = xla_hlo.constant  dense<0> : tensor<i64>
  // CHECK: %[[INDEX:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<3x7xi64>
  // CHECK: %[[REDUCE:.*]]:2 = "xla_hlo.reduce"(%arg0, %[[INDEX]], %[[INIT]], %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<3xi64>
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x7xf32>, tensor<i32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// CHECK-LABEL: func @argmax_dynamic_shape_input_output
func @argmax_dynamic_shape_input_output(%arg0: tensor<3x?xi32>) -> tensor<?xi32> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<-2147483648> : tensor<i32>
  // CHECK: %[[INDEX_INIT:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: %[[INDEX:.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<3x?xi32>
  // CHECK: %[[REDUCE:.*]]:2 = "xla_hlo.reduce"(%arg0, %[[INDEX]], %[[INIT]], %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<?xi32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x?xi32>, tensor<i32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @argmax_dynamic_shape_input
func @argmax_dynamic_shape_input(%arg0: tensor<3x?xi32>) -> tensor<3xi32> {
  // CHECK: %[[INIT:.*]] = xla_hlo.constant dense<-2147483648> : tensor<i32>
  // CHECK: %[[INDEX_INIT:.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: %[[INDEX:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<3x?xi32>
  // CHECK: %[[REDUCE:.*]]:2 = "xla_hlo.reduce"(%arg0, %[[INDEX]], %[[INIT]], %[[INDEX_INIT]])
  // CHECK: return %[[REDUCE]]#1 : tensor<3xi32>
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %axis) : (tensor<3x?xi32>, tensor<i32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

//===----------------------------------------------------------------------===//
// Random op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @rng_uniform
func @rng_uniform(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[ONE:.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[CONV:.*]] = "xla_hlo.convert"(%arg0) : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK: %[[F32:.*]] = "xla_hlo.rng_uniform"(%[[ZERO]], %[[ONE]], %[[CONV]]) {{.*}} -> tensor<12x?x64xf32>
  %0 = "tf.RandomUniform"(%arg0) : (tensor<3xi32>) -> tensor<12x?x64xf32>
  // CHECK: return %[[F32]]
  return %0 : tensor<12x?x64xf32>
}

// CHECK-LABEL: func @rng_std_normal
func @rng_std_normal(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
  // CHECK: %[[ZERO:.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[ONE:.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[CONV:.*]] = "xla_hlo.convert"(%arg0) : (tensor<3xi32>) -> tensor<3xi64>
  // CHECK: %[[F32:.*]] = "xla_hlo.rng_normal"(%[[ZERO]], %[[ONE]], %[[CONV]]) {{.*}} -> tensor<12x?x64xf32>
  %0 = "tf.RandomStandardNormal"(%arg0) : (tensor<3xi32>) -> tensor<12x?x64xf32>
  // CHECK: return %[[F32]]
  return %0 : tensor<12x?x64xf32>
}

//===----------------------------------------------------------------------===//
// Range op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @range
// CHECK-SAME: [[START:%.*]]: tensor<f32>, [[DELTA:%.*]]: tensor<f32>
func @range(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<5xf32> {
  %1 = "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "range/limit", value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: [[IOTA:%.*]] = "xla_hlo.iota"
  // CHECK-DAG: [[MUL:%.*]] = xla_chlo.broadcast_multiply [[IOTA]], [[DELTA]] {broadcast_dimensions = dense<[]> : tensor<0xi64>}
  // CHECK: xla_chlo.broadcast_add [[MUL]], [[START]] {broadcast_dimensions = dense<[]> : tensor<0xi64>}
  %3 = "tf.Range"(%arg0, %1, %arg1) {Tidx = "tfdtype$DT_FLOAT", device = "", name = "range"} : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
  return %3 : tensor<5xf32>
}

// CHECK-LABEL: func @linspace_static
// CHECK-SAME: [[START:%.*]]: tensor<f32>, [[STOP:%.*]]: tensor<f32>
func @linspace_static(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // CHECK-DAG: [[NUM:%.*]] = xla_hlo.constant dense<4>
  // CHECK-DAG: [[NUM_CAST:%.*]] = tensor_cast [[NUM]]
  // CHECK-DAG: [[NUM_F32:%.*]] = "xla_hlo.convert"([[NUM_CAST]])
  // CHECK-DAG: [[ONE:%.*]] = xla_hlo.constant dense<1.000000e+00>
  // CHECK-DAG: [[STEP_DENOMINATOR:%.*]] = xla_chlo.broadcast_subtract [[NUM_F32]], [[ONE]]
  // CHECK-DAG: [[STEP_NUMERATOR:%.*]] = xla_chlo.broadcast_subtract [[STOP]], [[START]]
  // CHECK-DAG: [[STEP:%.*]] = xla_chlo.broadcast_divide [[STEP_NUMERATOR]], [[STEP_DENOMINATOR]]
  // CHECK-DAG: [[IOTA:%.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64}
  // CHECK-DAG: [[MUL:%.*]] = xla_chlo.broadcast_multiply [[IOTA]], [[STEP]] {broadcast_dimensions = dense<[]> : tensor<0xi64>}
  // CHECK-DAG: [[LINSPACE:%.*]] = xla_chlo.broadcast_add [[MUL]], [[START]] {broadcast_dimensions = dense<[]> : tensor<0xi64>}
  // CHECK: return [[LINSPACE]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.LinSpace"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @linspace_dynamic
func @linspace_dynamic(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<?xf32> {
  // CHECK: "tf.LinSpace"
  %0 = "tf.LinSpace"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @linspace_invalid_num
func @linspace_invalid_num(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK: xla_hlo.constant dense<[]> : tensor<0xi32>
  // CHECK: "tf.LinSpace"
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %1 = "tf.LinSpace"(%arg0, %arg1, %0) : (tensor<f32>, tensor<f32>, tensor<0xi32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// Conv op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: conv_simple
func @conv_simple(%arg0: tensor<256x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {

  // CHECK: "xla_hlo.convolution"(%arg0, %arg1)

  // Default attributes
  // CHECK-NOT: lhs_dilation
  // CHECK-NOT: precision_config

  // CHECK-DAG-SAME: window_strides = dense<[4, 5]>
  // CHECK-DAG-SAME: padding = dense<{{\[\[}}44, 45], [60, 60]]>
  // CHECK-DAG-SAME: rhs_dilation = dense<[2, 3]>

  // CHECK-DAG-SAME: dimension_numbers
  // CHECK-DAG-SAME:   input_batch_dimension = 0
  // CHECK-DAG-SAME:   input_feature_dimension = 3
  // CHECK-DAG-SAME:   input_spatial_dimensions = dense<[1, 2]>
  // CHECK-DAG-SAME:   kernel_input_feature_dimension = 2
  // CHECK-DAG-SAME:   kernel_output_feature_dimension = 3
  // CHECK-DAG-SAME:   kernel_spatial_dimensions = dense<[0, 1]>
  // CHECK-DAG-SAME:   output_batch_dimension = 0
  // CHECK-DAG-SAME:   output_feature_dimension = 3
  // CHECK-DAG-SAME:   output_spatial_dimensions = dense<[1, 2]>

  // CHECK-DAG-SAME: feature_group_count = 2
  // CHECK-DAG-SAME: batch_group_count = 1

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: conv3d_simple
func @conv3d_simple(%arg0: tensor<256x32x32x32x6xf32>, %arg1: tensor<3x3x3x3x16xf32>) -> tensor<256x30x30x30x16xf32> {

  // CHECK: "xla_hlo.convolution"(%arg0, %arg1)

  // Default attributes
  // CHECK-NOT: lhs_dilation
  // CHECK-NOT: precision_config

  // CHECK-DAG-SAME: window_strides = dense<[5, 6, 7]>
  // CHECK-DAG-SAME: padding = dense<[[1, 2], [2, 3], [2, 3]]>
  // CHECK-DAG-SAME: rhs_dilation = dense<[2, 3, 4]>

  // CHECK-DAG-SAME: dimension_numbers
  // CHECK-DAG-SAME:   input_batch_dimension = 0
  // CHECK-DAG-SAME:   input_feature_dimension = 4
  // CHECK-DAG-SAME:   input_spatial_dimensions = dense<[1, 2, 3]>
  // CHECK-DAG-SAME:   kernel_input_feature_dimension = 3
  // CHECK-DAG-SAME:   kernel_output_feature_dimension = 4
  // CHECK-DAG-SAME:   kernel_spatial_dimensions = dense<[0, 1, 2]>
  // CHECK-DAG-SAME:   output_batch_dimension = 0
  // CHECK-DAG-SAME:   output_feature_dimension = 4
  // CHECK-DAG-SAME:   output_spatial_dimensions = dense<[1, 2, 3]>

  // CHECK-DAG-SAME: feature_group_count = 2
  // CHECK-DAG-SAME: batch_group_count = 1

  %0 = "tf.Conv3D"(%arg0, %arg1) {data_format = "NDHWC", dilations = [1, 2, 3, 4, 1], padding = "SAME", strides = [1, 5, 6, 7, 1]} : (tensor<256x32x32x32x6xf32>, tensor<3x3x3x3x16xf32>) -> tensor<256x30x30x30x16xf32>
  return %0 : tensor<256x30x30x30x16xf32>
}

// CHECK-LABEL: depthwiseconv_simple
func @depthwiseconv_simple(%arg0: tensor<2x4x5x3xf32>, %arg1: tensor<2x2x3x3xf32>) -> tensor<2x3x4x9xf32> {
  // CHECK: %[[RESHAPED_FILTER:.*]] = "xla_hlo.reshape"(%arg1) : (tensor<2x2x3x3xf32>) -> tensor<2x2x1x9xf32>
  // CHECK: "xla_hlo.convolution"(%arg0, %[[RESHAPED_FILTER]])
  // CHECK: feature_group_count = 3
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {
    data_format = "NHWC",
    device = "",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1]
  } : (tensor<2x4x5x3xf32>, tensor<2x2x3x3xf32>) -> tensor<2x3x4x9xf32>
  return %0 : tensor<2x3x4x9xf32>
}

// CHECK-LABEL: conv_valid_padding
func @conv_valid_padding(%arg0: tensor<1x4x5x1xf32>, %arg1: tensor<3x3x1x1xf32>) -> tensor<1x2x3x1xf32> {
  // CHECK: "xla_hlo.convolution"(%arg0, %arg1)

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x4x5x1xf32>, tensor<3x3x1x1xf32>) -> tensor<1x2x3x1xf32>
  return %0 : tensor<1x2x3x1xf32>
}

// CHECK-LABEL: conv_explicit_paddings
func @conv_explicit_paddings(%arg0: tensor<256x32x32x6xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {

  // CHECK: "xla_hlo.convolution"(%arg0, %arg1)
  // CHECK-SAME: padding = dense<{{\[\[}}6, 0], [3, 3]]>

  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "EXPLICIT", explicit_paddings = [0, 0, 6, 0, 3, 3, 0, 0], strides = [1, 4, 5, 1]} : (tensor<256x32x32x6xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  return %0 : tensor<256x32x32x16xf32>
}

// CHECK-LABEL: @conv2d_backprop_input
func @conv2d_backprop_input(
    %filter: tensor<3x3x1x32xf32>,
    %out_backprop: tensor<100x26x26x32xf32>
  ) -> tensor<100x28x28x1xf32> {
    // CHECK: %[[REV_FILTER:.*]] = "xla_hlo.reverse"(%arg0) {dimensions = dense<[0, 1]> : tensor<2xi64>}
    // CHECK: %[[RESULT:.*]] = "xla_hlo.convolution"(%arg1, %[[REV_FILTER]]) {
    // CHECK-SAME: batch_group_count = 1 : i64,
    // CHECK-SAME: dimension_numbers = {
    // CHECK-SAME:   input_batch_dimension = 0 : i64,
    // CHECK-SAME:   input_feature_dimension = 3 : i64,
    // CHECK-SAME:   input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
    // CHECK-SAME:   kernel_input_feature_dimension = 3 : i64,
    // CHECK-SAME:   kernel_output_feature_dimension = 2 : i64,
    // CHECK-SAME:   kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
    // CHECK-SAME:   output_batch_dimension = 0 : i64,
    // CHECK-SAME:   output_feature_dimension = 3 : i64,
    // CHECK-SAME:   output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    // CHECK-SAME: },
    // CHECK-SAME: feature_group_count = 1 : i64,
    // CHECK-SAME: lhs_dilation = dense<1> : tensor<2xi64>,
    // CHECK-SAME: padding = dense<2> : tensor<2x2xi64>,
    // CHECK-SAME: rhs_dilation = dense<1> : tensor<2xi64>,
    // CHECK-SAME: window_strides = dense<1> : tensor<2xi64>
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
  return %result : tensor<100x28x28x1xf32>
}

// CHECK-LABEL: @conv3d_backprop_input
func @conv3d_backprop_input(%filter: tensor<3x3x3x1x6xf32>, %out_backprop: tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32> {
  // CHECK: %[[REV_FILTER:.*]] = "xla_hlo.reverse"(%arg0) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
  // CHECK: %[[RESULT:.*]] = "xla_hlo.convolution"(%arg1, %[[REV_FILTER]])

  // CHECK-DAG-SAME: batch_group_count = 1 : i64,

  // CHECK-DAG-SAME: dimension_numbers =
  // CHECK-DAG-SAME:   input_batch_dimension = 0 : i64
  // CHECK-DAG-SAME:   input_feature_dimension = 4 : i64
  // CHECK-DAG-SAME:   input_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>
  // CHECK-DAG-SAME:   kernel_input_feature_dimension = 4 : i64
  // CHECK-DAG-SAME:   kernel_output_feature_dimension = 3 : i64
  // CHECK-DAG-SAME:   kernel_spatial_dimensions = dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK-DAG-SAME:   output_batch_dimension = 0 : i64
  // CHECK-DAG-SAME:   output_feature_dimension = 4 : i64
  // CHECK-DAG-SAME:   output_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>

  // CHECK-DAG-SAME: feature_group_count = 1 : i64
  // CHECK-DAG-SAME: lhs_dilation = dense<1> : tensor<3xi64>
  // CHECK-DAG-SAME: padding = dense<1> : tensor<3x2xi64>
  // CHECK-DAG-SAME: rhs_dilation = dense<1> : tensor<3xi64>
  // CHECK-DAG-SAME: window_strides = dense<1> : tensor<3xi64>

  // CHECK: return %[[RESULT]]
  %input_sizes = "tf.Const" () {value = dense<[2, 8, 8, 8, 1]> : tensor<5xi32>} : () -> tensor<5xi32>
  %result = "tf.Conv3DBackpropInputV2"(%input_sizes, %filter, %out_backprop) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1],  padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<5xi32>, tensor<3x3x3x1x6xf32>, tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32>
  return %result : tensor<2x8x8x8x1xf32>
}

// CHECK-LABEL: @conv2d_backprop_filter
func @conv2d_backprop_filter(
    %input: tensor<100x28x28x1xf32>,
    %out_backprop: tensor<100x26x26x32xf32>
  ) -> tensor<100x28x28x1xf32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.convolution"(%arg0, %arg1) {
  // CHECK-SAME:  batch_group_count = 1 : i64,
  // CHECK-SAME:  dimension_numbers = {
  // CHECK-SAME:    input_batch_dimension = 3 : i64,
  // CHECK-SAME:    input_feature_dimension = 0 : i64,
  // CHECK-SAME:    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
  // CHECK-SAME:    kernel_input_feature_dimension = 0 : i64,
  // CHECK-SAME:    kernel_output_feature_dimension = 3 : i64,
  // CHECK-SAME:    kernel_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
  // CHECK-SAME:    output_batch_dimension = 2 : i64,
  // CHECK-SAME:    output_feature_dimension = 3 : i64,
  // CHECK-SAME:    output_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>
  // CHECK-SAME:  },
  // CHECK-SAME:  feature_group_count = 1 : i64,
  // CHECK-SAME:  lhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:  padding = dense<0> : tensor<2x2xi64>,
  // CHECK-SAME:  rhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:  window_strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]]
  %filter_sizes = "tf.Const" () { value = dense<[3,3,1,32]> : tensor<4xi32> } : () -> tensor<4xi32>
  %result = "tf.Conv2DBackpropFilter"(%input, %filter_sizes, %out_backprop) {
    data_format = "NHWC",
    dilations = [1, 1, 1, 1],
    explicit_paddings = [],
    padding = "VALID",
    strides = [1, 1, 1, 1],
    use_cudnn_on_gpu = true
  } : (tensor<100x28x28x1xf32>, tensor<4xi32>, tensor<100x26x26x32xf32>) -> tensor<100x28x28x1xf32>
  return %result : tensor<100x28x28x1xf32>
}

// CHECK-LABEL: @conv3d_backprop_filter
func @conv3d_backprop_filter(%input: tensor<2x8x8x8x1xf32>, %out_backprop: tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.convolution"(%arg0, %arg1)

  // CHECK-DAG-SAME: batch_group_count = 1 : i64

  // CHECK-DAG-SAME: dimension_numbers =
  // CHECK-DAG-SAME:   input_batch_dimension = 4 : i64
  // CHECK-DAG-SAME:   input_feature_dimension = 0 : i64
  // CHECK-DAG-SAME:   input_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>
  // CHECK-DAG-SAME:   kernel_input_feature_dimension = 0 : i64
  // CHECK-DAG-SAME:   kernel_output_feature_dimension = 4 : i64
  // CHECK-DAG-SAME:   kernel_spatial_dimensions = dense<[1, 2, 3]> : tensor<3xi64>
  // CHECK-DAG-SAME:   output_batch_dimension = 3 : i64
  // CHECK-DAG-SAME:   output_feature_dimension = 4 : i64
  // CHECK-DAG-SAME:   output_spatial_dimensions = dense<[0, 1, 2]> : tensor<3xi64>

  // CHECK-DAG-SAME: feature_group_count = 1 : i64
  // CHECK-DAG-SAME: lhs_dilation = dense<1> : tensor<3xi64>
  // CHECK-DAG-SAME: padding = dense<1> : tensor<3x2xi64>
  // CHECK-DAG-SAME: rhs_dilation = dense<1> : tensor<3xi64>
  // CHECK-DAG-SAME: window_strides = dense<1> : tensor<3xi64>

  // CHECK: return %[[RESULT]]
  %filter_sizes = "tf.Const"() {value = dense<[3, 3, 3, 1, 6]> : tensor<5xi32>} : () -> tensor<5xi32>
  %result = "tf.Conv3DBackpropFilterV2"(%input, %filter_sizes, %out_backprop) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1],  padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<2x8x8x8x1xf32>, tensor<5xi32>, tensor<2x8x8x8x6xf32>) -> tensor<2x8x8x8x1xf32>
  return %result : tensor<2x8x8x8x1xf32>
}

// CHECK-LABEL: @cross_replica_sum
func @cross_replica_sum(%input: tensor<10xf32>) -> tensor<10xf32> {
  %replica_groups = "tf.Const" () {
    value = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  } : () -> tensor<2x4xi32>

  // CHECK: xla_hlo.cross-replica-sum
  // CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  %result = "tf.CrossReplicaSum" (%input, %replica_groups) : (tensor<10xf32>, tensor<2x4xi32>) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

//===----------------------------------------------------------------------===//
// tf.Size legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @size_rank_one_i32
func @size_rank_one_i32(%input: tensor<f32>) -> (tensor<i32>) {
  // CHECK: %[[CONST:.*]] = xla_hlo.constant dense<1>
  // CHECK-SAME: tensor<i32>
  %size = "tf.Size"(%input) {T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<f32>) -> tensor<i32>
  // CHECK: return %[[CONST]]
  return %size : tensor<i32>
}

// CHECK-LABEL: @size_rank_one_i64
func @size_rank_one_i64(%input: tensor<f32>) -> (tensor<i64>) {
  // CHECK: %[[CONST:.*]] = xla_hlo.constant dense<1>
  // CHECK-SAME: tensor<i64>
  %size = "tf.Size"(%input) {T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT64"} : (tensor<f32>) -> tensor<i64>
  // CHECK: return %[[CONST]]
  return %size : tensor<i64>
}

// CHECK-LABEL: @size_ranked
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x?x8xf32>)
func @size_ranked(%input: tensor<2x?x8xf32>) -> (tensor<i32>) {
  // CHECK: %[[CONST:.*]] = xla_hlo.constant dense<1>
  // CHECK: %[[DIM_0:.*]] = "xla_hlo.get_dimension_size"(%[[INPUT]])
  // CHECK-SAME: dimension = 0
  // CHECK: %[[MUL_0:.*]] = xla_chlo.broadcast_multiply %[[CONST]], %[[DIM_0]]
  // CHECK: %[[DIM_1:.*]] = "xla_hlo.get_dimension_size"(%[[INPUT]])
  // CHECK-SAME: dimension = 1
  // CHECK: %[[MUL_1:.*]] = xla_chlo.broadcast_multiply %[[MUL_0]], %[[DIM_1]]
  // CHECK: %[[DIM_2:.*]] = "xla_hlo.get_dimension_size"(%[[INPUT]])
  // CHECK-SAME: dimension = 2
  // CHECK: %[[MUL_2:.*]] = xla_chlo.broadcast_multiply %[[MUL_1]], %[[DIM_2]]
  %size = "tf.Size"(%input) {T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<2x?x8xf32>) -> tensor<i32>
  // CHECK: return %[[MUL_2]]
  return %size : tensor<i32>
}

// CHECK-LABEL: @size_unranked
func @size_unranked(%input: tensor<*xf32>) -> (tensor<i32>) {
  // CHECK: tf.Size
  %size = "tf.Size"(%input) {T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<i32>
  return %size : tensor<i32>
}

//===----------------------------------------------------------------------===//
// tf.Split legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @split_not_match_non_const_split_dim
func @split_not_match_non_const_split_dim(%input: tensor<4x4xf32>, %split_dim: tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK: tf.Split
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<i32>, tensor<4x4xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @split_not_match_unknown_input_dim
func @split_not_match_unknown_input_dim(%input: tensor<4x?x4xf32>) -> (tensor<4x?x4xf32>, tensor<4x?x4xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.Split
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x?x4xf32>) -> (tensor<4x?x4xf32>, tensor<4x?x4xf32>)
  return %0#0, %0#1 : tensor<4x?x4xf32>, tensor<4x?x4xf32>
}

// CHECK-LABEL: @split_match_and_split_into_two
func @split_match_and_split_into_two(%input: tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>) {
  %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[2, 6]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<2x6xf32>
  // CHECK: %[[TWO:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<2x6xf32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
  // CHECK: return %[[ONE]], %[[TWO]]
  return %0#0, %0#1 : tensor<2x6xf32>, tensor<2x6xf32>
}

// CHECK-LABEL: @split_match_and_split_into_two_dynamic
func @split_match_and_split_into_two_dynamic(%input: tensor<4x?xf32>) -> (tensor<2x?xf32>, tensor<2x?xf32>) {
  %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[2, -1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x?xf32>) -> tensor<2x?xf32>
  // CHECK: %[[TWO:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[4, -1]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x?xf32>) -> tensor<2x?xf32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x?xf32>) -> (tensor<2x?xf32>, tensor<2x?xf32>)
  // CHECK: return %[[ONE]], %[[TWO]]
  return %0#0, %0#1 : tensor<2x?xf32>, tensor<2x?xf32>
}

// CHECK-LABEL: @split_match_and_split_into_three
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x6xf32>)
func @split_match_and_split_into_three(%input: tensor<4x6xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[TWO:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<4> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[THREE:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 4]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  %0:3 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
  // CHECK: return %[[ONE]], %[[TWO]], %[[THREE]]
  return %0#0, %0#1, %0#2 : tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>
}

//===----------------------------------------------------------------------===//
// tf.TopKV2 legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: topk_v2_non_const_k
func @topk_v2_non_const_k(%input: tensor<16xf32>, %k: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // CHECK: tf.TopKV2
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
  return %0#0, %0#1: tensor<?xf32>, tensor<?xi32>
}

// CHECK-LABEL: topk_v2_unknown_input_last_dim
func @topk_v2_unknown_input_last_dim(%input: tensor<16x?xf32>) -> (tensor<16x?xf32>, tensor<16x?xi32>) {
  %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.TopKV2
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x?xf32>, tensor<i32>) -> (tensor<16x?xf32>, tensor<16x?xi32>)
  return %0#0, %0#1: tensor<16x?xf32>, tensor<16x?xi32>
}

// CHECK-LABEL: topk_v2
// CHECK-SAME: %[[INPUT:.*]]: tensor<16x16xf32>
func @topk_v2(%input: tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>

  // CHECK:      %[[IOTA:.*]] = "xla_hlo.iota"() {iota_dimension = 1 : i64}
  // CHECK-NEXT: %[[SORT:.*]] = "xla_hlo.sort"(%[[INPUT]], %[[IOTA]]) ( {
  // CHECK-NEXT: ^{{.*}}(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>, %{{.*}}: tensor<i32>, %{{.*}}: tensor<i32>):
  // CHECK-NEXT:   %[[CMP:.*]] = "xla_hlo.compare"(%[[LHS]], %[[RHS]]) {comparison_direction = "GT"}
  // CHECK-NEXT:   "xla_hlo.return"(%[[CMP]])
  // CHECK-NEXT: }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  // CHECK-NEXT: %[[TUPL0:.*]] = "xla_hlo.get_tuple_element"(%[[SORT]]) {index = 0 : i32}
  // CHECK-NEXT: %[[TUPL1:.*]] = "xla_hlo.get_tuple_element"(%[[SORT]]) {index = 1 : i32}
  // CHECK-NEXT: %[[VAL:.*]] = "xla_hlo.slice"(%[[TUPL0]]) {limit_indices = dense<[16, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-NEXT: %[[IDX:.*]] = "xla_hlo.slice"(%[[TUPL1]]) {limit_indices = dense<[16, 8]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-NEXT: return %[[VAL]], %[[IDX]]
  %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x16xf32>, tensor<i32>) -> (tensor<16x8xf32>, tensor<16x8xi32>)
  return %0#0, %0#1: tensor<16x8xf32>, tensor<16x8xi32>
}

//===----------------------------------------------------------------------===//
// tf.SplitV legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @splitv_match_and_split_into_three
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x6xf32>)
func @splitv_match_and_split_into_three(%input: tensor<4x6xf32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[ONE:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x1xf32>
  // CHECK: %[[TWO:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x2xf32>
  // CHECK: %[[THREE:.*]] = "xla_hlo.slice"(%[[ARG]]) {limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>) -> tensor<4x3xf32>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
  // CHECK: return %[[ONE]], %[[TWO]], %[[THREE]]
  return %0#0, %0#1, %0#2 : tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>
}

// CHECK-LABEL: @splitv_match_and_split_into_three_dynamic
func @splitv_match_and_split_into_three_dynamic(%input: tensor<?x6xf32>) -> (tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x1xf32>
  // CHECK: "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x2xf32>
  // CHECK: "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[-1, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<?x6xf32>) -> tensor<?x3xf32>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<?x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>)
  return %0#0, %0#1, %0#2 : tensor<?x1xf32>, tensor<?x2xf32>, tensor<?x3xf32>
}

// CHECK-LABEL: @splitv_dynamic_dim_in_split_sizes
func @splitv_dynamic_dim_in_split_sizes(%input: tensor<4x6xf32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>) {
  %split_sizes = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: limit_indices = dense<[4, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>
  // CHECK: limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>
  // CHECK: limit_indices = dense<[4, 6]> : tensor<2xi64>, start_indices = dense<[0, 3]> : tensor<2xi64>
  %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
  return %0#0, %0#1, %0#2 : tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>
}

//===----------------------------------------------------------------------===//
// tf.Assert legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @assert
func @assert(%arg0: tensor<i1>, %arg1: tensor<*xf32>) {
  // CHECK-NOT: tf.Assert
  "tf.Assert"(%arg0, %arg1) {summarize = 1} : (tensor<i1>, tensor<*xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// tf.Unpack legalization
//===----------------------------------------------------------------------===//

// TODO(b/156340000): Re-enable when fixed.
// // C-HECK-LABEL: @unpack
// func @unpack(%input: tensor<4x3x6xf32>) -> (tensor<4x?xf32>, tensor<4x6xf32>, tensor<4x6xf32>) {
//   // C-HECK: %[[SLICE1:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[4, 1, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
//   // C-HECK: %[[RES1:.*]] = "xla_hlo.reshape"(%[[SLICE1]]) : (tensor<4x1x6xf32>) -> tensor<4x?xf32>
//   // C-HECK: %[[SLICE2:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[4, 2, 6]> : tensor<3xi64>, start_indices = dense<[0, 1, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
//   // C-HECK: %[[RES2:.*]] = "xla_hlo.reshape"(%[[SLICE2]]) : (tensor<4x1x6xf32>) -> tensor<4x6xf32>
//   // C-HECK: %[[SLICE3:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[4, 3, 6]> : tensor<3xi64>, start_indices = dense<[0, 2, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x3x6xf32>) -> tensor<4x1x6xf32>
//   // C-HECK: %[[RES3:.*]] = "xla_hlo.reshape"(%[[SLICE3]]) : (tensor<4x1x6xf32>) -> tensor<4x6xf32>

//   %0:3 = "tf.Unpack"(%input) {axis = 1} : (tensor<4x3x6xf32>) -> (tensor<4x?xf32>, tensor<4x6xf32>, tensor<4x6xf32>)
//   // return %[[RES1]], %[[RES2]], %[[RES3]]
//   return %0#0, %0#1, %0#2 : tensor<4x?xf32>, tensor<4x6xf32>, tensor<4x6xf32>
// }

// // C-HECK-LABEL: @unpack_dynamic
// func @unpack_dynamic(%input: tensor<?x?x2xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
//   // C-HECK: %[[SLICE1:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[-1, -1, 1]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<?x?x2xf32>) -> tensor<?x?x1xf32>
//   // C-HECK: "xla_hlo.reshape"(%[[SLICE1]]) : (tensor<?x?x1xf32>) -> tensor<?x?xf32>
//   // C-HECK: %[[SLICE2:.*]] = "xla_hlo.slice"(%{{.*}}) {limit_indices = dense<[-1, -1, 2]> : tensor<3xi64>, start_indices = dense<[0, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<?x?x2xf32>) -> tensor<?x?x1xf32>
//   // C-HECK: "xla_hlo.reshape"(%[[SLICE2]]) : (tensor<?x?x1xf32>) -> tensor<?x?xf32>

//   %0:2 = "tf.Unpack"(%input) {axis = -1} : (tensor<?x?x2xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
//   return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
// }

//===----------------------------------------------------------------------===//
// tf.UnsortedSegment{Max|Min|Prod|Sum} legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @unsorted_segment_sum
// CHECK-SAME: [[DATA:%.*]]: tensor<8x16x64xf32>
// CHECK-SAME: [[SI:%.*]]: tensor<8x16xi32>
func @unsorted_segment_sum(%data: tensor<8x16x64xf32>, %segment_ids : tensor<8x16xi32>) -> (tensor<4x64xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[ZERO:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[INIT:%.*]] = "xla_hlo.broadcast"([[ZERO]]) {broadcast_sizes = dense<[4, 64]> : tensor<2xi64>} : (tensor<f32>) -> tensor<4x64xf32>
  // CHECK: [[SCATTER:%.*]] = "xla_hlo.scatter"([[INIT]], [[SI]], [[DATA]]) ( {
  // CHECK: ^{{.*}}([[LHS:%.*]]: tensor<f32>, [[RHS:%.*]]: tensor<f32>):
  // CHECK:   [[ADD:%.*]] = xla_hlo.add [[LHS]], [[RHS]] : tensor<f32>
  // CHECK:   "xla_hlo.return"([[ADD]])
  // CHECK: }) {indices_are_sorted = false, scatter_dimension_numbers = {index_vector_dim = 2 : i64, inserted_window_dims = dense<0> : tensor<1xi64>, scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>, update_window_dims = dense<2> : tensor<1xi64>}, unique_indices = false} : (tensor<4x64xf32>, tensor<8x16xi32>, tensor<8x16x64xf32>) -> tensor<4x64xf32>
  // CHECK: return [[SCATTER]]
  %0 = "tf.UnsortedSegmentSum"(%data, %segment_ids, %num_segments) : (tensor<8x16x64xf32>, tensor<8x16xi32>, tensor<i32>) -> (tensor<4x64xf32>)
  return %0: tensor<4x64xf32>
}

// CHECK-LABEL: @unsorted_segment_prod
// CHECK-SAME: [[DATA:%.*]]: tensor<8x?x64xf32>
// CHECK-SAME: [[SI:%.*]]: tensor<?x16xi32>
func @unsorted_segment_prod(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[ONE:%.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: [[INIT:%.*]] = "xla_hlo.broadcast"([[ONE]]) {broadcast_sizes = dense<[4, 64]> : tensor<2xi64>} : (tensor<f32>) -> tensor<4x64xf32>
  // CHECK: [[SCATTER:%.*]] = "xla_hlo.scatter"([[INIT]], [[SI]], [[DATA]]) ( {
  // CHECK: ^{{.*}}([[LHS:%.*]]: tensor<f32>, [[RHS:%.*]]: tensor<f32>):
  // CHECK:   [[MUL:%.*]] = xla_hlo.multiply [[LHS]], [[RHS]] : tensor<f32>
  // CHECK:   "xla_hlo.return"([[MUL]])
  // CHECK: }) {indices_are_sorted = false, scatter_dimension_numbers = {index_vector_dim = 2 : i64, inserted_window_dims = dense<0> : tensor<1xi64>, scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>, update_window_dims = dense<2> : tensor<1xi64>}, unique_indices = false} : (tensor<4x64xf32>, tensor<?x16xi32>, tensor<8x?x64xf32>) -> tensor<4x?xf32>
  // CHECK: return [[SCATTER]]
  %0 = "tf.UnsortedSegmentProd"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  return %0: tensor<4x?xf32>
}

// CHECK-LABEL: @unsorted_segment_min
func @unsorted_segment_min(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: xla_hlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: xla_hlo.scatter
  // CHECK: xla_hlo.minimum
  %0 = "tf.UnsortedSegmentMin"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  return %0: tensor<4x?xf32>
}

// CHECK-LABEL: @unsorted_segment_max
func @unsorted_segment_max(%data: tensor<8x?x64xf32>, %segment_ids : tensor<?x16xi32>) -> (tensor<4x?xf32>) {
  %num_segments = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: xla_hlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: xla_hlo.scatter
  // CHECK: xla_hlo.maximum
  %0 = "tf.UnsortedSegmentMax"(%data, %segment_ids, %num_segments) : (tensor<8x?x64xf32>, tensor<?x16xi32>, tensor<i32>) -> (tensor<4x?xf32>)
  return %0: tensor<4x?xf32>
}

//===----------------------------------------------------------------------===//
// tf.GatherV2 legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @gather_v2
func @gather_v2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5xf32> {
  // CHECK: "xla_hlo.torch_index_select"(%arg0, %arg1) {batch_dims = 1 : i64, dim = 2 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>) -> tensor<16x2x5xf32>
  %0 = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5xf32>
  return %1 : tensor<16x2x5xf32>
}

// CHECK-LABEL: @gather_v2_dynamic
func @gather_v2_dynamic(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xi32>) -> tensor<*xf32> {
  // CHECK: "xla_hlo.torch_index_select"(%arg0, %arg1) {batch_dims = 1 : i64, dim = 2 : i64} : (tensor<?x?x?xf32>, tensor<?x?xi32>) -> tensor<*xf32>
  %0 = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<?x?x?xf32>, tensor<?x?xi32>, tensor<1xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// CHECK-LABEL: @gather_v2_unranked
func @gather_v2_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) -> tensor<*xf32> {
  // CHECK: tf.GatherV2
  %0 = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<*xf32>, tensor<*xi32>, tensor<1xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// tf.StridedSliceGrad legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: strided_slice_grad
// CHECK-SAME: [[GRAD:%.*]]: tensor<4x16x1022xf32>
func @strided_slice_grad(%grad: tensor<4x16x1022xf32>) -> tensor<4x128x1024xf32> {

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

  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"(%arg0) : (tensor<4x16x1022xf32>) -> tensor<4x16x1022xf32>
  // CHECK: [[REVERSE:%.*]] = "xla_hlo.reverse"([[RESHAPE]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<4x16x1022xf32>) -> tensor<4x16x1022xf32>
  // CHECK: [[ZERO:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "xla_hlo.pad"([[REVERSE]], [[ZERO]]) {edge_padding_high = dense<[0, 63, 2]> : tensor<3xi64>, edge_padding_low = dense<[0, 4, 0]> : tensor<3xi64>, interior_padding = dense<[0, 3, 0]> : tensor<3xi64>} : (tensor<4x16x1022xf32>, tensor<f32>) -> tensor<4x128x1024xf32>

  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 1, end_mask = 4} : (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<4x16x1022xf32>) -> tensor<4x128x1024xf32>
  // CHECK: return [[PAD]]
  return %0: tensor<4x128x1024xf32>
}

// CHECK-LABEL: strided_slice_grad_shrink_axis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<8xf32>
func @strided_slice_grad_shrink_axis_mask(%grad: tensor<8xf32>) -> tensor<4x8xf32> {
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

  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"([[GRAD]]) : (tensor<8xf32>) -> tensor<1x8xf32>
  // CHECK: [[ZEROS:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "xla_hlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[2, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<[1, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<2xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, shrink_axis_mask = 1} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<8xf32>) -> tensor<4x8xf32>

  // CHECK: return [[PAD]] : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: strided_slice_grad_new_axis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<1x2xf32>
func @strided_slice_grad_new_axis_mask(%grad: tensor<1x2xf32>) -> tensor<8xf32> {
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

  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"([[GRAD]]) : (tensor<1x2xf32>) -> tensor<2xf32>
  // CHECK: [[ZEROS:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "xla_hlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<2> : tensor<1xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<4> : tensor<1xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<1xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, new_axis_mask = 1} : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<1x2xf32>) -> tensor<8xf32>

  // CHECK: return [[PAD]] : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: strided_slice_grad_ellipsis_mask
// CHECK-SAME: [[GRAD:%.*]]: tensor<2x4x8xf32>
func @strided_slice_grad_ellipsis_mask(%grad: tensor<2x4x8xf32>) -> tensor<4x4x8xf32> {
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

  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"([[GRAD]]) : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  // CHECK: [[ZEROS:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[PAD:%.*]] = "xla_hlo.pad"([[RESHAPE]], [[ZEROS]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[2, 0, 0]> : tensor<3xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<0> : tensor<3xi64>
  // CHECK-DAG-SAME: interior_padding = dense<0> : tensor<3xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 0, end_mask = 0, ellipsis_mask = 2} : (tensor<3xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2x4x8xf32>) -> tensor<4x4x8xf32>

  // CHECK: return [[PAD]] : tensor<4x4x8xf32>
  return %0 : tensor<4x4x8xf32>
}


// CHECK-LABEL: strided_slice_grad_all_masks
// CHECK-SAME: [[GRAD:%.*]]: tensor<1x4x8x8x10x2x1xf32>
func @strided_slice_grad_all_masks(%grad: tensor<1x4x8x8x10x2x1xf32>) -> tensor<2x4x8x16x32x64xf32> {
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
  // CHECK: [[RESHAPE:%.*]] = "xla_hlo.reshape"([[GRAD]]) : (tensor<1x4x8x8x10x2x1xf32>) -> tensor<1x4x8x8x10x2xf32>
  // CHECK: [[ZERO:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // The edge_padding_low, edge_padding_high and interior_padding attributes of
  // xla_hlo.pad would reflect the padding required to get the shape of the
  // input of StridedSlice op.
  // CHECK: [[PAD:%.*]] = "xla_hlo.pad"([[RESHAPE]], [[ZERO]])
  // CHECK-DAG-SAME: edge_padding_low = dense<[1, 0, 0, 8, 0, 2]> : tensor<6xi64>
  // CHECK-DAG-SAME: edge_padding_high = dense<[0, 0, 0, 0, 22, 59]> : tensor<6xi64>
  // CHECK-DAG-SAME: interior_padding = dense<[0, 0, 0, 0, 0, 1]> : tensor<6xi64>
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %grad) {begin_mask = 16, end_mask = 8, shrink_axis_mask = 1, ellipsis_mask = 4, new_axis_mask = 66} : (tensor<6xi32>, tensor<7xi32>, tensor<7xi32>, tensor<7xi32>, tensor<1x4x8x8x10x2x1xf32>) -> tensor<2x4x8x16x32x64xf32>

  // CHECK: return [[PAD]] : tensor<2x4x8x16x32x64xf32>
  return %0 : tensor<2x4x8x16x32x64xf32>
}

// CHECK-LABEL: @tensor_scatter_update
func @tensor_scatter_update(%tensor: tensor<?x?x?xf32>, %indices: tensor<?x2xi32>, %updates: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: "xla_hlo.scatter"(%arg0, %arg1, %arg2) ( {
  // CHECK:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  // CHECK:    "xla_hlo.return"(%arg4) : (tensor<f32>) -> ()
  // CHECK:  })
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   index_vector_dim = 1 : i64
  // CHECK-SAME:   inserted_window_dims = dense<[0, 1]> : tensor<2xi64>
  // CHECK-SAME:   scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>
  // CHECK-SAME:   update_window_dims = dense<1> : tensor<1xi64>
  // CHECK-SAME: unique_indices = false
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<?x?x?xf32>, tensor<?x2xi32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

//===----------------------------------------------------------------------===//
// tf.RandomShuffle legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @random_shuffle_first_dim_1
// CHECK-SAME: [[INPUT:%.*]]: tensor<1x?xf32>
func @random_shuffle_first_dim_1(%input: tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = "tf.RandomShuffle"(%input) : (tensor<1x?xf32>) -> (tensor<1x?xf32>)
  // CHECK-NEXT: return [[INPUT]]
  return %0: tensor<1x?xf32>
}

// CHECK-LABEL: @random_shuffle_1D_16
// CHECK-SAME: [[INPUT:%.*]]: tensor<16xf32>
func @random_shuffle_1D_16(%input: tensor<16xf32>) -> tensor<16xf32> {
  // CHECK: [[SHAPE:%.*]] = xla_hlo.constant dense<16> : tensor<1xi64>
  // CHECK: [[LOWER:%.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: [[UPPER:%.*]] = xla_hlo.constant dense<-1> : tensor<i32>
  // CHECK: [[RNG:%.*]] = "xla_hlo.rng_uniform"([[LOWER]], [[UPPER]], [[SHAPE]])
  // CHECK: [[SORT:%.*]] = "xla_hlo.sort"([[RNG]], [[INPUT]]) ( {
  // CHECK: ^{{.*}}([[ARG1:%.*]]: tensor<i32>, [[ARG2:%.*]]: tensor<i32>, {{.*}}: tensor<f32>, {{.*}}: tensor<f32>):
  // CHECK:   "xla_hlo.compare"([[ARG1]], [[ARG2]]) {comparison_direction = "LT"}
  // CHECK: }) {dimension = -1 : i64, is_stable = true} : (tensor<16xi32>, tensor<16xf32>) -> tuple<tensor<16xi32>, tensor<16xf32>>
  // CHECK: [[RES:%.*]] = "xla_hlo.get_tuple_element"([[SORT]]) {index = 1 : i32}
  // CHECK: return [[RES]]
  %0 = "tf.RandomShuffle"(%input) : (tensor<16xf32>) -> (tensor<16xf32>)
  return %0: tensor<16xf32>
}

// CHECK-LABEL: @random_shuffle_1D_10240
func @random_shuffle_1D_10240(%input: tensor<10240xf32>) -> tensor<10240xf32> {
  // CHECK: xla_hlo.rng_uniform
  // CHECK: xla_hlo.sort
  // CHECK: xla_hlo.get_tuple_element
  // CHECK: xla_hlo.rng_uniform
  // CHECK: xla_hlo.sort
  // CHECK: xla_hlo.get_tuple_element
  %0 = "tf.RandomShuffle"(%input) : (tensor<10240xf32>) -> (tensor<10240xf32>)
  return %0: tensor<10240xf32>
}

// CHECK-LABEL: @random_shuffle_3D
// CHECK-SAME: [[INPUT:%.*]]: tensor<4x?x16xf32>
func @random_shuffle_3D(%input: tensor<4x?x16xf32>) -> tensor<4x?x16xf32> {
  // CHECK: [[INDICES:%.*]] = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>

  // CHECK: [[RNG_SHAPE:%.*]] = xla_hlo.constant dense<4> : tensor<1xi64>
  // CHECK: [[RNG_LOWER:%.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: [[RNG_UPPER:%.*]] = xla_hlo.constant dense<4> : tensor<i32>
  // CHECK: [[SWAPS:%.*]] = "xla_hlo.rng_uniform"([[RNG_LOWER]], [[RNG_UPPER]], [[RNG_SHAPE]])

  // CHECK: [[IV_INIT:%.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: [[WHILE_INIT:%.*]] = "xla_hlo.tuple"([[IV_INIT]], [[SWAPS]], [[INDICES]])

  // CHECK: [[WHILE_OUT:%.*]] = "xla_hlo.while"([[WHILE_INIT]]) ( {
  // CHECK: ^{{.*}}([[COND_ARG:%.*]]: tuple<tensor<i32>, tensor<4xi32>, tensor<4xi32>>):
  // CHECK:   [[IV:%.*]] = "xla_hlo.get_tuple_element"([[COND_ARG]]) {index = 0 : i32}
  // CHECK:   [[LIMIT:%.*]] = xla_hlo.constant dense<4> : tensor<i32>
  // CHECK:   [[CMP:%.*]] = "xla_hlo.compare"([[IV]], [[LIMIT]]) {comparison_direction = "LT"}
  // CHECK:   "xla_hlo.return"([[CMP]])
  // CHECK: },  {
  // CHECK: ^{{.*}}([[BODY_ARG:%.*]]: tuple<tensor<i32>, tensor<4xi32>, tensor<4xi32>>):
  // CHECK:   [[IV:%.*]] = "xla_hlo.get_tuple_element"([[BODY_ARG]]) {index = 0 : i32}
  // CHECK:   [[SWAPS:%.*]] = "xla_hlo.get_tuple_element"([[BODY_ARG]]) {index = 1 : i32}
  // CHECK:   [[INDICES:%.*]] = "xla_hlo.get_tuple_element"([[BODY_ARG]]) {index = 2 : i32}
  // CHECK:   [[SRC_IDX:%.*]] = "xla_hlo.dynamic-slice"([[INDICES]], [[IV]]) {slice_sizes = dense<1> : tensor<i64>} : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:   [[SWP_IDX:%.*]] = "xla_hlo.dynamic-slice"([[SWAPS]], [[IV]]) {slice_sizes = dense<1> : tensor<i64>} : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:   [[SWP:%.*]] = "xla_hlo.reshape"([[SWP_IDX]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK:   [[TGT_IDX:%.*]] = "xla_hlo.dynamic-slice"([[INDICES]], [[SWP]]) {slice_sizes = dense<1> : tensor<i64>}
  // CHECK:   [[INDICES1:%.*]] = "xla_hlo.dynamic-update-slice"([[INDICES]], [[TGT_IDX]], [[IV]]) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK:   [[INDICES2:%.*]] = "xla_hlo.dynamic-update-slice"([[INDICES1]], [[SRC_IDX]], [[SWP]]) : (tensor<4xi32>, tensor<1xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK:   [[ONE:%.*]] = xla_hlo.constant dense<1> : tensor<i32>
  // CHECK:   [[NEW_IV:%.*]] = xla_chlo.broadcast_add [[IV]], [[ONE]]
  // CHECK:   [[NEW_TUPLE:%.*]] = "xla_hlo.tuple"([[NEW_IV]], [[SWAPS]], [[INDICES2]])
  // CHECK:   "xla_hlo.return"([[NEW_TUPLE]])
  // CHECK: }) : (tuple<tensor<i32>, tensor<4xi32>, tensor<4xi32>>) -> tuple<tensor<i32>, tensor<4xi32>, tensor<4xi32>>

  // CHECK: [[SWAPED_INDICES:%.*]] = "xla_hlo.get_tuple_element"([[WHILE_OUT]]) {index = 2 : i32} : (tuple<tensor<i32>, tensor<4xi32>, tensor<4xi32>>) -> tensor<4xi32>
  // CHECK: [[GATHER:%.*]] = "xla_hlo.gather"([[INPUT]], [[SWAPED_INDICES]])
  // CHECK-SAME: dimension_numbers = {collapsed_slice_dims = dense<0> : tensor<1xi64>, index_vector_dim = 1 : i64, offset_dims = dense<[1, 2, 3]> : tensor<3xi64>, start_index_map = dense<0> : tensor<1xi64>}
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: slice_sizes = dense<[1, -1, 16]> : tensor<3xi64>
  // CHECK: (tensor<4x?x16xf32>, tensor<4xi32>) -> tensor<4x?x16xf32>

  // CHECK: return [[GATHER]]

  %0 = "tf.RandomShuffle"(%input) : (tensor<4x?x16xf32>) -> (tensor<4x?x16xf32>)
  return %0: tensor<4x?x16xf32>
}

//===----------------------------------------------------------------------===//
// tf.VariableShape legalization
//===----------------------------------------------------------------------===//

// CHECK-LABLE: @variable_shape32
func @variable_shape32(%input: tensor<!tf.resource<tensor<2x4x8xf32>>>) -> tensor<3xi32> {
  // CHECK: [[CST:%.*]] = xla_hlo.constant dense<[2, 4, 8]> : tensor<3xi32>
  // CHECK: [[CST_CAST:%.*]] = tensor_cast [[CST]]
  %0 = "tf.VariableShape"(%input) : (tensor<!tf.resource<tensor<2x4x8xf32>>>) -> (tensor<3xi32>)
  // CHECK: return [[CST_CAST]]
  return %0: tensor<3xi32>
}

// CHECK-LABLE: @variable_shape64
func @variable_shape64(%input: tensor<!tf.resource<tensor<2x4x8xf32>>>) -> tensor<3xi64> {
  // CHECK: [[CST:%.*]] = xla_hlo.constant dense<[2, 4, 8]> : tensor<3xi64>
  // CHECK: [[CST_CAST:%.*]] = tensor_cast [[CST]]
  %0 = "tf.VariableShape"(%input) : (tensor<!tf.resource<tensor<2x4x8xf32>>>) -> (tensor<3xi64>)
  // CHECK: return [[CST_CAST]]
  return %0: tensor<3xi64>
}

// CHECK-LABEL: @variable_shape_unknown_resource
func @variable_shape_unknown_resource(%input: tensor<!tf.resource>) -> tensor<?xi32> {
  // CHECK: tf.VariableShape
  %0 = "tf.VariableShape"(%input) : (tensor<!tf.resource>) -> (tensor<?xi32>)
  return %0: tensor<?xi32>
}

// CHECK-LABEL: @variable_shape_unknown_resource_shape
func @variable_shape_unknown_resource_shape(%input: tensor<!tf.resource<tensor<?x?xf32>>>) -> tensor<2xi32> {
  // CHECK: tf.VariableShape
  %0 = "tf.VariableShape"(%input) : (tensor<!tf.resource<tensor<?x?xf32>>>) -> (tensor<2xi32>)
  return %0: tensor<2xi32>
}

//===----------------------------------------------------------------------===//
// tf.AvgPool legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: avgpool_valid_padding
// CHECK-SAME: [[ARG:%.+]]: tensor<2x12x20x7xf16>
func @avgpool_valid_padding(%arg0: tensor<2x12x20x7xf16>) -> tensor<2x3x5x7xf16> {
  // CHECK: [[CONV32:%.+]] = "xla_hlo.convert"(%arg0) : (tensor<2x12x20x7xf16>) -> tensor<2x12x20x7xf32>
  // CHECK: [[INIT:%.+]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.+]] = "xla_hlo.reduce_window"([[CONV32]], [[INIT]]) ( {
  // CHECK: ^bb0([[ARG1:%.+]]: tensor<f32>, [[ARG2:%.+]]: tensor<f32>):
  // CHECK:   [[ADD:%.+]] = xla_hlo.add [[ARG1]], [[ARG2]]
  // CHECK:   "xla_hlo.return"([[ADD]])
  // CHECK: }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>} : (tensor<2x12x20x7xf32>, tensor<f32>) -> tensor<2x3x5x7xf32>
  // CHECK: [[COUNT:%.+]] = xla_hlo.constant dense<4.000000e+00> : tensor<f32>
  // CHECK: [[DIV:%.+]] = xla_chlo.broadcast_divide [[REDUCE]], [[COUNT]] {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<2x3x5x7xf32>, tensor<f32>) -> tensor<2x3x5x7xf32>
  // CHECK: [[CONV16:%.+]] = "xla_hlo.convert"([[DIV]]) : (tensor<2x3x5x7xf32>) -> tensor<2x3x5x7xf16>
  // CHECK: return [[CONV16]]
  %0 = "tf.AvgPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 4, 4, 1]} : (tensor<2x12x20x7xf16>) -> tensor<2x3x5x7xf16>
  return %0 : tensor<2x3x5x7xf16>
}

// CHECK-LABEL: avgpool_same_padding
func @avgpool_same_padding(%arg0: tensor<2x13x25x7xf32>) -> tensor<2x4x7x7xf32> {
  // CHECK: tf.AvgPool
  %0 = "tf.AvgPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 4, 1]} : (tensor<2x13x25x7xf32>) -> tensor<2x4x7x7xf32>
  return %0 : tensor<2x4x7x7xf32>
}

// CHECK-LABEL: xla_sharding
func @xla_sharding(%arg0: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // CHECK-NEXT: "xla_hlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", has_side_effect = false, xla_hlo.sharding = ""}
  %0 = "tf.XlaSharding"(%arg0) {_XlaSharding = ""} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: inplace_update_one
func @inplace_update_one(%arg0: tensor<8x4xf32>, %arg1: tensor<1x4xf32>, %arg2: tensor<1xi32>) -> tensor<8x4xf32> {
  // CHECK-DAG: [[CST:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[SLICE1:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE2:%.+]] = "xla_hlo.slice"(%arg1) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  // CHECK-DAG: [[RESHAPE1:%.+]] = "xla_hlo.reshape"([[SLICE1]])
  // CHECK-DAG: [[UPDATE:%.+]] = "xla_hlo.dynamic-update-slice"(%arg0, [[SLICE2]], [[RESHAPE1]], [[CST]])
  %0 = "tf.InplaceUpdate"(%arg0, %arg2, %arg1) : (tensor<8x4xf32>, tensor<1xi32>, tensor<1x4xf32>) -> tensor<8x4xf32>

  // CHECK: return [[UPDATE]]
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: inplace_update_three
func @inplace_update_three(%arg0: tensor<8x8x4xf32>, %arg1: tensor<3x8x4xf32>, %arg2: tensor<3xi32>) -> tensor<8x8x4xf32> {
  // CHECK-DAG: [[CST:%.+]] = xla_hlo.constant dense<0>
  // CHECK-DAG: [[SLICE1:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE2:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE3:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[SLICE4:%.+]] = "xla_hlo.slice"(%arg1) {limit_indices = dense<[1, 8, 4]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[SLICE5:%.+]] = "xla_hlo.slice"(%arg1) {limit_indices = dense<[2, 8, 4]> : tensor<3xi64>, start_indices = dense<[1, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[SLICE6:%.+]] = "xla_hlo.slice"(%arg1) {limit_indices = dense<[3, 8, 4]> : tensor<3xi64>, start_indices = dense<[2, 0, 0]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}
  // CHECK-DAG: [[RESHAPE1:%.+]] = "xla_hlo.reshape"([[SLICE1]])
  // CHECK-DAG: [[RESHAPE2:%.+]] = "xla_hlo.reshape"([[SLICE2]])
  // CHECK-DAG: [[RESHAPE3:%.+]] = "xla_hlo.reshape"([[SLICE3]])
  // CHECK-DAG: [[UPDATE1:%.+]] = "xla_hlo.dynamic-update-slice"(%arg0, [[SLICE4]], [[RESHAPE1]], [[CST]], [[CST]])
  // CHECK-DAG: [[UPDATE2:%.+]] = "xla_hlo.dynamic-update-slice"([[UPDATE1]], [[SLICE5]], [[RESHAPE2]], [[CST]], [[CST]])
  // CHECK-DAG: [[UPDATE3:%.+]] = "xla_hlo.dynamic-update-slice"([[UPDATE2]], [[SLICE6]], [[RESHAPE3]], [[CST]], [[CST]])
  %0 = "tf.InplaceUpdate"(%arg0, %arg2, %arg1) : (tensor<8x8x4xf32>, tensor<3xi32>, tensor<3x8x4xf32>) -> tensor<8x8x4xf32>

  // CHECK:  return [[UPDATE3]] : tensor<8x8x4xf32>
  return %0 : tensor<8x8x4xf32>
}


// CHECK-LABEL: xla_dynamic_update_slice
func @xla_dynamic_update_slice(%arg0: tensor<4x16xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2xi32>) -> tensor<4x16xf32> {
  // CHECK: [[SLICE0:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE0:%.+]] = "xla_hlo.reshape"([[SLICE0]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[SLICE1:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE1:%.+]] = "xla_hlo.reshape"([[SLICE1]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[DUS:%.+]] = "xla_hlo.dynamic-update-slice"(%arg0, %arg1, [[RESHAPE0]], [[RESHAPE1]]) : (tensor<4x16xf32>, tensor<2x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x16xf32>
  // CHECK: return [[DUS]]
  %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<4x16xf32>, tensor<2x4xf32>, tensor<2xi32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: xla_dynamic_update_slice2
func @xla_dynamic_update_slice2(%arg0: tensor<4xf32>, %arg1: tensor<2xf32>, %arg2: tensor<1xi32>) -> tensor<4xf32> {
  // CHECK: [[SLICE0:%.+]] = "xla_hlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: [[RESHAPE0:%.+]] = "xla_hlo.reshape"([[SLICE0]]) : (tensor<1xi32>) -> tensor<i32>
  // CHECK: [[DUS:%.+]] = "xla_hlo.dynamic-update-slice"(%arg0, %arg1, [[RESHAPE0]]) : (tensor<4xf32>, tensor<2xf32>, tensor<i32>) -> tensor<4xf32>
  // CHECK: return [[DUS]]
  %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<2xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

//===----------------------------------------------------------------------===//
// AllToAll op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @alltoall_basic
func @alltoall_basic(%input: tensor<10xf32>) -> tensor<10xf32> {
  %group_assignment = "tf.Const" () {
    value = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [3, 5, 6, 8]]> : tensor<3x4xi32>
  } : () -> tensor<3x4xi32>
  %result = "tf.AllToAll"(%input, %group_assignment) {T = f32, concat_dimension = 1 : i64, split_count = 2 : i64, split_dimension = 0 : i64} :  (tensor<10xf32>, tensor<3x4xi32>)  -> tensor<10xf32>
  // CHECK: xla_hlo.all_to_all
  // CHECK-SAME: replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7], [3, 5, 6, 8]]> : tensor<3x4xi64>
  return %result : tensor<10xf32>
}

//===----------------------------------------------------------------------===//
// Cumsum op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @cumsum_static
// CHECK-SAME: [[X:%.*]]: tensor<4xf32>
func @cumsum_static(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: [[AXIS:%.*]] = xla_hlo.constant dense<0> : tensor<i32>
  // CHECK: [[CONVERT_X:%.*]] = "xla_hlo.convert"([[X]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: [[INIT:%.*]] = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: [[REDUCE:%.*]] = "xla_hlo.reduce_window"([[CONVERT_X]], [[INIT]]) ( {
  // CHECK: ^bb0([[A:%.*]]: tensor<f32>, [[B:%.*]]: tensor<f32>):
  // CHECK:   [[SUM:%.*]] = xla_hlo.add [[A]], [[B]] : tensor<f32>
  // CHECK:   "xla_hlo.return"([[SUM]]) : (tensor<f32>) -> ()
  // CHECK: }) {padding = dense<{{\[\[}}3, 0]]> : tensor<1x2xi64>, window_dimensions = dense<4> : tensor<1xi64>, window_strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK: [[CONVERT_REDUCE:%.*]] = "xla_hlo.convert"([[REDUCE]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return [[CONVERT_REDUCE]]
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = false, reverse = false} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @cumsum_exclusive
func @cumsum_exclusive(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: "tf.Cumsum"
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = true, reverse = false} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @cumsum_reverse
func @cumsum_reverse(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: "tf.Cumsum"
  %0 = "tf.Const"() {_output_shapes = ["tfshape$"], device = "", dtype = i32, value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Cumsum"(%arg0, %0) {exclusive = false, reverse = true} : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @cumsum_dynamic
func @cumsum_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
  // CHECK: "tf.Cumsum"
  %0 = "tf.Cumsum"(%arg0, %arg1) : (tensor<?xf32>, tensor<i32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK:  func @qr([[VAL_0:%.*]]: tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>)
func @qr(%arg0: tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>) {
  // The tf.Qr lowering is a full algorithm that is not effective to verify with
  // FileCheck. Just verify that it converted.
  // TODO(laurenzo): Move this out of the mainline tf2xla conversion as it is
  // really only applicable to certain legacy uses.
  // CHECK-NOT: "tf.Qr"
  %0:2 = "tf.Qr"(%arg0) {full_matrices = false} : (tensor<500x100x75xf32>) -> (tensor<500x100x75xf32>, tensor<500x75x75xf32>)
  return %0#0, %0#1 : tensor<500x100x75xf32>, tensor<500x75x75xf32>
}
