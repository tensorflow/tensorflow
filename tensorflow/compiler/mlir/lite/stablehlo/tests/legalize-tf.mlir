// RUN: odml-to-stablehlo-opt --tf-stablehlo \
// RUN:     %s | FILECHECK_OPTS="" FileCheck %s

//===----------------------------------------------------------------------===//
// BatchNorm op legalizations.
//===----------------------------------------------------------------------===//

// -----

// fusedBatchNormV2 is almost identical to fusedBatchNormV3 (and uses the same
// code), so only do a couple of basic checks.

// CHECK-LABEL: fusedBatchNormV2_noTraining
func.func @fusedBatchNormV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "stablehlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV2_training
func.func @fusedBatchNormV2_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "stablehlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:5 = "tf.FusedBatchNormV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining
func.func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "stablehlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_noTraining_mixedPrecision
// CHECK-SAME:  ([[X:%.*]]: tensor<8x8x8x8xbf16>, [[SCALE:%.*]]: tensor<8xf32>, [[OFFSET:%.*]]: tensor<8xf32>, [[MEAN:%.*]]: tensor<8xf32>, [[VARIANCE:%.*]]: tensor<8xf32>)
func.func @fusedBatchNormV3_noTraining_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) {
  // CHECK: [[DUMMY:%.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0xf32>
  // CHECK: [[CONVERT_X:%.*]] = stablehlo.convert [[X]] : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK: [[Y:%.*]] = "stablehlo.batch_norm_inference"([[CONVERT_X]], [[SCALE]], [[OFFSET]], [[MEAN]], [[VARIANCE]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)
  // CHECK: [[Y_CONVERT:%.*]] = stablehlo.convert [[Y]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK: [[DUMMY_CAST:%.*]] = tensor.cast [[DUMMY]] : tensor<0xf32> to tensor<*xf32>
  // CHECK: return [[Y_CONVERT]], [[MEAN]], [[VARIANCE]], [[MEAN]], [[VARIANCE]], [[DUMMY_CAST]]
  func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training
func.func @fusedBatchNormV3_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "stablehlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: func @fusedBatchNormV3_training_batchVariance
func.func @fusedBatchNormV3_training_batchVariance(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "stablehlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: return %[[VAR]]
  func.return %0#4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_exponentialAvgFactor
func.func @fusedBatchNormV3_training_exponentialAvgFactor(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  // CHECK-DAG: %[[ALPHA:.*]] = stablehlo.constant dense<0.199999988>
  // CHECK-DAG: %[[BETA:.*]] = stablehlo.constant dense<8.000000e-01>
  // CHECK-DAG: %[[FACTOR:.*]] = stablehlo.constant dense<1.00195694>
  // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "stablehlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 0.8 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: %[[CORRECTED_VAR:.*]] = stablehlo.multiply %[[VAR]], %[[FACTOR]]

  // CHECK: %[[ALPHA_MUL_OLD_MEAN:.*]] = stablehlo.multiply %arg3, %[[ALPHA]]
  // CHECK: %[[BETA_MUL_BATCH_MEAN:.*]] = stablehlo.multiply %[[MEAN]], %[[BETA]]
  // CHECK: %[[NEW_BATCH_MEAN:.*]] = stablehlo.add %[[ALPHA_MUL_OLD_MEAN]], %[[BETA_MUL_BATCH_MEAN]]

  // CHECK: %[[ALPHA_MUL_OLD_VAR:.*]] = stablehlo.multiply %arg4, %[[ALPHA]]
  // CHECK: %[[BETA_MUL_CORRECTED_VAR:.*]] = stablehlo.multiply %[[CORRECTED_VAR]], %[[BETA]]
  // CHECK: %[[NEW_BATCH_VAR:.*]] = stablehlo.add %[[ALPHA_MUL_OLD_VAR]], %[[BETA_MUL_CORRECTED_VAR]]

  // CHECK: return %[[NEW_BATCH_MEAN]], %[[NEW_BATCH_VAR]], %[[MEAN]], %[[VAR]]
  func.return %0#1, %0#2, %0#3, %0#4 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_training_mixedPrecision
func.func @fusedBatchNormV3_training_mixedPrecision(%arg0: tensor<8x8x8x8xbf16>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK: stablehlo.convert %arg0 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK: stablehlo.convert {{.*}} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormV3_NCHW
func.func @fusedBatchNormV3_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "stablehlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
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
  // CHECK: "stablehlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<?x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
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
  // CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>

  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<8xf32>
  // CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>

  // CHECK-NEXT: %[[MUL2:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
  // CHECK-NEXT: %[[BCAST_MUL2:.+]] = stablehlo.broadcast_in_dim %[[MUL2]], {{.*}} : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[MUL3:.*]] = stablehlo.multiply %arg0, %[[BCAST_MUL2]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[MUL3]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGrad_Training
func.func @fusedBatchNormGrad_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%arg1, %arg2, %arg3, %arg4, %arg0) {{.*}}
  // CHECK-NEXT: return %[[GRAD_OPERAND]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGrad"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_noTraining
func.func @fusedBatchNormGradV2_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<8xf32>
  // CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[MUL2:.*]] = stablehlo.multiply %arg0, %[[BCAST]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[MUL2:.*]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_Training
func.func @fusedBatchNormGradV2_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%arg1, %arg2, %arg3, %arg4, %arg0) {{.*}}
  // CHECK-NEXT: return %[[GRAD_OPERAND]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_noTraining_mixed_precision
func.func @fusedBatchNormGradV2_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEST: %[[CST:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>
  // CHECK-NEST: %[[ADD:.*]] = stablehlo.add %arg4, %[[CST]] : tensor<8xf32>
  // CHECK-NEST: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>
  // CHECK-NEST: %[[MUL:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
  // CHECK-NEST: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEST: %[[MUL2:.*]] = stablehlo.multiply %arg0, %[[BCAST]] : tensor<8x8x8x8xf32>
  // CHECK-NEST: %[[CONVERT:.*]] = stablehlo.convert %[[MUL2]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEST: return %[[CONVERT]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV2_Training_mixed_precision
func.func @fusedBatchNormGradV2_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%[[CONVERT]], %arg2, %arg3, %arg4, %arg0) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.convert %[[GRAD_OPERAND]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[CONVERT]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV2"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining
func.func @fusedBatchNormGradV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
// CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<8xf32>
// CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>
// CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
// CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-NEXT: %[[MUL2:.*]] = stablehlo.multiply %arg0, %[[BCAST]] : tensor<8x8x8x8xf32>
// CHECK-NEXT: return %[[MUL2]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training
func.func @fusedBatchNormGradV3_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>) {
  // CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0xf32>
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%arg1, %arg2, %arg3, %arg4, %arg0) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[CAST:.*]] = tensor.cast %[[EPS]] : tensor<0xf32> to tensor<*xf32>
  // CHECK-NEXT: return %[[GRAD_OPERAND]], %[[EPS]], %[[CAST]] : tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<0xf32>, tensor<*xf32>)
  func.return %0#0, %0#3, %0#4 : tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_mixed_precision
func.func @fusedBatchNormGradV3_noTraining_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<8xf32>
  // CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[MUL2:.*]] = stablehlo.multiply %arg0, %[[BCAST]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.convert %[[MUL2]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[CONVERT]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training_mixed_precision
func.func @fusedBatchNormGradV3_Training_mixed_precision(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xbf16>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xbf16>) {
  // CHECK-NEXT: %[[CONVERT:.*]] = stablehlo.convert %arg1 : (tensor<8x8x8x8xbf16>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%[[CONVERT]], %arg2, %arg3, %arg4, %arg0) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[CONVERT2:.*]] = stablehlo.convert %[[GRAD_OPERAND]] : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xbf16>
  // CHECK-NEXT: return %[[CONVERT2]] : tensor<8x8x8x8xbf16>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xbf16>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xbf16>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_noTraining_NCHW
func.func @fusedBatchNormGradV3_noTraining_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[EPS:.*]] = stablehlo.constant dense<1.000000e-03> : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg4, %[[EPS]] : tensor<8xf32>
  // CHECK-NEXT: %[[RSQRT:.*]] = stablehlo.rsqrt %[[ADD]] : tensor<8xf32>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg2, %[[RSQRT]] : tensor<8xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[MUL]], dims = [1] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
  // CHECK-NEXT: %[[MUL2:.*]] = stablehlo.multiply %arg0, %[[BCAST]] : tensor<8x8x8x8xf32>
  // CHECK-NEXT: return %[[MUL2]] : tensor<8x8x8x8xf32>

  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: fusedBatchNormGradV3_Training_NCHW
func.func @fusedBatchNormGradV3_Training_NCHW(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: %[[GRAD_OPERAND:.*]], %[[GRAD_SCALE:.*]], %[[GRAD_OFFSET:.*]] = "stablehlo.batch_norm_grad"(%arg1, %arg2, %arg3, %arg4, %arg0) <{epsilon = 1.000000e-03 : f32, feature_index = 1 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: return %[[GRAD_OPERAND]] : tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

//===----------------------------------------------------------------------===//
// Bias op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @biasAdd_default
func.func @biasAdd_default(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xi32>) -> tensor<1x32x10x32xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BCAST]] : tensor<1x32x10x32xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<1x32x10x32xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_NHWC
func.func @biasAdd_NHWC(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<32xi32>) -> tensor<1x32x10x32xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BCAST]] : tensor<1x32x10x32xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<1x32x10x32xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_NCHW
func.func @biasAdd_NCHW(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xi32>) -> tensor<1x32x10x32xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BCAST]] : tensor<1x32x10x32xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<1x32x10x32xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_dynamic
func.func @biasAdd_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?x?xi32> {
  // CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<?x?x?x?xi32> -> tensor<4xindex>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %arg1, %[[SHAPE]], dims = [1] : (tensor<?xi32>, tensor<4xindex>) -> tensor<?x?x?x?xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BCAST]] : tensor<?x?x?x?xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<?x?x?x?xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW"} : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  func.return %0 : tensor<?x?x?x?xi32>
}

// -----

// CHECK-LABEL: func @biasAdd_partial_dynamic
func.func @biasAdd_partial_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<512xi32>) -> tensor<?x?x?x512xi32> {
  // CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<?x?x?x?xi32> -> tensor<4xindex>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %arg1, %[[SHAPE]], dims = [3] : (tensor<512xi32>, tensor<4xindex>) -> tensor<?x?x?x?xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[BCAST]] : tensor<?x?x?x?xi32>
  // CHECK-NEXT: %[[CAST:.*]] = tensor.cast %[[ADD]] : tensor<?x?x?x?xi32> to tensor<?x?x?x512xi32>
  // CHECK-NEXT: return %[[CAST]] : tensor<?x?x?x512xi32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<?x?x?x?xi32>, tensor<512xi32>) -> tensor<?x?x?x512xi32>
  func.return %0 : tensor<?x?x?x512xi32>
}


//===----------------------------------------------------------------------===//
// ClipByValue
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @clip
func.func @clip(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<f32> {
  // CHECK: [[VAL:%.+]] = stablehlo.clamp %arg1, %arg0, %arg2

  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return [[VAL]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @clip_dynamic
func.func @clip_dynamic(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: [[CLAMP:%.+]] = stablehlo.clamp %arg1, %arg0, %arg2
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @clip_static_broadcast
func.func @clip_static_broadcast(%arg0 : tensor<5xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<5xf32> {
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // CHECK-DAG: [[CLAMP:%.+]] = stablehlo.clamp [[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]]
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<5xf32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>

  // CHECK: return [[CLAMP]]
  func.return %0 : tensor<5xf32>
}


// CHECK-LABEL: @clip_dynamic_broadcast
func.func @clip_dynamic_broadcast(%arg0 : tensor<?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?xf32> {
  // CHECK: [[SHP:%.+]] = shape.shape_of %arg0
  // CHECK: [[SHPIDX:%.+]] = arith.index_cast [[SHP]] : tensor<1xindex> to tensor<1xi32>
  // CHECK-DAG: [[BROADCAST_MIN:%.+]] = stablehlo.dynamic_broadcast_in_dim %arg1, [[SHPIDX]], dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK-DAG: [[BROADCAST_MAX:%.+]] = stablehlo.dynamic_broadcast_in_dim %arg2, [[SHPIDX]], dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK-DAG: [[CLAMP:%.+]] = stablehlo.clamp [[BROADCAST_MIN]], %arg0, [[BROADCAST_MAX]]
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
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<12x12xf32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 : (tensor<4x3x4x3xf32>) -> tensor<12x12xf32>
  // CHECK-NEXT: %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<12xi32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[IOTA]], dims = [0] : (tensor<12xi32>) -> tensor<12x12xi32>
  // CHECK-NEXT: %[[IOTA2:.*]] = stablehlo.iota dim = 0 : tensor<12xi32>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast_in_dim %[[IOTA2]], dims = [1] : (tensor<12xi32>) -> tensor<12x12xi32>
  // CHECK-NEXT: %[[CMP:.*]] = stablehlo.compare  EQ, %[[BCAST]], %[[BCAST2]],  NOTYPE : (tensor<12x12xi32>, tensor<12x12xi32>) -> tensor<12x12xi1>
  // CHECK-NEXT: %[[SEL:.*]] = stablehlo.select %[[CMP]], %[[RESHAPE]], %[[CST0]] : tensor<12x12xi1>, tensor<12x12xf32>
  // CHECK-NEXT: %[[REDUCE:.*]] = stablehlo.reduce(%[[SEL]] init: %[[CST1]]) applies stablehlo.add across dimensions = [0] : (tensor<12x12xf32>, tensor<f32>) -> tensor<12xf32>
  // CHECK-NEXT: %[[RESHAPE2:.*]] = stablehlo.reshape %[[REDUCE]] : (tensor<12xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: return %[[RESHAPE2]] : tensor<4x3xf32>
  
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
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<42> : tensor<7x22x128xi32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<128> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CST2:.*]] = stablehlo.constant dense<140> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CST3:.*]] = stablehlo.constant dense<11> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CST4:.*]] = stablehlo.constant dense<0> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[IOTA0:.*]] = stablehlo.iota dim = 0 : tensor<22xi32>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[IOTA0]], dims = [1] : (tensor<22xi32>) -> tensor<1x22x128xi32>
  // CHECK-NEXT: %[[IOTA1:.*]] = stablehlo.iota dim = 0 : tensor<128xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %[[IOTA1]], dims = [2] : (tensor<128xi32>) -> tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB0:.*]] = stablehlo.subtract %[[CST3]], %[[BCAST0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[NEG0:.*]] = stablehlo.negate %[[SUB0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MIN0:.*]] = stablehlo.minimum %[[SUB0]], %[[CST4]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %[[MIN0]], %[[CST2]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MAX0:.*]] = stablehlo.maximum %[[SUB0]], %[[CST4]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB1:.*]] = stablehlo.subtract %[[CST1]], %[[MAX0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MIN1:.*]] = stablehlo.minimum %[[ADD0]], %[[SUB1]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  GE, %[[SUB0]], %[[CST4]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[SUB2:.*]] = stablehlo.subtract %[[CST1]], %[[MIN1]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SELECT0:.*]] = stablehlo.select %[[CMP0]], %[[SUB2]], %[[CST4]] : tensor<1x22x128xi1>, tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MAX1:.*]] = stablehlo.maximum %[[SUB0]], %[[CST4]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB2:.*]] = stablehlo.subtract %[[MAX1]], %[[SELECT0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MAX2:.*]] = stablehlo.maximum %[[NEG0]], %[[CST4]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB3:.*]] = stablehlo.subtract %[[MAX2]], %[[SELECT0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[BCAST1]], %[[SUB2]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[ADD2:.*]] = stablehlo.add %[[BCAST1]], %[[SUB3]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CMP1:.*]] = stablehlo.compare  GE, %[[ADD1]], %[[CST4]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP2:.*]] = stablehlo.compare  LT, %[[ADD1]], %[[CST1]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND0:.*]] = stablehlo.and %[[CMP1]], %[[CMP2]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP3:.*]] = stablehlo.compare  GE, %[[ADD2]], %[[CST4]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP4:.*]] = stablehlo.compare  LT, %[[ADD2]], %[[CST2]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND1:.*]] = stablehlo.and %[[CMP3]], %[[CMP4]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND2:.*]] = stablehlo.and %[[AND0]], %[[AND1]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[AND2]] : (tensor<1x22x128xi1>) -> tensor<22x128xi1>
  // CHECK-NEXT: %[[CONCAT0:.*]] = stablehlo.concatenate %[[ADD2]], %[[ADD1]], dim = 0 : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<2x22x128xi32>
  // CHECK-NEXT: %[[GATHER0:.*]] = "stablehlo.gather"(%arg0, %[[CONCAT0]]) <{dimension_numbers = #{{.*}}<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [1, 2]>, indices_are_sorted = false, slice_sizes = array<i64: 7, 1, 1>}> : (tensor<7x140x128xi32>, tensor<2x22x128xi32>) -> tensor<7x22x128xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast %[[RESHAPE0]], sizes = [7] : (tensor<22x128xi1>) -> tensor<7x22x128xi1>
  // CHECK-NEXT: %[[SELECT1:.*]] = stablehlo.select %[[BCAST1]], %[[GATHER0]], %[[CST0]] : tensor<7x22x128xi1>, tensor<7x22x128xi32>
  // CHECK-NEXT: return %[[SELECT1]] : tensor<7x22x128xi32>

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
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<42> : tensor<7x1x128xi32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<128> : tensor<1x1x128xi32>
  // CHECK-NEXT: %[[CST2:.*]] = stablehlo.constant dense<140> : tensor<1x1x128xi32>
  // CHECK-NEXT: %[[FALSE:.*]] = stablehlo.constant dense<0> : tensor<1x1x128xi32>
  // CHECK-NEXT: %[[IOTA0:.*]] = stablehlo.iota dim = 0 : tensor<128xi32>
  // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[IOTA0]] : (tensor<128xi32>) -> tensor<1x1x128xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  GE, %[[RESHAPE0]], %[[FALSE]] : (tensor<1x1x128xi32>, tensor<1x1x128xi32>) -> tensor<1x1x128xi1>
  // CHECK-NEXT: %[[CMP1:.*]] = stablehlo.compare  LT, %[[RESHAPE0]], %[[CST1]] : (tensor<1x1x128xi32>, tensor<1x1x128xi32>) -> tensor<1x1x128xi1>
  // CHECK-NEXT: %[[AND0:.*]] = stablehlo.and %[[CMP0]], %[[CMP1]] : tensor<1x1x128xi1>
  // CHECK-NEXT: %[[CMP2:.*]] = stablehlo.compare  GE, %[[RESHAPE0]], %[[FALSE]] : (tensor<1x1x128xi32>, tensor<1x1x128xi32>) -> tensor<1x1x128xi1>
  // CHECK-NEXT: %[[CMP3:.*]] = stablehlo.compare  LT, %[[RESHAPE0]], %[[CST2]] : (tensor<1x1x128xi32>, tensor<1x1x128xi32>) -> tensor<1x1x128xi1>
  // CHECK-NEXT: %[[AND1:.*]] = stablehlo.and %[[CMP2]], %[[CMP3]] : tensor<1x1x128xi1>
  // CHECK-NEXT: %[[AND2:.*]] = stablehlo.and %[[AND0]], %[[AND1]] : tensor<1x1x128xi1>
  // CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[AND2]] : (tensor<1x1x128xi1>) -> tensor<1x128xi1>
  // CHECK-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[RESHAPE0]], %[[RESHAPE0]], dim = 0 : (tensor<1x1x128xi32>, tensor<1x1x128xi32>) -> tensor<2x1x128xi32>
  // CHECK-NEXT: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %[[CONCAT]]) <{dimension_numbers = #{{.*}}<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [1, 2]>, indices_are_sorted = false, slice_sizes = array<i64: 7, 1, 1>}> : (tensor<7x140x128xi32>, tensor<2x1x128xi32>) -> tensor<7x1x128xi32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast %[[RESHAPE1]], sizes = [7] : (tensor<1x128xi1>) -> tensor<7x1x128xi1>
  // CHECK-NEXT: %[[SELECT0:.*]] = stablehlo.select %[[BCAST]], %[[GATHER]], %[[CST0]] : tensor<7x1x128xi1>, tensor<7x1x128xi32>
  // CHECK-NEXT: %[[RESHAPE2:.*]] = stablehlo.reshape %[[SELECT0]] : (tensor<7x1x128xi32>) -> tensor<7x128xi32>
  // CHECK-NEXT: return %[[RESHAPE2]] : tensor<7x128xi32>
  
  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<0> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "RIGHT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x128xi32>
  func.return %2: tensor<7x128xi32>
}

// -----

// CHECK-LABEL: func @matrix_diag_part_align_ll
func.func @matrix_diag_part_align_ll(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
  // CHECK-NEXT: %[[CST0:.*]] = stablehlo.constant dense<42> : tensor<7x22x128xi32>
  // CHECK-NEXT: %[[CST1:.*]] = stablehlo.constant dense<128> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CST2:.*]] = stablehlo.constant dense<140> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CST3:.*]] = stablehlo.constant dense<11> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[FALSE:.*]] = stablehlo.constant dense<0> : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[IOTA0:.*]] = stablehlo.iota dim = 0 : tensor<22xi32>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[IOTA0]], dims = [1] : (tensor<22xi32>) -> tensor<1x22x128xi32>
  // CHECK-NEXT: %[[IOTA1:.*]] = stablehlo.iota dim = 0 : tensor<128xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %[[IOTA1]], dims = [2] : (tensor<128xi32>) -> tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB0:.*]] = stablehlo.subtract %[[CST3]], %[[BCAST0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[NEG0:.*]] = stablehlo.negate %[[SUB0]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MAX0:.*]] = stablehlo.maximum %[[SUB0]], %[[FALSE]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB1:.*]] = stablehlo.subtract %[[MAX0]], %[[FALSE]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[MAX1:.*]] = stablehlo.maximum %[[NEG0]], %[[FALSE]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[SUB2:.*]] = stablehlo.subtract %[[MAX1]], %[[FALSE]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %[[BCAST1]], %[[SUB1]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[BCAST1]], %[[SUB2]] : tensor<1x22x128xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  GE, %[[ADD0]], %[[FALSE]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP1:.*]] = stablehlo.compare  LT, %[[ADD0]], %[[CST1]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND0:.*]] = stablehlo.and %[[CMP0]], %[[CMP1]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP2:.*]] = stablehlo.compare  GE, %[[ADD1]], %[[FALSE]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[CMP3:.*]] = stablehlo.compare  LT, %[[ADD1]], %[[CST2]] : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND1:.*]] = stablehlo.and %[[CMP2]], %[[CMP3]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[AND2:.*]] = stablehlo.and %[[AND0]], %[[AND1]] : tensor<1x22x128xi1>
  // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[AND2]] : (tensor<1x22x128xi1>) -> tensor<22x128xi1>
  // CHECK-NEXT: %[[CONCAT0:.*]] = stablehlo.concatenate %[[ADD1]], %[[ADD0]], dim = 0 : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<2x22x128xi32>
  // CHECK-NEXT: %[[GATHER0:.*]] = "stablehlo.gather"(%arg0, %[[CONCAT0]]) <{dimension_numbers = #{{.*}}<offset_dims = [0], collapsed_slice_dims = [1, 2], start_index_map = [1, 2]>, indices_are_sorted = false, slice_sizes = array<i64: 7, 1, 1>}> : (tensor<7x140x128xi32>, tensor<2x22x128xi32>) -> tensor<7x22x128xi32>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast %[[RESHAPE0]], sizes = [7] : (tensor<22x128xi1>) -> tensor<7x22x128xi1>
  // CHECK-NEXT: %[[SELECT0:.*]] = stablehlo.select %[[BCAST2]], %[[GATHER0]], %[[CST0]] : tensor<7x22x128xi1>, tensor<7x22x128xi32>
  // CHECK-NEXT: return %[[SELECT0]] : tensor<7x22x128xi32>

  %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
  %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
  %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
      T = i32, align = "LEFT_LEFT"
  } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
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
  // CHECK: %[[LE:.*]] = stablehlo.compare  LE, %{{.*}}, %{{.*}} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{.*}} = stablehlo.select %[[LE]], %{{.*}}, %{{.*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>
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
  // CHECK: %[[GE:.*]] = stablehlo.compare  GE, %{{.*}}, %{{.*}} : (tensor<1x22x128xi32>, tensor<1x22x128xi32>) -> tensor<1x22x128xi1>
  // CHECK: %{{.*}} = stablehlo.select %[[GE]], %{{.*}}, %{{.*}} : tensor<1x22x128xi1>, tensor<1x22x128xi32>

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
  // CHECK-NOT: MatrixDiagPartV3
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
  // CHECK: mhlo.erf(%arg0) {{.*}} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = "tf.Erf"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Erfc
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @erfc
func.func @erfc(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-NOT: tf.Erfc
  %0 = "tf.Erfc"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

//===----------------------------------------------------------------------===//
// Einsum.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @einsum
func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK: stablehlo.einsum
  %0 = "tf.Einsum"(%arg0, %arg1) {equation = "ab,bc->ac"} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  func.return %0: tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @unary_einsum
func.func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.constant{{.*}}1.000000e+00
  // CHECK: stablehlo.einsum{{.*}}",ab->aa"
  %0 = "tf.Einsum"(%arg0) {equation = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  func.return %0: tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// FloorDiv and FloorMod.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @floordiv_broadcast_i32
func.func @floordiv_broadcast_i32(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1> : tensor<2x3xi32>
  // CHECK-NEXT: %[[ZEROS0:.*]] = stablehlo.constant dense<0> : tensor<3xi32>
  // CHECK-NEXT: %[[ZEROS1:.*]] = stablehlo.constant dense<0> : tensor<2x3xi32>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: %[[DIV0:.*]] = stablehlo.divide %arg0, %[[BCAST0]] : tensor<2x3xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: %[[MUL0:.*]] = stablehlo.multiply %[[DIV0]], %[[BCAST1]] : tensor<2x3xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  NE, %[[MUL0]], %arg0 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[CMP1:.*]] = stablehlo.compare  LT, %arg0, %[[ZEROS1]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[CMP2:.*]] = stablehlo.compare  LT, %arg1, %[[ZEROS0]] : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast_in_dim %[[CMP2]], dims = [1] : (tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[CMP3:.*]] = stablehlo.compare  NE, %[[CMP1]], %[[BCAST2]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[AND0:.*]] = stablehlo.and %[[CMP0]], %[[CMP3]] : tensor<2x3xi1>
  // CHECK-NEXT: %[[SUB0:.*]] = stablehlo.subtract %[[DIV0]], %[[ONES]] : tensor<2x3xi32>
  // CHECK-NEXT: %[[SELECT0:.*]] = stablehlo.select %[[AND0]], %[[SUB0]], %[[DIV0]] : tensor<2x3xi1>, tensor<2x3xi32>
  // CHECK-NEXT: return %[[SELECT0]] : tensor<2x3xi32>

  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floordiv_reverse_broadcast_i32
func.func @floordiv_reverse_broadcast_i32(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1> : tensor<2x3xi32>
  // CHECK-NEXT: %[[ZEROS0:.*]] = stablehlo.constant dense<0> : tensor<2x3xi32>
  // CHECK-NEXT: %[[ZEROS1:.*]] = stablehlo.constant dense<0> : tensor<3xi32>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: %[[DIV0:.*]] = stablehlo.divide %[[BCAST0]], %arg1 : tensor<2x3xi32>
  // CHECK-NEXT: %[[MUL0:.*]] = stablehlo.multiply %[[DIV0]], %arg1 : tensor<2x3xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  NE, %[[MUL0]], %[[BCAST1]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[CMP1:.*]] = stablehlo.compare  LT, %arg0, %[[ZEROS1]] : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: %[[CMP2:.*]] = stablehlo.compare  LT, %arg1, %[[ZEROS0]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [1] : (tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[CMP3:.*]] = stablehlo.compare  NE, %[[BCAST2]], %[[CMP2]] : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK-NEXT: %[[AND0:.*]] = stablehlo.and %[[CMP0]], %[[CMP3]] : tensor<2x3xi1>
  // CHECK-NEXT: %[[SUB0:.*]] = stablehlo.subtract %[[DIV0]], %[[ONES]] : tensor<2x3xi32>
  // CHECK-NEXT: %[[SELECT0:.*]] = stablehlo.select %[[AND0]], %[[SUB0]], %[[DIV0]] : tensor<2x3xi1>, tensor<2x3xi32>
  // CHECK-NEXT: return %[[SELECT0]] : tensor<2x3xi32>

  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floordiv_f32
func.func @floordiv_f32(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT: %[[DIV:.*]] = stablehlo.divide %arg0, %arg0
  // CHECK-NEXT: %[[FLOOR:.*]] = stablehlo.floor %[[DIV]]
  // CHECK-NEXT: return %[[FLOOR]] : tensor<2xf32>
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0: tensor<2xf32>
}

// -----

// CHECK-LABEL: func @floordiv_bf16
func.func @floordiv_bf16(%arg0: tensor<2xbf16>) -> tensor<2xbf16> {
  // CHECK-NEXT: stablehlo.convert
  // CHECK-NEXT: stablehlo.convert
  // CHECK-NEXT: stablehlo.divide
  // CHECK-NEXT: stablehlo.floor
  // CHECK-NEXT: stablehlo.convert
  // CHECK-NEXT: return
  %0 = "tf.FloorDiv"(%arg0, %arg0) : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  func.return %0: tensor<2xbf16>
}

// -----

// CHECK-LABEL: func @floordiv_f16_broadcast
func.func @floordiv_f16_broadcast(%arg0: tensor<2x3xf16>, %arg1: tensor<3xf16>) -> tensor<2x3xf16> {
  // CHECK-NEXT: stablehlo.broadcast_in_dim
  // CHECK-NEXT: stablehlo.divide
  // CHECK-NEXT: stablehlo.floor
  // CHECK-NEXT: return
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
  func.return %0: tensor<2x3xf16>
}

// -----

// CHECK-LABEL: func @floordiv_dynamic
func.func @floordiv_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.divide
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.multiply
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.compare
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.and
  //
  // CHECK: %[[SELECT:.*]] = stablehlo.select
  // CHECK: return %[[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floordiv_unsigned
func.func @floordiv_unsigned(%arg0: tensor<?x?xui32>, %arg1: tensor<?xui32>) -> tensor<?x?xui32> {
  // CHECK: %[[RESULT:.*]] = shape.assuming
  // CHECK: %[[BCAST0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg0,
  // CHECK: %[[BCAST1:.*]] = stablehlo.dynamic_broadcast_in_dim %arg1,
  // CHECK: %[[DIV:.*]] = stablehlo.divide %[[BCAST0]], %[[BCAST1]]
  // CHECK: shape.assuming_yield %[[DIV]]
  // CHECK: return %[[RESULT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?x?xui32>, tensor<?xui32>) -> tensor<?x?xui32>
  func.return %0: tensor<?x?xui32>
}

// -----

// CHECK-LABEL: func @floordiv_int
func.func @floordiv_int(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.divide
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.multiply
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.compare
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.compare
  // CHECK: shape.assuming
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.and
  //
  // CHECK: %[[SELECT:.*]] = stablehlo.select
  // CHECK: return %[[SELECT]]
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_numerator
func.func @floormod_broadcast_numerator(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK: %[[REM:.*]] = stablehlo.remainder %[[BCAST0]], %arg1 : tensor<2x3xi32>
  // CHECK: %[[AND:.*]] = stablehlo.and
  // CHECK: %[[ADD:.*]] = stablehlo.add
  // CHECK: %[[SELECT:.*]] = stablehlo.select %[[AND]], %[[ADD]], %[[REM]]
  // CHECK-NEXT: return %[[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_broadcast_denominator
func.func @floormod_broadcast_denominator(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  // CHECK: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK: %[[REM:.*]] = stablehlo.remainder %arg0, %[[BCAST0]]
  // CHECK: %[[AND:.*]] = stablehlo.and
  // CHECK: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xi32>) -> tensor<2x3xi32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[BCAST1]], %[[REM]]
  // CHECK: %[[SELECT:.*]] = stablehlo.select %[[AND]], %[[ADD]], %[[REM]]
  // CHECK-NEXT: return %[[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  func.return %0: tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @floormod_unsigned_broadcast_denominator
func.func @floormod_unsigned_broadcast_denominator(%arg0: tensor<2x3xui32>, %arg1: tensor<3xui32>) -> tensor<2x3xui32> {
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<3xui32>) -> tensor<2x3xui32>
  // CHECK-NEXT: %[[REM:.*]] = stablehlo.remainder %arg0, %[[BCAST0]] : tensor<2x3xui32>
  // CHECK-NEXT: return %[[REM]] : tensor<2x3xui32>
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<2x3xui32>, tensor<3xui32>) -> tensor<2x3xui32>
  func.return %0: tensor<2x3xui32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_numerator
func.func @floormod_dynamic_broadcast_numerator_(%arg0: tensor<?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[REM:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.remainder
  // CHECK: shape.assuming {{.*}} {
  // CHECK: stablehlo.compare
  // CHECK: %[[AND:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.and
  // CHECK: %[[ADD:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.add
  // CHECK: %[[SELECT:.*]] = stablehlo.select %[[AND]], %[[ADD]], %[[REM]]
  // CHECK-NEXT: return %[[SELECT]]
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  func.return %0: tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @floormod_dynamic_broadcast_denominator
func.func @floormod_dynamic_broadcast_denominator_(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-NOT: tf.FloorMod
  // CHECK: %[[REM:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.remainder
  // CHECK: shape.assuming {{.*}} {
  // CHECK: stablehlo.compare
  // CHECK: %[[AND:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.and
  // CHECK: %[[ADD:.*]] = shape.assuming {{.*}} {
  // CHECK: stablehlo.add
  // CHECK: %[[SELECT:.*]] = stablehlo.select %[[AND]], %[[ADD]], %[[REM]]
  // CHECK-NEXT: return %[[SELECT]]
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
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<2x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ONES]], %[[SHAPE]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<2x?xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2x?xf32>
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
  // CHECK-NEXT: %[[ZEROS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %arg0 : tensor<2x?xf32> -> tensor<2xindex>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ZEROS]], %[[SHAPE]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<2x?xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2x?xf32>
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
  // CHECK: stablehlo.broadcast_in_dim %arg0
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}

//===----------------------------------------------------------------------===//
// Complex op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @complex
func.func @complex(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xcomplex<f32>> {
  // CHECK: stablehlo.complex
  %1 = "tf.Complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @imag
func.func @imag(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: stablehlo.imag
  %1 = "tf.Imag"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @real
func.func @real(%arg0: tensor<3xcomplex<f32>>) -> tensor<3xf32> {
  // CHECK: stablehlo.real
  %1 = "tf.Real"(%arg0) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

//===----------------------------------------------------------------------===//
// Concat op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @concat_v2
func.func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: stablehlo.concatenate %arg0, %arg1, dim = 0
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_neg_axis
func.func @concat_v2_neg_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: stablehlo.concatenate %arg0, %arg1, dim = 0

  %axis = "tf.Const"() { value = dense<-2> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_1d_axis
func.func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  // CHECK: stablehlo.concatenate %arg0, %arg1, dim = 1

  %axis = "tf.Const"() { value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi64>) -> tensor<3x6xf32>
  func.return %1 : tensor<3x6xf32>
}

// -----

// CHECK-LABEL: func @concat_v2_non_const_axis
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 12 : i32}} {
func.func @concat_v2_non_const_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %axis: tensor<i64>) -> tensor<3x6xf32> {
  // CHECK: "tf.ConcatV2"
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<3x6xf32>
  func.return %1 : tensor<3x6xf32>
}
}

//===----------------------------------------------------------------------===//
// Pad op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @padv2_1D
func.func @padv2_1D(%arg0: tensor<3xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1], high = [2], interior = [0]
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<6xf32>
  func.return %1 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func @padv2_2D
func.func @padv2_2D(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi64> } : () -> tensor<2x2xi64>
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1, 3], high = [2, 4], interior = [0, 0]
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi64>, tensor<f32>) -> tensor<6x9xf32>
  func.return %1 : tensor<6x9xf32>
}

// -----

// CHECK-LABEL: func @padv2_i32_paddings
func.func @padv2_i32_paddings(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<6x9xf32> {
  %padding = "tf.Const"() { value = dense<[[1,2],[3,4]]> : tensor<2x2xi32> } : () -> tensor<2x2xi32>
  // CHECK: stablehlo.pad %arg0, %arg1, low = [1, 3], high = [2, 4], interior = [0, 0]
  %1 = "tf.PadV2"(%arg0, %padding, %arg1) : (tensor<3x2xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<6x9xf32>
  func.return %1 : tensor<6x9xf32>
}

// -----

// CHECK-LABEL: func @padv2_dynamic
func.func @padv2_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1x2xi64>) -> tensor<?xf32> {
  // CHECK-NEXT: %[[ZEROS:.*]] = stablehlo.constant dense<0> : tensor<1xi64>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg2 : (tensor<1x2xi64>) -> tensor<2xi64>
  // CHECK-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHAPE]] [0:1] : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHAPE]] [1:2] : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_pad %arg0, %arg1, %[[SLICE0]], %[[SLICE1]], %[[ZEROS]] : (tensor<?xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<?xf32>

  %1 = "tf.PadV2"(%arg0, %arg2, %arg1) : (tensor<?xf32>, tensor<1x2xi64>, tensor<f32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// Identity op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @identity
func.func @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  %0 = "tf.Identity"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @identityN
func.func @identityN(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>) {
  // CHECK-NEXT: return %arg0, %arg1 : tensor<1xi32>, tensor<1xf32>
  %0:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xf32>) -> (tensor<1xi32>, tensor<1xf32>)
  func.return %0#0, %0#1: tensor<1xi32>, tensor<1xf32>
}

// -----

// CHECK-LABEL: func @stopgradient
func.func @stopgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  %0 = "tf.StopGradient"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @preventgradient
func.func @preventgradient(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  %0 = "tf.PreventGradient"(%arg0) {message = "fin gradients"} : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @checkNumerics
func.func @checkNumerics(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-NEXT: return %arg0 : tensor<1xf32>
  %0 = "tf.CheckNumerics"(%arg0) {message = "check numerics"} : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

//===----------------------------------------------------------------------===//
// InfeedDequeueTuple legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @infeed_dequeue_tuple
func.func @infeed_dequeue_tuple() -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>) {
  // CHECK: [[TOKEN:%.*]] = stablehlo.create_token  : !stablehlo.token
  // CHECK: [[INFEED:%.*]]:3 = "stablehlo.infeed"([[TOKEN]]) <{infeed_config = ""{{.*}}}> : (!stablehlo.token) -> (tensor<1x8x4x4xi32>, tensor<1x100x1xf32>, !stablehlo.token)
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
  // CHECK: "stablehlo.infeed"
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
  // CHECK: stablehlo.constant dense<0> : tensor<2xi32>
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
  func.return %0: tensor<2xi32>
}

// -----

// CHECK-LABEL: @const_dynamic_output
func.func @const_dynamic_output() -> tensor<*xi32> {
  // CHECK: [[CONST:%.*]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK: [[CAST:%.*]] = tensor.cast [[CONST]] : tensor<2xi32> to tensor<*xi32>
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> (tensor<*xi32>)
  // CHECK: return [[CAST]]
  func.return %0: tensor<*xi32>
}

// -----

// CHECK-LABEL: @opaque_const
func.func @opaque_const() -> tensor<!tf_type.variant<tensor<2xi32>>> {
  // CHECK-NOT: stablehlo.constant
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
  // CHECK: stablehlo.dot %[[A]], %[[B]]
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<5x7xf32>, tensor<7x11xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// -----

// CHECK-LABEL: matmul_transpose_b
// CHECK-SAME: (%[[A:.*]]: tensor<5x7xf32>, %[[B:.*]]: tensor<11x7xf32>)
func.func @matmul_transpose_b(%a: tensor<5x7xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_B:.*]] = stablehlo.transpose %[[B]], dims = [1, 0]
  // CHECK: stablehlo.dot %[[A]], %[[UPDATED_B]]
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = true} : (tensor<5x7xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// -----

// CHECK-LABEL: matmul_transpose_both
// CHECK-SAME: (%[[A:.*]]: tensor<7x5xf32>, %[[B:.*]]: tensor<11x7xf32>)
func.func @matmul_transpose_both(%a: tensor<7x5xf32>, %b: tensor<11x7xf32>) -> tensor<5x11xf32> {
  // CHECK: %[[UPDATED_A:.*]] = stablehlo.transpose %[[A]]
  // CHECK: %[[UPDATED_B:.*]] = stablehlo.transpose %[[B]]
  // CHECK: stablehlo.dot %[[UPDATED_A]], %[[UPDATED_B]]
  %0 = "tf.MatMul"(%a, %b) {transpose_a = true, transpose_b = true} : (tensor<7x5xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

  func.return %0 : tensor<5x11xf32>
}

// Verify that MatMul with ranked inputs are lowered to HLO.
// CHECK-LABEL: matmul_ranked
func.func @matmul_ranked(%a: tensor<?x7xf32>, %b: tensor<7x?xf32>) -> tensor<?x?xf32> {
  // CHECK: stablehlo.dot
  %0 = "tf.MatMul"(%a, %b) {transpose_a = false, transpose_b = false} : (tensor<?x7xf32>, tensor<7x?xf32>) -> tensor<?x?xf32>

  func.return %0 : tensor<?x?xf32>
}

// Verify SparseMatMul is legalized to dot.
// CHECK-LABEL: test_sparse_mat_mul
func.func @test_sparse_mat_mul(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  // CHECK: stablehlo.dot
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = false} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

// SparseMatMul where one operand needs to be transposed and the other one not.
//
// CHECK-LABEL:   @test_sparse_mat_mul_with_transpose
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x4xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<5x4xf32>
  // CHECK-SAME: -> tensor<3x5xf32>
  // CHECK:      %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG1]]
  // CHECK-SAME: dims = [1, 0]
  // CHECK-SAME: -> tensor<4x5xf32>
  // CHECK:      %[[RESULT:.*]] = stablehlo.dot %[[ARG0]], %[[TRANSPOSE]]
  // CHECK-SAME: -> tensor<3x5xf32>
  // CHECK:      return %[[RESULT]]
func.func @test_sparse_mat_mul_with_transpose(%arg0: tensor<3x4xf32>, %arg1: tensor<5x4xf32>) -> tensor<3x5xf32> {
  %0 = "tf.SparseMatMul"(%arg0, %arg1) {a_is_sparse = true, b_is_sparse = false, transpose_a = false, transpose_b = true} : (tensor<3x4xf32>, tensor<5x4xf32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>
}

// SparseMatMul where one operand needs to be casted and the other one not.
//
// CHECK-LABEL:   @test_sparse_mat_mul_with_cast
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x4xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<4x5xbf16>
  // CHECK-SAME: -> tensor<3x5xf32>
  // CHECK:      %[[CAST:.*]] = stablehlo.convert %[[ARG1]]
  // CHECK-SAME: -> tensor<4x5xf32>
  // CHECK:      %[[RESULT:.*]] = stablehlo.dot %[[ARG0]], %[[CAST]]
  // CHECK-SAME: -> tensor<3x5xf32>
  // CHECK:      return %[[RESULT]]
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
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<-2147483648> : tensor<i32>
  // CHECK: "stablehlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK-SAME: <{window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 4, 4, 1>}>
  // CHECK: stablehlo.maximum
  // CHECK: stablehlo.return

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
  // CHECK: %[[INIT:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: "stablehlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK-SAME: <{window_dimensions = array<i64: 1, 1, 2, 2, 1>, window_strides = array<i64: 1, 1, 4, 4, 1>}>
  // CHECK: stablehlo.maximum
  // CHECK: stablehlo.return

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
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "stablehlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]])
  // CHECK-SAME: <{window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = stablehlo.compare  GE, %[[VALUE_A]], %[[VALUE_B]], NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: stablehlo.return %[[SELECT_RESULT]] : tensor<i1>
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = stablehlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: stablehlo.return %[[SELECT_RESULT]] : tensor<f32>
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
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RESULT:.*]] = "stablehlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]])
  // CHECK-SAME: <{window_dimensions = array<i64: 1, 1, 2, 2, 1>, window_strides = array<i64: 1, 1, 2, 2, 1>}> ({
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = stablehlo.compare  GE, %[[VALUE_A]], %[[VALUE_B]], NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK: stablehlo.return %[[SELECT_RESULT]] : tensor<i1>
  // CHECK: },  {
  // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
  // CHECK: %[[SELECT_RESULT:.*]] = stablehlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
  // CHECK: stablehlo.return %[[SELECT_RESULT]] : tensor<f32>
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
  // CHECK-NEXT: %[[IOTA0:.*]] = stablehlo.iota dim = 0 : tensor<5xi32>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[IOTA0]], dims = [1] : (tensor<5xi32>) -> tensor<3x5xi32>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<3xi32>) -> tensor<3x5xi32>
  // CHECK-NEXT: %[[CMP0:.*]] = stablehlo.compare  EQ, %[[BCAST1]], %[[BCAST0]],  NOTYPE : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<3x5xi1>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast %arg1, sizes = [3, 5] : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[BCAST3:.*]] = stablehlo.broadcast %arg2, sizes = [3, 5] : (tensor<f32>) -> tensor<3x5xf32>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.select %[[CMP0]], %[[BCAST2]], %[[BCAST3]] : tensor<3x5xi1>, tensor<3x5xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<3x5xf32>
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
  // CHECK: [[TOKEN:%.*]] = stablehlo.create_token  : !stablehlo.token
  // CHECK: "stablehlo.outfeed"([[VAL_0]], [[VAL_1]], [[TOKEN]]) <{outfeed_config = ""}> : (tensor<3xi32>, tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
  "tf.OutfeedEnqueueTuple"(%data_1, %data_2) : (tensor<3xi32>, tensor<4xf32>) -> ()
  func.return
}

//===----------------------------------------------------------------------===//
// Pack op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @pack
func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: stablehlo.reshape {{.*}} : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: stablehlo.reshape {{.*}} : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: stablehlo.concatenate {{.*}}, {{.*}}, dim = 0

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

  // CHECK: [[VAL:%.+]] = stablehlo.reverse %arg0, dims = [0] : tensor<5xi32>
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi32>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_64
func.func @reverse_func_64(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  %axis = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> (tensor<1xi64>)

  // CHECK: [[VAL:%.+]] = stablehlo.reverse %arg0, dims = [0] : tensor<5xi32>
  %reversed = "tf.ReverseV2"(%arg0, %axis) : (tensor<5xi32>, tensor<1xi64>) -> tensor<5xi32>

  // CHECK: return [[VAL]] : tensor<5xi32>
  func.return %reversed : tensor<5xi32>
}

// -----

// CHECK-LABEL: @reverse_func_neg
func.func @reverse_func_neg(%arg0: tensor<5x5xi32>) -> tensor<5x5xi32> {
  %axis = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: [[VAL:%.+]] = stablehlo.reverse %arg0, dims = [1] : tensor<5x5xi32>
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
  // CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK-DAG: %[[PRED:.*]] = stablehlo.compare GT, %arg0, %[[ZERO]]
  // CHECK-DAG: %[[EXP:.*]] = stablehlo.exponential_minus_one %arg0
  // CHECK: %[[RESULT:.*]] = stablehlo.select %[[PRED]], %arg0, %[[EXP]]
  // CHECK: return %[[RESULT]]
  %0 = "tf.Elu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

// -----

// CHECK-LABEL: func @elu_grad
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<4x8xf32>, %[[FEATURES:.*]]: tensor<?x?xf32>)
func.func @elu_grad(%gradients: tensor<4x8xf32>, %features: tensor<?x?xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[BCAST0:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ZERO]], {{.*}}, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[PRED:.*]] = stablehlo.compare  GT, %[[FEATURES]], %[[BCAST0]]
  // CHECK-DAG: %[[BCAST1:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ONE]], {{.*}}, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[ADD1:.*]] = stablehlo.add %[[FEATURES]], %[[BCAST1]]
  // CHECK-DAG: %[[MULGRAD:.*]] = stablehlo.multiply %[[GRADIENTS]], %[[ADD1]] : (tensor<4x8xf32>, tensor<?x?xf32>) -> tensor<4x8xf32>
  // CHECK: %[[RESULT:.*]] = stablehlo.select %[[PRED]], %[[GRADIENTS]], %[[MULGRAD]]
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
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<1xi32>
  // CHECK: stablehlo.maximum %arg0, %[[ZERO]]
  %0 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu_unsigned
func.func @relu_unsigned(%arg0: tensor<?xui32>) -> tensor<?xui32> {
  // CHECK: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<ui32>
  // CHECK: %[[BCAST0:.*]] = stablehlo.dynamic_broadcast_in_dim %[[ZERO]], {{.*}}, dims = []
  // CHECK: stablehlo.maximum %arg0, %[[BCAST0]]
  %0 = "tf.Relu"(%arg0) : (tensor<?xui32>) -> tensor<?xui32>
  func.return %0: tensor<?xui32>
}

// -----

// CHECK-LABEL: func @relu6
func.func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG: %[[SIX:.*]] = stablehlo.constant dense<6> : tensor<i32>
  // CHECK: stablehlo.clamp %[[ZERO]], %arg0, %[[SIX]]
  %0 = "tf.Relu6"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @relu6_unsigned
func.func @relu6_unsigned(%arg0: tensor<?xui32>) -> tensor<?xui32> {
  // CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<ui32>
  // CHECK-DAG: %[[SIX:.*]] = stablehlo.constant dense<6> : tensor<ui32>
  // CHECK: stablehlo.clamp %[[ZERO]], %arg0, %[[SIX]]
  %0 = "tf.Relu6"(%arg0) : (tensor<?xui32>) -> tensor<?xui32>
  func.return %0: tensor<?xui32>
}

// -----

// CHECK-LABEL: func @leaky_relu
func.func @leaky_relu(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = stablehlo.constant dense<2.000000e-01>
    // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
    // CHECK-NEXT: %[[LEAKY:.*]] = stablehlo.multiply %arg0, %[[ALPHA]]
    // CHECK-NEXT: %[[CMP:.*]] = stablehlo.compare GT, %arg0, %[[ZERO]]
    // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[CMP]], %arg0, %[[LEAKY]]
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4x3xf32>
    %0 = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    func.return %0 : tensor<1x4x4x3xf32>
}

// -----

// CHECK-LABEL: func @leaky_relu_grad
func.func @leaky_relu_grad(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {tf.entry_function = {}} {
    // CHECK-NEXT: %[[ALPHA:.*]] = stablehlo.constant dense<2.000000e-01>
    // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00>
    // CHECK-NEXT: %[[LEAKYGRAD:.*]] = stablehlo.multiply %[[GRADIENT:.*]], %[[ALPHA]]
    // CHECK-NEXT: %[[CMP:.*]] = stablehlo.compare GT, %[[INP:.*]], %[[ZERO]], NOTYPE
    // CHECK-NEXT: %[[RES:.*]] = stablehlo.select %[[CMP]], %[[GRADIENT]], %[[LEAKYGRAD]]
    // CHECK-NEXT: return %[[RES]] : tensor<1x4x4xf32>
    %0 = "tf.LeakyReluGrad"(%arg0, %arg1) {alpha = 2.000000e-01 : f32, device = ""} : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    func.return %0 : tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: func @softsign
func.func @softsign(%arg0: tensor<4x10xf32>) -> tensor<4x10xf32> {
    // CHECK-NEXT: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %{{.*}}
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ONE]]
    // CHECK-NEXT: %[[DIV:.*]] = stablehlo.divide %{{.*}}, %[[ADD]]
    // CHECK-NEXT: return %[[DIV]] : tensor<4x10xf32>
    %0 = "tf.Softsign"(%arg0) : (tensor<4x10xf32>) -> tensor<4x10xf32>
    func.return %0 : tensor<4x10xf32>
}

// -----

// CHECK-LABEL: func @softsign_grad
func.func @softsign_grad(%arg0: tensor<4x10xf32>, %arg1: tensor<4x10xf32>) -> tensor<4x10xf32> {

    // CHECK-NEXT: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %{{.*}} : tensor<4x10xf32>
    // CHECK-NEXT: %[[BROADCAST_ADD:.*]] = stablehlo.add %[[ABS]], %[[ONE]]
    // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[BROADCAST_ADD]], %[[BROADCAST_ADD]]
    // CHECK-NEXT: %[[BROADCAST_DIV:.*]] = stablehlo.divide %{{.*}}, %[[MUL]]
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
  // CHECK: %[[AXIS_SIZE:.*]] = stablehlo.constant dense<512> : tensor<i32>
  // CHECK: %[[T1:.+]] = stablehlo.remainder %arg1, %[[AXIS_SIZE]] : tensor<i32>
  // CHECK: %[[T2:.+]] = stablehlo.add %[[T1]], %[[AXIS_SIZE]] : tensor<i32>
  // CHECK: %[[T3:.+]] = stablehlo.remainder %[[T2]], %[[AXIS_SIZE]] : tensor<i32>
  // CHECK: %[[CONCAT:.+]] = stablehlo.concatenate %arg0, %arg0, dim = 0
  // CHECK: %[[OFFSET:.+]] = stablehlo.subtract %[[AXIS_SIZE]], %[[T3]] : tensor<i32>
  // CHECK: stablehlo.dynamic_slice %[[CONCAT]], %[[OFFSET]], sizes = [512]
  %0 = "tf.Roll"(%arg0, %shift, %axis) {device = ""} : (tensor<512xi32>, tensor<i32>, tensor<i32>) -> tensor<512xi32>
  func.return %0 : tensor<512xi32>
}

//===----------------------------------------------------------------------===//
// Select op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @select_batch_static
func.func @select_batch_static(%arg0: tensor<2xi1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0]
  // CHECK: stablehlo.select %[[BCAST]], %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_r1
func.func @select_batch_static_r1(%arg0: tensor<i1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: stablehlo.select %arg0, %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_static_all_same
func.func @select_batch_static_all_same(%arg0: tensor<2x6x8xi1>, %arg1: tensor<2x6x8xi32>, %arg2: tensor<2x6x8xi32>) -> tensor<2x6x8xi32> {
  // CHECK: stablehlo.select %arg0, %arg1, %arg2
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2x6x8xi1>, tensor<2x6x8xi32>, tensor<2x6x8xi32>) -> tensor<2x6x8xi32>
  func.return %0: tensor<2x6x8xi32>
}

// -----

// CHECK-LABEL: func @select_batch_dynamic_r1
func.func @select_batch_dynamic_r1(%arg0: tensor<?xi1>, %arg1: tensor<?x?x8xi32>, %arg2: tensor<?x?x8xi32>) -> tensor<?x?x8xi32> {
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  // CHECK-NEXT: %[[SHAPE1:.*]] = shape.shape_of %arg1 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPE2:.*]] = shape.shape_of %arg2 : tensor<?x?x8xi32> -> tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ1:.*]] = shape.cstr_eq %[[SHAPE1]], %[[SHAPE2]] : tensor<3xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[HEAD:.*]], %[[TAIL:.*]] = "shape.split_at"(%[[SHAPE1]], %[[C1]]) : (tensor<3xindex>, index) -> (tensor<1xindex>, tensor<2xindex>)
  // CHECK-NEXT: %[[SHAPEEQ2:.*]] = shape.cstr_eq %[[SHAPE0]], %[[HEAD]] : tensor<1xindex>, tensor<1xindex>
  // CHECK-NEXT: %[[SHAPEEQ:.*]] = shape.assuming_all %[[SHAPEEQ1]], %[[SHAPEEQ2]]
  // CHECK-NEXT: %[[ASSUMING:.*]] = shape.assuming %[[SHAPEEQ]] -> (tensor<?x?x8xi32>) {
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %arg0, %[[SHAPE1]], dims = [0]
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %[[BCAST]], %arg1, %arg2 : tensor<?x?x8xi1>, tensor<?x?x8xi32>
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
  // CHECK-NEXT: %[[SHAPEEQ3:.*]] = shape.cstr_eq %[[SHAPE1]], %[[SHAPE2]], %[[SHAPE0]], %[[SHAPE1]] : tensor<3xindex>, tensor<3xindex>, tensor<3xindex>, tensor<3xindex>
  // CHECK-NEXT: %[[SHAPEEQ:.*]] = shape.assuming %[[SHAPEEQ3]]
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %arg0, %arg1, %arg2 : tensor<?x?x8xi1>, tensor<?x?x8xi32>
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
  // CHECK: stablehlo.select
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  func.return %0: tensor<2x?x8xi32>
}

//===----------------------------------------------------------------------===//
// Fast Fourier Transform op legalization.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @fft_1D
func.func @fft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: stablehlo.fft %arg0, type = FFT, length = [8]
  %0 = "tf.FFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @ifft_1D
func.func @ifft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK: stablehlo.fft %arg0, type = IFFT, length = [8]
  %0 = "tf.IFFT"(%arg0) : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  func.return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D
func.func @rfft_1D(%arg0: tensor<8xf32>) -> tensor<5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: stablehlo.fft %arg0, type = RFFT, length = [8]
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<8xf32>, tensor<1xi32>) -> tensor<5xcomplex<f32>>
  func.return %0 : tensor<5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_padded
func.func @rfft_1D_padded(%arg0: tensor<7xf32>) -> tensor<5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[PADDED:.*]] = stablehlo.pad %arg0, %{{.*}}, low = [0], high = [1], interior = [0]
  // CHECK: stablehlo.fft %[[PADDED]], type = RFFT, length = [8]
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<7xf32>, tensor<1xi32>) -> tensor<5xcomplex<f32>>
  func.return %0 : tensor<5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @rfft_1D_sliced
func.func @rfft_1D_sliced(%arg0: tensor<2x9xf32>) -> tensor<2x5xcomplex<f32>> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = stablehlo.slice %arg0 [0:2, 0:8]
  // CHECK: stablehlo.fft %[[SLICED]], type = RFFT, length = [8]
  %0 = "tf.RFFT"(%arg0, %fftlength) : (tensor<2x9xf32>, tensor<1xi32>) -> tensor<2x5xcomplex<f32>>
  func.return %0 : tensor<2x5xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @irfft_1D
func.func @irfft_1D(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xf32> {
  %fftlength = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: %[[SLICED:.*]] = stablehlo.slice %arg0 [0:5]
  // CHECK: stablehlo.fft %[[SLICED]], type = IRFFT, length = [8]
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
  // CHECK: stablehlo.transpose
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_3d_int32
func.func @transpose_3d_int32(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  // CHECK: stablehlo.transpose
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @transpose_3d
func.func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> (tensor<3xi64>)
  // CHECK: stablehlo.transpose
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @transpose_dynamic_2d
func.func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: stablehlo.transpose
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

//===----------------------------------------------------------------------===//
// Unary op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @abs
func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.abs %arg0 : tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @abs_dynamic
func.func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.abs %arg0 : tensor<?xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @acos
func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK: %[[TEMP_1:.*]] = stablehlo.subtract %[[TEMP_0]], %arg0 : tensor<2xf32>
  // CHECK: %[[TEMP_2:.*]] = stablehlo.add %arg0, %[[TEMP_0]] : tensor<2xf32>
  // CHECK: %[[TEMP_3:.*]] = stablehlo.multiply %[[TEMP_1]], %[[TEMP_2]] : tensor<2xf32>
  // CHECK: %[[TEMP_4:.*]] = stablehlo.sqrt %[[TEMP_3]] : tensor<2xf32>
  // CHECK: %[[TEMP_5:.*]] = stablehlo.atan2 %[[TEMP_4]], %arg0 : tensor<2xf32>
  // CHECK: return %[[TEMP_5]] : tensor<2xf32>
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @acos_complex
func.func @acos_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
// CHECK-NEXT: %[[TEMP_cst:.*]] = stablehlo.constant dense<4.33680869E-19> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_0:.*]] = stablehlo.constant dense<0.693147182> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_1:.*]] = stablehlo.constant dense<2.30584283E+20> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_2:.*]] = stablehlo.constant dense<2.30584274E+12> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_3:.*]] = stablehlo.constant dense<2.30584285E+30> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_4:.*]] = stablehlo.constant dense<2.30584287E+18> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_5:.*]] = stablehlo.constant dense<1.500000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_6:.*]] = stablehlo.constant dense<0x7F800000> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_7:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_9:.*]] = stablehlo.constant dense<1.41421354> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_10:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_cst_11:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_0:.*]] = stablehlo.real %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT: %[[TEMP_1:.*]] = stablehlo.abs %[[TEMP_0]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_2:.*]] = stablehlo.imag %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT: %[[TEMP_3:.*]] = stablehlo.abs %[[TEMP_2]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_4:.*]] = stablehlo.maximum %[[TEMP_1]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_5:.*]] = stablehlo.compare  GE, %[[TEMP_4]], %[[TEMP_cst_5:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_6:.*]] = stablehlo.compare  LE, %[[TEMP_1]], %[[TEMP_cst_11:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_7:.*]] = stablehlo.add %[[TEMP_1]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_8:.*]] = stablehlo.abs %[[TEMP_7]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_9:.*]] = stablehlo.maximum %[[TEMP_8]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_10:.*]] = stablehlo.minimum %[[TEMP_8]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_11:.*]] = stablehlo.compare  EQ, %[[TEMP_9]], %[[TEMP_10]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_12:.*]] = stablehlo.multiply %[[TEMP_9]], %[[TEMP_cst_4:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_13:.*]] = stablehlo.divide %[[TEMP_10]], %[[TEMP_9]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_14:.*]] = stablehlo.multiply %[[TEMP_13]], %[[TEMP_13]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_15:.*]] = stablehlo.add %[[TEMP_14]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_16:.*]] = stablehlo.sqrt %[[TEMP_15]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_17:.*]] = stablehlo.compare  EQ, %[[TEMP_16]], %[[TEMP_cst_11:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_18:.*]] = stablehlo.compare  GT, %[[TEMP_14]], %[[TEMP_cst_8:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_19:.*]] = stablehlo.and %[[TEMP_17]], %[[TEMP_18]] : tensor<2xi1>
// CHECK-NEXT: %[[TEMP_20:.*]] = stablehlo.multiply %[[TEMP_9]], %[[TEMP_14]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_21:.*]] = stablehlo.divide %[[TEMP_20]], %[[TEMP_cst_9:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_22:.*]] = stablehlo.add %[[TEMP_9]], %[[TEMP_21]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_23:.*]] = stablehlo.multiply %[[TEMP_9]], %[[TEMP_16]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_24:.*]] = stablehlo.select %[[TEMP_19]], %[[TEMP_22]], %[[TEMP_23]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_25:.*]] = stablehlo.select %[[TEMP_11]], %[[TEMP_12]], %[[TEMP_24]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_26:.*]] = stablehlo.subtract %[[TEMP_1]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_27:.*]] = stablehlo.abs %[[TEMP_26]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_28:.*]] = stablehlo.maximum %[[TEMP_27]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_29:.*]] = stablehlo.minimum %[[TEMP_27]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_30:.*]] = stablehlo.compare  EQ, %[[TEMP_28]], %[[TEMP_29]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_31:.*]] = stablehlo.multiply %[[TEMP_28]], %[[TEMP_cst_4:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_32:.*]] = stablehlo.divide %[[TEMP_29]], %[[TEMP_28]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_33:.*]] = stablehlo.multiply %[[TEMP_32]], %[[TEMP_32]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_34:.*]] = stablehlo.add %[[TEMP_33]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_35:.*]] = stablehlo.sqrt %[[TEMP_34]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_36:.*]] = stablehlo.compare  EQ, %[[TEMP_35]], %[[TEMP_cst_11:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_37:.*]] = stablehlo.compare  GT, %[[TEMP_33]], %[[TEMP_cst_8:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_38:.*]] = stablehlo.and %[[TEMP_36]], %[[TEMP_37]] : tensor<2xi1>
// CHECK-NEXT: %[[TEMP_39:.*]] = stablehlo.multiply %[[TEMP_28]], %[[TEMP_33]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_40:.*]] = stablehlo.divide %[[TEMP_39]], %[[TEMP_cst_9:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_41:.*]] = stablehlo.add %[[TEMP_28]], %[[TEMP_40]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_42:.*]] = stablehlo.multiply %[[TEMP_28]], %[[TEMP_35]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_43:.*]] = stablehlo.select %[[TEMP_38]], %[[TEMP_41]], %[[TEMP_42]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_44:.*]] = stablehlo.select %[[TEMP_30]], %[[TEMP_31]], %[[TEMP_43]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_45:.*]] = stablehlo.add %[[TEMP_25]], %[[TEMP_44]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_46:.*]] = stablehlo.multiply %[[TEMP_45]], %[[TEMP_cst_10:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_47:.*]] = stablehlo.add %[[TEMP_46]], %[[TEMP_1]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_48:.*]] = stablehlo.multiply %[[TEMP_47]], %[[TEMP_cst_10:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_49:.*]] = stablehlo.multiply %[[TEMP_3]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_50:.*]] = stablehlo.add %[[TEMP_25]], %[[TEMP_7]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_51:.*]] = stablehlo.divide %[[TEMP_49]], %[[TEMP_50]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_52:.*]] = stablehlo.subtract %[[TEMP_44]], %[[TEMP_26]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_53:.*]] = stablehlo.add %[[TEMP_51]], %[[TEMP_52]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_54:.*]] = stablehlo.multiply %[[TEMP_48]], %[[TEMP_53]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_55:.*]] = stablehlo.sqrt %[[TEMP_54]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_56:.*]] = stablehlo.divide %[[TEMP_48]], %[[TEMP_50]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_57:.*]] = stablehlo.add %[[TEMP_44]], %[[TEMP_26]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_58:.*]] = stablehlo.divide %[[TEMP_48]], %[[TEMP_57]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_59:.*]] = stablehlo.add %[[TEMP_56]], %[[TEMP_58]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_60:.*]] = stablehlo.sqrt %[[TEMP_59]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_61:.*]] = stablehlo.multiply %[[TEMP_3]], %[[TEMP_60]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_62:.*]] = stablehlo.select %[[TEMP_6]], %[[TEMP_55]], %[[TEMP_61]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_63:.*]] = stablehlo.select %[[TEMP_5]], %[[TEMP_3]], %[[TEMP_62]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_64:.*]] = stablehlo.compare  LT, %[[TEMP_1]], %[[TEMP_cst_3:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_65:.*]] = stablehlo.select %[[TEMP_64]], %[[TEMP_cst_2:.*]], %[[TEMP_cst_1:.*]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_66:.*]] = stablehlo.compare  GE, %[[TEMP_3]], %[[TEMP_65]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_67:.*]] = stablehlo.select %[[TEMP_66]], %[[TEMP_3]], %[[TEMP_1]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_68:.*]] = stablehlo.select %[[TEMP_66]], %[[TEMP_65]], %[[TEMP_cst_5:.*]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_69:.*]] = stablehlo.compare  GE, %[[TEMP_67]], %[[TEMP_68]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_70:.*]] = stablehlo.log %[[TEMP_67]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_71:.*]] = stablehlo.add %[[TEMP_70]], %[[TEMP_cst_0:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_72:.*]] = stablehlo.compare  EQ, %[[TEMP_3]], %[[TEMP_cst_7:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_73:.*]] = stablehlo.not %[[TEMP_72]] : tensor<2xi1>
// CHECK-NEXT: %[[TEMP_74:.*]] = stablehlo.and %[[TEMP_66]], %[[TEMP_73]] : tensor<2xi1>
// CHECK-NEXT: %[[TEMP_75:.*]] = stablehlo.divide %[[TEMP_1]], %[[TEMP_3]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_76:.*]] = stablehlo.select %[[TEMP_74]], %[[TEMP_75]], %[[TEMP_cst_8:.*]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_77:.*]] = stablehlo.multiply %[[TEMP_76]], %[[TEMP_76]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_78:.*]] = stablehlo.log_plus_one %[[TEMP_77]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_79:.*]] = stablehlo.multiply %[[TEMP_78]], %[[TEMP_cst_10:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_80:.*]] = stablehlo.add %[[TEMP_71]], %[[TEMP_79]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_81:.*]] = stablehlo.compare  LT, %[[TEMP_3]], %cst : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_82:.*]] = stablehlo.compare  LT, %[[TEMP_1]], %[[TEMP_cst_11:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_83:.*]] = stablehlo.and %[[TEMP_81]], %[[TEMP_82]] : tensor<2xi1>
// CHECK-NEXT: %[[TEMP_84:.*]] = stablehlo.multiply %[[TEMP_7]], %[[TEMP_26]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_85:.*]] = stablehlo.add %[[TEMP_46]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_86:.*]] = stablehlo.divide %[[TEMP_84]], %[[TEMP_85]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_87:.*]] = stablehlo.negate %[[TEMP_86]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_88:.*]] = stablehlo.compare  GE, %[[TEMP_1]], %[[TEMP_cst_11:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_89:.*]] = stablehlo.multiply %[[TEMP_49]], %[[TEMP_cst_10:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_90:.*]] = stablehlo.divide %[[TEMP_89]], %[[TEMP_50]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_91:.*]] = stablehlo.multiply %[[TEMP_57]], %[[TEMP_cst_10:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_92:.*]] = stablehlo.add %[[TEMP_90]], %[[TEMP_91]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_93:.*]] = stablehlo.compare  LE, %[[TEMP_46]], %[[TEMP_cst_6:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_94:.*]] = stablehlo.divide %[[TEMP_89]], %[[TEMP_52]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_95:.*]] = stablehlo.add %[[TEMP_90]], %[[TEMP_94]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_96:.*]] = stablehlo.subtract %[[TEMP_46]], %[[TEMP_cst_11:.*]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_97:.*]] = stablehlo.select %[[TEMP_93]], %[[TEMP_95]], %[[TEMP_96]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_98:.*]] = stablehlo.select %[[TEMP_88]], %[[TEMP_92]], %[[TEMP_97]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_99:.*]] = stablehlo.select %[[TEMP_83]], %[[TEMP_87]], %[[TEMP_98]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_100:.*]] = stablehlo.multiply %[[TEMP_99]], %[[TEMP_85]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_101:.*]] = stablehlo.sqrt %[[TEMP_100]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_102:.*]] = stablehlo.divide %[[TEMP_3]], %[[TEMP_101]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_103:.*]] = stablehlo.add %[[TEMP_99]], %[[TEMP_101]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_104:.*]] = stablehlo.log_plus_one %[[TEMP_103]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_105:.*]] = stablehlo.select %[[TEMP_83]], %[[TEMP_102]], %[[TEMP_104]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_106:.*]] = stablehlo.select %[[TEMP_69]], %[[TEMP_80]], %[[TEMP_105]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_107:.*]] = stablehlo.real %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT: %[[TEMP_108:.*]] = stablehlo.atan2 %[[TEMP_63]], %[[TEMP_107]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_109:.*]] = stablehlo.imag %[[TEMP_arg0:.*]] : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK-NEXT: %[[TEMP_110:.*]] = stablehlo.compare  LT, %[[TEMP_109]], %[[TEMP_cst_8:.*]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// CHECK-NEXT: %[[TEMP_111:.*]] = stablehlo.negate %[[TEMP_106]] : tensor<2xf32>
// CHECK-NEXT: %[[TEMP_112:.*]] = stablehlo.select %[[TEMP_110]], %[[TEMP_106]], %[[TEMP_111]] : tensor<2xi1>, tensor<2xf32>
// CHECK-NEXT: %[[TEMP_113:.*]] = stablehlo.complex %[[TEMP_108]], %[[TEMP_112]] : tensor<2xcomplex<f32>>
// CHECK-NEXT: return %[[TEMP_113:.*]] : tensor<2xcomplex<f32>>

  %0 = "tf.Acos"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @acos_dynamic
func.func @acos_dynamic(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.Acos"
  %0 = "tf.Acos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @cast_dynamic_i2f
func.func @cast_dynamic_i2f(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK: stablehlo.convert %arg0 : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @cast_i2f
func.func @cast_i2f(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @cast_c2f
func.func @cast_c2f(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK: stablehlo.convert %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @ceil
func.func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.ceil %arg0 : tensor<2xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @ceil_dynamic
func.func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.ceil %arg0 : tensor<?xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @complex_abs
func.func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  // CHECK: stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  %0 = "tf.ComplexAbs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @cos
func.func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.cosine %arg0 : tensor<2xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tan
func.func @tan(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.tan %arg0 : tensor<2xf32>
  %0 = "tf.Tan"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @cos_dynamic
func.func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.cosine %arg0 : tensor<?xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @exp
func.func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.exponential %arg0 : tensor<2xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @expm1
func.func @expm1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.exponential_minus_one %arg0 : tensor<2xf32>
  %0 = "tf.Expm1"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @exp_dynamic
func.func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.exponential %arg0 : tensor<?xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @floor
func.func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.floor %arg0 : tensor<2xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @floor_dynamic
func.func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.floor %arg0 : tensor<?xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  // CHECK: stablehlo.is_finite %arg0 : (tensor<2xf32>) -> tensor<2xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// -----

// CHECK-LABEL: func @is_finite_dynamic
func.func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK: stablehlo.is_finite %arg0 : (tensor<?xf32>) -> tensor<?xi1>
  %0 = "tf.IsFinite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @log
func.func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.log %arg0 : tensor<2xf32>
  %0 = "tf.Log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @log_dynamic
func.func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.log %arg0 : tensor<?xf32>
  %0 = "tf.Log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @log1p
func.func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.log_plus_one %arg0 : tensor<2xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @log1p_dynamic
func.func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.log_plus_one %arg0 : tensor<?xf32>
  %0 = "tf.Log1p"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @neg
func.func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.negate %arg0 : tensor<2xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @neg_dynamic
func.func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: stablehlo.negate %arg0 : tensor<?xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: stablehlo.logistic
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @sigmoid_complex
func.func @sigmoid_complex(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: stablehlo.logistic
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: @xla_scatter
func.func @xla_scatter(%arg0: tensor<2x10xi1>, %arg1: tensor<1xi32>, %arg2: tensor<2x3xi1>) -> tensor<2x10xi1> {
  // CHECK:       %[[RESULT:.*]] = "stablehlo.scatter"(%arg0, %arg1, %arg2)
  // CHECK-SAME:  <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], scatter_dims_to_operand_dims = [1]>, unique_indices = false}>
  // CHECK-NEXT:  ^bb0(%[[VALUE_A:.*]]: tensor<i1>, %[[VALUE_B:.*]]: tensor<i1>):
  // CHECK-NEXT:    %[[IDENTITY:.*]] = func.call @scatter_update(%[[VALUE_A]], %[[VALUE_B]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK-NEXT:    stablehlo.return %[[IDENTITY]] : tensor<i1>
  // CHECK-NEXT:  }) : (tensor<2x10xi1>, tensor<1xi32>, tensor<2x3xi1>) -> tensor<2x10xi1>
  // CHECK-NEXT:  return %[[RESULT]] : tensor<2x10xi1>
  %0 = "tf.XlaScatter"(%arg0, %arg1, %arg2) <{dimension_numbers = "\0A\02\00\01\1A\01\01", indices_are_sorted = true, update_computation = @scatter_update}> : (tensor<2x10xi1>, tensor<1xi32>, tensor<2x3xi1>) -> tensor<2x10xi1>
  return %0 : tensor<2x10xi1>
}

func.func private @scatter_update(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  return %arg1 : tensor<i1>
}