// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics -stablehlo-test-tf-to-stablehlo --mlir-print-ir-after-all | FileCheck %s

func.func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}
// CHECK: func.func @main(%[[ARG_0:.+]]: tensor<8x8x8x8xf32>, %[[ARG_1:.+]]: tensor<8xf32>, %[[ARG_2:.+]]: tensor<8xf32>, %[[ARG_3:.+]]: tensor<8xf32>, %[[ARG_4:.+]]: tensor<8xf32>) -> tensor<8x8x8x8xf32>
// CHECK: %[[BN:.+]] = "stablehlo.batch_norm_inference"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]], %[[ARG_4]]) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
// CHECK: return %[[BN]] : tensor<8x8x8x8xf32>
