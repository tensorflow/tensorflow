// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics -stablehlo-test-tf-to-stablehlo | FileCheck %s

func.func @fused_batchnorm_no_training(%arg0: tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]> : tensor<8xf32>} : () -> tensor<8xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4]> : tensor<8xf32>} : () -> tensor<8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}
// CHECK: func.func @main(%[[ARG_0:.+]]: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-DAG: %[[MUL:.*]] = stablehlo.multiply %arg0, %[[BROADCAST_0]] : tensor<8x8x8x8xf32>
// CHECK-DAG: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[CONST_1]], dims = [3] : (tensor<8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[BROADCAST_1]] : tensor<8x8x8x8xf32>
// CHECK: return %[[ADD]] : tensor<8x8x8x8xf32>

// -----

func.func @fuse_conv_batchnorm(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %1#0 : tensor<1x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[CONST_1]], dims = [3] : (tensor<2xf32>) -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[BROADCAST_0]]) {{.*}} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST_1]] : tensor<1x3x2x2xf32>
// CHECK: return %[[ADD]] : tensor<1x3x2x2xf32>

// -----

func.func @func_conv_batchnorm_relu6(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %2 = "tf.Relu6"(%1#0) : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_2:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CONST_3:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[CONST_1]], dims = [3] : (tensor<2xf32>) -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[BROADCAST_0]]) {{.*}} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST_1]] : tensor<1x3x2x2xf32>
// CHECK-DAG: %[[RELU6:.*]] = stablehlo.clamp %[[CONST_3]], %[[ADD]], %[[CONST_2]] : (tensor<f32>, tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[RELU6]] : tensor<1x3x2x2xf32>

