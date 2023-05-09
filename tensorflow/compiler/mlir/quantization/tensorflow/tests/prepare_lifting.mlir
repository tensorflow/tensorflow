// RUN: tf-quant-opt %s -quant-prepare-lifting -split-input-file | FileCheck %s
// RUN: tf-quant-opt %s -quant-prepare-lifting='target-opset=XLA' | FileCheck --check-prefix=XLA-CHECK %s

func.func @decompose_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %add, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %add : tensor<*xf32>
}
// CHECK: func @decompose_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.49743462E-5> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.999950051> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[mul:.*]] = "tf.Mul"(%arg0, %[[CONST_0]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK: %[[add:.*]] = "tf.AddV2"(%[[mul]], %[[CONST]]) : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[add]] : tensor<*xf32>

// -----

func.func @not_decompose_batch_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %bn, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %cst, %cst_0, %cst_0, %cst) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %bn : tensor<*xf32>
}
// CHECK: func @not_decompose_batch_norm
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[bn:.*]], %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%arg0, %[[CONST]], %[[CONST_0]], %[[CONST_0]], %[[CONST]]) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
// CHECK-NEXT: return %[[bn]] : tensor<*xf32>

// -----

func.func @convert_add_to_biasadd(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.AddV2"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @convert_add_to_biasadd
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

// -----

func.func @not_convert_add_to_biasadd(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x3xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x3xf32>} : () -> tensor<2x3x3x3xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<1x3x2x3xf32>} : () -> tensor<1x3x2x3xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x3xf32>) -> tensor<1x3x2x3xf32>
  %1 = "tf.AddV2"(%0, %cst_0) : (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
  func.return %1 : tensor<1x3x2x3xf32>
}
// CHECK: func @not_convert_add_to_biasadd
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x3xf32>} : () -> tensor<2x3x3x3xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<1x3x2x3xf32>} : () -> tensor<1x3x2x3xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x3xf32>) -> tensor<1x3x2x3xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.AddV2"(%[[CONV2D]], %[[CONST_0]]) : (tensor<1x3x2x3xf32>, tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x3xf32>

// -----

func.func @fuse_conv2d_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[CONV2D]] : tensor<1x3x2x2xf32>

// -----

func.func @not_fuse_conv2d_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2x2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1 : tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.Mul"(%[[CONV2D]], %[[CONST_0]]) : (tensor<1x3x2x2xf32>, tensor<2x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x2xf32>

// -----

func.func @fuse_conv2d_with_bias_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

// -----

func.func @not_fuse_conv2d_with_bias_and_mul(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.800000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.Mul"(%0, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %1, %2 : tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[MUL:.*]] = "tf.Mul"(%[[CONV2D]], %[[CONST_1]]) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]], %[[MUL]] : tensor<1x3x2x2xf32>, tensor<1x3x2x2xf32>

// -----

func.func @fuse_conv2d_with_bias_and_add(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.AddV2"(%1, %cst_1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @fuse_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<1x3x2x2xf32>

// -----

func.func @not_fuse_conv2d_with_bias_and_add(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %2 = "tf.AddV2"(%1, %arg1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func @not_fuse_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NEXT: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[CONST]]) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: %[[ADD:.*]] = "tf.AddV2"(%[[BIASADD]], %arg1) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-NEXT: return %[[ADD]] : tensor<1x3x2x2xf32>

// -----

func.func @match_depthwise_conv2d_and_add(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
  %1 = "tf.AddV2"(%0, %cst_0) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}
// CHECK: func @match_depthwise_conv2d_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<*xf32>

// -----

func.func @match_depthwise_conv2d_and_mul(%arg0: tensor<*xf32>) -> (tensor<?x?x?x3xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
  %1 = "tf.Mul"(%0, %cst_0) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
  func.return %1 : tensor<?x?x?x3xf32>
}
// CHECK: func @match_depthwise_conv2d_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: return %[[DEPTHWISE_CONV2D]] : tensor<?x?x?x3xf32>

// -----

func.func @match_depthwise_conv2d_with_bias_and_add(%arg0: tensor<*xf32>) -> (tensor<?x?x?x3xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
  %2 = "tf.AddV2"(%1, %cst_1) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
  func.return %2 : tensor<?x?x?x3xf32>
}
// CHECK: func @match_depthwise_conv2d_with_bias_and_add
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<8.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<?x?x?x3xf32>

// -----

func.func @match_depthwise_conv2d_with_bias_and_mul(%arg0: tensor<*xf32>) -> (tensor<?x?x?x3xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
  func.return %2 : tensor<?x?x?x3xf32>
}
// CHECK: func @match_depthwise_conv2d_with_bias_and_mul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e-01> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-NEXT: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]]) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[DEPTHWISE_CONV2D]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<?x?x?x3xf32>, tensor<3xf32>) -> tensor<?x?x?x3xf32>
// CHECK-NEXT: return %[[BIASADD]] : tensor<?x?x?x3xf32>

// -----

func.func @lower_einsum(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "tf.Einsum"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", equation = "ijk,ikm->ijm"}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}
// CHECK-LABEL: lower_einsum
// CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>

// -----

func.func @removing_identity_after_const(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %identity = "tf.Identity"(%cst) : (tensor<2x3x3x1xf32>) -> tensor<2x3x3x1xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %identity) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}
// CHECK: func @removing_identity_after_const
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[CONST]])

// -----

func.func @not_removing_identity_of_returning_value(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %cst_0 = "tf.Const"() {value = dense<0.400000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<0.500000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  %2 = "tf.Mul"(%1, %cst_1) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}
// CHECK: func @not_removing_identity_of_returning_value
// CHECK: %[[identity:.*]] = "tf.Identity"
// CHECK: return %[[identity]] : tensor<*xf32>

// -----

func.func @batch_norm_with_q_dq(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {device = "", value = dense<5.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "quantfork.qcast"(%cst_1) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.003937007874015748,0.003937007874015748}>>
  %1 = "quantfork.dcast"(%0) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.003937007874015748,0.003937007874015748}>>) -> tensor<2x3x3x2xf32>
  %2 = "quantfork.qcast"(%arg0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>
  %3 = "quantfork.dcast"(%2) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>) -> tensor<1x3x4x3xf32>
  %4 = "tf.Conv2D"(%3, %1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %y, %batch_mean, %batch_variance, %reserve_space_1, %reserve_space_2, %reserve_space_3 = "tf.FusedBatchNormV3"(%4, %cst, %cst_0, %cst, %cst_0) {data_format = "NHWC", device = "", epsilon = 9.99999974E-5 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<*xf32>)
  %5 = "tf.Relu6"(%y) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %6 = "quantfork.qcast"(%5) : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2x!quant.uniform<i8<-127:127>:f32:3, {0.0026771653824903836:-60,0.0032283464285332388:-28}>>
  %7 = "quantfork.dcast"(%6) : (tensor<1x3x2x2x!quant.uniform<i8<-127:127>:f32:3, {0.0026771653824903836:-60,0.0032283464285332388:-28}>>) -> tensor<1x3x2x2xf32>
  %8 = "tf.Identity"(%7) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %9 = "tf.Identity"(%8) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %9 : tensor<1x3x2x2xf32>
}

// CHECK: func @batch_norm_with_q_dq
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<0.707036077> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() {value = dense<-0.914072155> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[q_input:.*]] = "quantfork.qcast"(%arg0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>
// CHECK: %[[dq_input:.*]] = "quantfork.dcast"(%[[q_input]]) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>) -> tensor<1x3x4x3xf32>
// CHECK: %[[q_weight:.*]] = "quantfork.qcast"(%[[cst]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.005567213212411235,0.005567213212411235}>>
// CHECK: %[[dq_weight:.*]] = "quantfork.dcast"(%[[q_weight]]) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.005567213212411235,0.005567213212411235}>>) -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%[[dq_input]], %[[dq_weight]])
// CHECK: %[[bias:.*]] = "tf.BiasAdd"(%[[conv]], %[[cst_0]]) {data_format = "NHWC"}
// CHECK: %[[relu6:.*]] = "tf.Relu6"(%[[bias]])

// -----

func.func @xla_dot_v2(%arg0: tensor<?x2x3xf32>, %arg1: tensor<3x4x5xf32>) -> (tensor<?x2x4x5xf32>) {
  %0 = "tf.XlaDotV2"(%arg0, %arg1) {device = "", dimension_numbers = "\0A\01\02\12\01\00", precision_config = ""} : (tensor<?x2x3xf32>, tensor<3x4x5xf32>) -> tensor<?x2x4x5xf32>
  func.return %0 : tensor<?x2x4x5xf32>
}

// CHECK: func @xla_dot_v2
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<[3, 20]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() {value = dense<[-1, 2, 4, 5]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%arg1, %[[cst]]) : (tensor<3x4x5xf32>, tensor<2xi64>) -> tensor<3x20xf32>
// CHECK: %[[batch_matmul:.*]] = "tf.BatchMatMulV2"(%arg0, %[[reshape]]) {adj_x = false, adj_y = false} : (tensor<?x2x3xf32>, tensor<3x20xf32>) -> tensor<?x2x20xf32>
// CHECK: %[[reshape_0:.*]] = "tf.Reshape"(%[[batch_matmul]], %[[cst_0]]) : (tensor<?x2x20xf32>, tensor<4xi64>) -> tensor<?x2x4x5xf32>
// CHECK: return %[[reshape_0]] : tensor<?x2x4x5xf32>

// XLA-CHECK: func @xla_dot_v2
// XLA-CHECK: %[[einsum:.*]] = "tf.Einsum"(%arg0, %arg1) {equation = "abc,cde->abde"} : (tensor<?x2x3xf32>, tensor<3x4x5xf32>) -> tensor<?x2x4x5xf32>
// XLA-CHECK: return %[[einsum]] : tensor<?x2x4x5xf32>

// -----

// dimension_numbers: {
//   offset_dims: 0
//   collapsed_slice_dims: 1
//   start_index_map: 1
// }
func.func @xla_gather(%arg0: tensor<?x2xf32>, %arg1: tensor<1xi32>, %arg2: tensor<2xi32>) -> tensor<*xf32> {
  %0 = "tf.XlaGather"(%arg0, %arg1, %arg2) {device = "", dimension_numbers = "\0A\01\00\12\01\01\1A\01\01", indices_are_sorted = true} : (tensor<?x2xf32>, tensor<1xi32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK: func @xla_gather
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<0> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() {value = dense<1> : tensor<1x1xi64>} : () -> tensor<1x1xi64>
// CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK: %[[arg1_i64:.*]] = "tf.Cast"(%arg1) {Truncate = false} : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[tensor_scatter_update:.*]] = "tf.TensorScatterUpdate"(%[[cst]], %[[cst_0]], %[[arg1_i64]]) : (tensor<2xi64>, tensor<1x1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK: %[[arg2_i64:.*]] = "tf.Cast"(%arg2) {Truncate = false} : (tensor<2xi32>) -> tensor<2xi64>
// CHECK: %[[slice:.*]] = "tf.Slice"(%arg0, %[[tensor_scatter_update]], %[[arg2_i64]]) : (tensor<?x2xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[slice]], %[[cst_1]]) : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK: return %[[reshape]] : tensor<*xf32>

// -----

// Tests that the converted `tf.Slice` has the correct number of dimensions
// when the output shape is known (`tensor<i32>` instead of `tensor<*xi32>`).

func.func @xla_gather_known_output_shape(%arg0: tensor<5xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<i32> {
  // dimension_numbers: {
  //   collapsed_slice_dims: 0
  //   start_index_map: 0
  // }
  %0 = "tf.XlaGather"(%arg0, %arg1, %arg2) {device = "", dimension_numbers = "\12\01\00\1A\01\00", indices_are_sorted = true} : (tensor<5xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK: func @xla_gather_known_output_shape
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() {value = dense<0> : tensor<1x1xi64>} : () -> tensor<1x1xi64>
// CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
// CHECK: %[[arg1_i64:.*]] = "tf.Cast"(%arg1) {Truncate = false} : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[tensor_scatter_update:.*]] = "tf.TensorScatterUpdate"(%[[cst]], %[[cst_0]], %[[arg1_i64]]) : (tensor<1xi64>, tensor<1x1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK: %[[arg2_i64:.*]] = "tf.Cast"(%arg2) {Truncate = false} : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[slice:.*]] = "tf.Slice"(%arg0, %[[tensor_scatter_update]], %[[arg2_i64]]) : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi32>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[slice]], %[[cst_1]]) : (tensor<1xi32>, tensor<0xi64>) -> tensor<i32>
// CHECK: return %[[reshape]] : tensor<i32>

// -----

func.func @replace_checknumerics_to_identity(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.CheckNumerics"(%arg0) {device = "", message = "transformer"} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK: func @replace_checknumerics_to_identity
// CHECK: %[[out:.*]] = "tf.Identity"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>