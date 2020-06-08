// RUN: tf-opt %s -op-fusion | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// Conv2D + BiasAdd + <Activation> fusions.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: conv2DBiasAdd_noActivation
func @conv2DBiasAdd_noActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd"], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_reluActivation
func @conv2DBiasAdd_reluActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu"], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Relu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_relu6Activation
func @conv2DBiasAdd_relu6Activation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu6"], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_eluActivation
func @conv2DBiasAdd_eluActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Elu"], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_convMultipleUses
func @conv2DBiasAdd_convMultipleUses(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "tf.Identity"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %3, %4 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_biasAddMultipleUse
func @conv2DBiasAdd_biasAddMultipleUse(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK-DAG: %[[VAL:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd"], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_0:.*]] = "tf.Elu"(%[[VAL]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_2:.*]] = "tf.Identity"(%[[VAL]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]], %[[VAL_2]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "tf.Identity"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  return %3, %4 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: conv2D_noFusion
func @conv2D_noFusion(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2D_noFusion1
func @conv2D_noFusion1(%arg0: tensor<*xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  // The result of the conv must be the first input to BiasAdd to be fusable.
  %1 = "tf.BiasAdd"(%arg0, %0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2D_dataFormatMismatch
func @conv2D_dataFormatMismatch(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  // The result of the conv must be the first input to BiasAdd to be fusable.
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NCHW"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  return %3 : tensor<*xf32>
}
