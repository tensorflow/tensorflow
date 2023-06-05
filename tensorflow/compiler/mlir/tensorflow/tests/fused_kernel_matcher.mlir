// RUN: tf-opt %s -tf-fused-kernel-matcher | FileCheck %s

//===----------------------------------------------------------------------===//
// Conv2D + BiasAdd + <Activation> fusions.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: conv2DBiasAdd_noActivation
func.func @conv2DBiasAdd_noActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {TArgs = [f32], data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd"], num_args = 1 : i64, operand_segment_sizes = array<i32: 1, 1, 1, 0>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_reluActivation
func.func @conv2DBiasAdd_reluActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {TArgs = [f32], data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu"], num_args = 1 : i64, operand_segment_sizes = array<i32: 1, 1, 1, 0>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Relu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_relu6Activation
func.func @conv2DBiasAdd_relu6Activation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {TArgs = [f32], data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Relu6"], num_args = 1 : i64, operand_segment_sizes = array<i32: 1, 1, 1, 0>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_eluActivation
func.func @conv2DBiasAdd_eluActivation(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_0:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {TArgs = [f32], data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd", "Elu"], num_args = 1 : i64, operand_segment_sizes = array<i32: 1, 1, 1, 0>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_convMultipleUses
func.func @conv2DBiasAdd_convMultipleUses(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "tf.Identity"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3, %4 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: conv2DBiasAdd_biasAddMultipleUse
func.func @conv2DBiasAdd_biasAddMultipleUse(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK-DAG: %[[VAL:.*]] = "tf._FusedConv2D"(%arg2, %arg1, %arg0) {TArgs = [f32], data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 0.000000e+00 : f32, explicit_paddings = [], fused_ops = ["BiasAdd"], num_args = 1 : i64, operand_segment_sizes = array<i32: 1, 1, 1, 0>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_0:.*]] = "tf.Elu"(%[[VAL]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_1:.*]] = "tf.Identity"(%[[VAL_0]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-DAG: %[[VAL_2:.*]] = "tf.Identity"(%[[VAL]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_1]], %[[VAL_2]]
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "tf.Identity"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3, %4 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: conv2D_noFusion
func.func @conv2D_noFusion(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2D_noFusion1
func.func @conv2D_noFusion1(%arg0: tensor<*xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  // The result of the conv must be the first input to BiasAdd to be fusable.
  %1 = "tf.BiasAdd"(%arg0, %0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

// CHECK-LABEL: conv2D_dataFormatMismatch
func.func @conv2D_dataFormatMismatch(%arg0: tensor<128xf32>, %arg1: tensor<1x1x3x128xf32>, %arg2: tensor<8x32x32x3xf32>) -> (tensor<*xf32>) {
  // CHECK-NOT: "tf._FusedConv2D"
  %0 = "tf.Conv2D"(%arg2, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<8x32x32x3xf32>, tensor<1x1x3x128xf32>) -> tensor<*xf32>
  // The result of the conv must be the first input to BiasAdd to be fusable.
  %1 = "tf.BiasAdd"(%0, %arg0) {data_format = "NCHW"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %2 = "tf.Elu"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.Identity"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// MatMul + BiasAdd + <Activation> fusions.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: matmulBiasAdd
func.func @matmulBiasAdd(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_3:.*]] = "tf._FusedMatMul"(%arg1, %arg2, %arg0) {epsilon = 0.000000e+00 : f32, fused_ops = ["BiasAdd"], transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_4:.*]] = "tf.Identity"(%[[VAL_3]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_4]]
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  %5 = "tf.Identity"(%4) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %5 : tensor<*xf32>
}

// CHECK-LABEL: matmulBiasAdd_relu
func.func @matmulBiasAdd_relu(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_3:.*]] = "tf._FusedMatMul"(%arg1, %arg2, %arg0) {epsilon = 0.000000e+00 : f32, fused_ops = ["BiasAdd", "Relu"], transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_4:.*]] = "tf.Identity"(%[[VAL_3]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_4]]
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  %5 = "tf.Relu"(%4) : (tensor<*xf32>) -> tensor<*xf32>
  %6 = "tf.Identity"(%5) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %6 : tensor<*xf32>
}

// CHECK-LABEL: matmulBiasAdd_relu6
func.func @matmulBiasAdd_relu6(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_3:.*]] = "tf._FusedMatMul"(%arg1, %arg2, %arg0) {epsilon = 0.000000e+00 : f32, fused_ops = ["BiasAdd", "Relu6"], transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_4:.*]] = "tf.Identity"(%[[VAL_3]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_4]]
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  %5 = "tf.Relu6"(%4) : (tensor<*xf32>) -> tensor<*xf32>
  %6 = "tf.Identity"(%5) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %6 : tensor<*xf32>
}

// CHECK-LABEL: matmulBiasAdd_elu
func.func @matmulBiasAdd_elu(%arg0: tensor<64xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x64xf32>) -> (tensor<*xf32>) {
  // CHECK: %[[VAL_3:.*]] = "tf._FusedMatMul"(%arg1, %arg2, %arg0) {epsilon = 0.000000e+00 : f32, fused_ops = ["BiasAdd", "Elu"], transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  // CHECK: %[[VAL_4:.*]] = "tf.Identity"(%[[VAL_3]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[VAL_4]]
  %3 = "tf.MatMul"(%arg1, %arg2) {transpose_a = false, transpose_b = false} : (tensor<8x32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %arg0) {data_format = "NHWC"} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  %5 = "tf.Elu"(%4) : (tensor<*xf32>) -> tensor<*xf32>
  %6 = "tf.Identity"(%5) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %6 : tensor<*xf32>
}
