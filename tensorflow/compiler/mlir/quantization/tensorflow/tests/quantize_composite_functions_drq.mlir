// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions='quantization-method=drq target-opset=UNIFORM_QUANTIZED' -quant-quantize-composite-functions='quantization-method=drq target-opset=UNIFORM_QUANTIZED' -symbol-dce | FileCheck %s

module {
  // TODO(b/260020937): Support transpose_a, transpose_b for matmul.
  func.func @matmul(%arg0: tensor<2x512xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x512xf32>, tensor<512x512xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x512xf32>, tensor<512x512xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @matmul
// CHECK-DAG: %[[q_w:.*]]  = "tf.Const"() {value = #tf_type<tensor_proto : "0x746
// CHECK-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<3.93700805E-9> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK: %0 = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// CHECK-SAME: f = @quantized_matmul_fn_0} : (tensor<2x512xf32>, tensor<512x512x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>

// CHECK-LABEL: func private @quantized_matmul_fn_0
// CHECK:  %0 = "tf.UniformQuantizedDotHybrid"(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME: rhs_quantization_axis = -1 : i64
// CHECK-SAME: rhs_quantization_max_val = 127 : i64
// CHECK-SAME: rhs_quantization_min_val = -127 : i64
}

// -----

module {
  func.func @conv(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %weight = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %1 = "tf.PartitionedCall"(%arg0, %weight) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    %2 = "tf.PartitionedCall"(%arg0, %weight) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_fn_2} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    func.return %1, %2 : tensor<*xf32>, tensor<*xf32>
  }

  func.func private @composite_conv2d_fn_1(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %conv = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    return %conv : tensor<*xf32>
  }

  func.func private @composite_conv2d_fn_2(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %conv = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    return %conv : tensor<*xf32>
  }

// CHECK-LABEL: func @conv
// CHECK-DAG: %[[q_w:.*]] = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674
// CHECK-DAG: %[[w_scale:.*]] = "tf.Const"() {value = dense<0.0157480314> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[w_zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK: %[[quantize_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[w_scale]], %[[w_zp]]) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK: %[[quantize_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[w_scale]], %[[w_zp]]) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_fn_0} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %[[quantize_1]], %[[quantize_2]]

// CHECK-LABEL: func private @quantized_conv2d_fn_0
// CHECK:      %[[CONV2D_0:.*]] = "tf.UniformQuantizedConvolutionHybrid"
// CHECK-SAME: batch_group_count = 1 : i64
// CHECK-SAME: dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02"
// CHECK-SAME: explicit_padding = []
// CHECK-SAME: feature_group_count = 1 : i64
// CHECK-SAME: lhs_dilation = [1, 1]
// CHECK-SAME: padding = "VALID"
// CHECK-SAME: rhs_dilation = [2, 2]
// CHECK-SAME: rhs_quantization_axis = -1 : i64
// CHECK-SAME: rhs_quantization_max_val = 127 : i64
// CHECK-SAME: rhs_quantization_min_val = -127 : i64
// CHECK-SAME: window_strides = [1, 2]
// CHECK-SAME: (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>

// CHECK-LABEL: func private @quantized_conv2d_fn_1
// CHECK:      %[[CONV2D_0:.*]] = "tf.UniformQuantizedConvolutionHybrid"
// CHECK-SAME: padding = "SAME"
}

// -----

module {
  func.func @depthwise_conv(%arg0: tensor<1x3x4x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
    %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
    %cst_2 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_1) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_depthwise_conv2d_fn} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
    %2 = "tf.PartitionedCall"(%arg0, %cst_2) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_depthwise_conv2d_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    func.return %1, %2: tensor<*xf32>, tensor<*xf32>
  }
  func.func private @composite_depthwise_conv2d_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x1xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {
      attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
    } : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
  func.func private @composite_depthwise_conv2d_fn_1(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {
      attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", data_format = "NHWC", device = "", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1]
    } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @depthwise_conv
// CHECK-DAG: %[[bias:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// CHECK-DAG: %[[q_w1:.*]] = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674
// CHECK-SAME:                                                                     -> tensor<2x3x1x3x!tf_type.qint8>
// CHECK-DAG: %[[q_w2:.*]] = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674
// CHECK-SAME:                                                                     -> tensor<2x3x1x6x!tf_type.qint8>
// CHECK-DAG: %[[w_scale:.*]] = "tf.Const"() {value = dense<0.0236220472> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[w_zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>

// CHECK: %[[quantize_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w1]], %[[w_scale]], %[[w_zp]]) {config = "", config_proto = "", executor_type = "", f = @quantized_depthwise_conv2d_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x1x3x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK: %[[quantize_1_add:.*]] = "tf.BiasAdd"(%[[quantize_1]], %[[bias]]) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK: %[[quantize_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w2]], %[[w_scale]], %[[w_zp]]) {config = "", config_proto = "", executor_type = "", f = @quantized_depthwise_conv2d_fn_0} : (tensor<1x3x4x3xf32>, tensor<2x3x1x6x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %[[quantize_1_add]], %[[quantize_2]]

// CHECK-LABEL: func private @quantized_depthwise_conv2d_fn_0
// CHECK:      %[[CONV2D_0:.*]] = "tf.UniformQuantizedConvolutionHybrid"
// CHECK-SAME: batch_group_count = 1 : i64,
// CHECK-SAME: dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02"
// CHECK-SAME: explicit_padding = [],
// CHECK-SAME: feature_group_count = 3 : i64,
// CHECK-SAME: lhs_dilation = [1, 1],
// CHECK-SAME: padding = "VALID",
// CHECK-SAME: rhs_dilation = [2, 2],
// CHECK-SAME: rhs_quantization_axis = -1 : i64,
// CHECK-SAME: rhs_quantization_max_val = 127 : i64,
// CHECK-SAME: rhs_quantization_min_val = -127 : i64,
// CHECK-SAME: window_strides = [1, 2]
// CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x1x6x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>

// CHECK-LABEL: func private @quantized_depthwise_conv2d_fn_1
// CHECK:      %[[CONV2D_0:.*]] = "tf.UniformQuantizedConvolutionHybrid"
// CHECK-SAME: padding = "SAME"
}
