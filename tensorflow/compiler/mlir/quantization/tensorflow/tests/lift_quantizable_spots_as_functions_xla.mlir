// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions='target-opset=XLA' | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}, tf_saved_model.semantics} {
  func.func @depthwise_conv(%arg0: tensor<1x3x4x3xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x2x2x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[7.72826624, 8.8264122, 3.64885974]> : tensor<3xf32>} : () -> tensor<3xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.16961098], [2.44217539], [-1.24544525]], [[5.70717144], [5.59951639], [-4.54814768]], [[-4.47071505], [6.03744364], [9.16278743]]], [[[7.51865291], [-2.84365463], [0.0199025106]], [[3.66925859], [4.25404072], [-2.59498501]], [[1.22392368], [0.0616633072], [-9.7246313]]]]> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<1x2x2x3xf32>
    %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<1x2x2x3xf32>, tensor<3xf32>) -> tensor<1x2x2x3xf32>
    %2 = "tf.Relu6"(%1) {device = ""} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    return %3 : tensor<1x2x2x3xf32>
  }
}

// CHECK-LABEL: func @depthwise_conv
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_depthwise_conv2d_with_bias_and_relu6_fn_1
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"

// CHECK-LABEL: private @composite_depthwise_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[DEPTHWISECONV2D_0:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %arg1)
// CHECK-SAME: <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]}>
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map

// -----

func.func @conv_with_non_constant_filter(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-LABEL: func @conv_with_non_constant_filter
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1, %[[CONST_0]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_1
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map

// -----

func.func @conv_with_dynamic_channel_dim(%arg0: tensor<1x3x4x?xf32>) -> tensor<*xf32> {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.16961098], [2.44217539], [-1.24544525]], [[5.70717144], [5.59951639], [-4.54814768]], [[-4.47071505], [6.03744364], [9.16278743]]], [[[7.51865291], [-2.84365463], [0.0199025106]], [[3.66925859], [4.25404072], [-2.59498501]], [[1.22392368], [0.0616633072], [-9.7246313]]]]> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %0 = "tf.Conv2D"(%arg0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x?xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// CHECK-LABEL: func @conv_with_dynamic_channel_dim
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {{.*}} : () -> tensor<2x3x3x1xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]], %[[CONST_0]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_1
// Check that the `_tfl_quant_trait` attribute has been removed.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
// Check that the `attr_map` attribute has been removed.
// CHECK-NOT: attr_map

// -----

func.func @const_filter_with_q_dq(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense<[[[[-0.308480561, 0.122108772], [-0.0622722618, 0.285358578], [0.279975802, -0.227407396]], [[-0.223535746, 0.301872164], [0.45813936, 0.375932634], [-0.142182723, 9.95125505E-4]], [[0.183462933, 0.212702021], [-0.129749238, 0.0611961856], [0.00308316527, -0.486231536]]], [[[0.272826612, 0.382641196], [-0.135114014, 0.115396179], [-0.424618751, -1.311760e-01]], [[0.433140099, 0.15137814], [-0.102797419, 0.288730145], [-0.183163881, 0.0680986494]], [[0.369127393, -0.0638265759], [0.302147657, -0.35623318], [0.204260975, 0.204581305]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "quantization.qcast"(%arg0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>
  %1 = "quantization.dcast"(%0) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>) -> tensor<1x3x4x3xf32>
  %q_w = "quantization.qcast"(%cst) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, 0.0125:-24>>
  %dq_w = "quantization.dcast"(%q_w) : (tensor<2x3x3x2x!quant.uniform<i8:f32, 0.0125:-24>>) -> tensor<2x3x3x2xf32>
  %2 = "tf.Conv2D"(%1, %dq_w) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %3 = "tf.BiasAdd"(%2, %cst_0) {data_format = "NHWC", device = ""} : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32>
  %4 = "tf.Relu"(%3) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %5 = "quantization.qcast"(%4) : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 0.0027450981093387976:-19>>
  %6 = "quantization.dcast"(%5) : (tensor<1x3x2x2x!quant.uniform<i8:f32, 0.0027450981093387976:-19>>) -> tensor<1x3x2x2xf32>
  %7 = "tf.Identity"(%6) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  %8 = "tf.Identity"(%7) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %8 : tensor<1x3x2x2xf32>
}

// CHECK-LABEL: func @const_filter_with_q_dq
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {{.*}} : () -> tensor<2x3x3x2xf32>
// CHECK-DAG: %[[BIAS:.*]] = "tf.Const"() <{value = dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>}> {device = ""}
// CHECK: %[[Q_W:.*]] = "quantization.qcast"(%[[WEIGHT]])
// CHECK: %[[DQ_W:.*]] = "quantization.dcast"(%[[Q_W]])
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"({{.*}}, %[[DQ_W]], %[[BIAS]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu_fn_1
// CHECK-SAME: _tfl_quant_trait = "fully_quantizable"

// CHECK-LABEL: func private @composite_conv2d_with_bias_and_relu_fn_1
