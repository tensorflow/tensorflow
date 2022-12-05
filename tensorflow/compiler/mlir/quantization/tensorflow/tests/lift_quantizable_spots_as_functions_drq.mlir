// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions-drq | FileCheck %s

// CHECK-LABEL: lift_float_matmul
func.func @lift_float_matmul(%arg0: tensor<1x12x12x512xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
  %out_1 = "tf.MatMul"(%arg0, %cst) {
    device = "", transpose_a = false, transpose_b = false
  } : (tensor<1x12x12x512xf32>, tensor<512x512xf32>) -> tensor<*xf32>
  %out_2 = "tf.MatMul"(%arg0, %arg0) {
    device = "", transpose_a = false, transpose_b = true
  } : (tensor<1x12x12x512xf32>, tensor<1x12x12x512xf32>) -> tensor<*xf32>
  func.return %out_1, %out_2 : tensor<*xf32>, tensor<*xf32>

// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
// CHECK: %[[PARTITIONEDCALL:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST]])
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @composite_matmul_fn_1}
// CHECK: %[[UNQUANTIZED_OUTPUT:.*]] = "tf.MatMul"(%arg0, %arg0)
// CHECK: }

// CHECK-LABEL: private @composite_matmul_fn_1
// CHECK-NEXT: %[[OUT:.*]] = "tf.MatMul"(%arg0, %arg1)
// CHECK-NEXT: return %[[OUT]]
}

// -----

// CHECK-LABEL: lift_float_conv
func.func @lift_float_conv(%arg0: tensor<1x3x4x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst_1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>

  %3 = "tf.Conv2D"(%arg0, %cst_1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>

  func.return %2, %4 : tensor<*xf32>, tensor<*xf32>

// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]])
// CHECK-SAME: {_tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @composite_conv2d_fn_2}
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%[[PARTITIONEDCALL_0]], %[[CONST_0]])
// CHECK: %[[RELU6_0:.*]] = "tf.Relu6"(%[[BIASADD_0]])
// CHECK: %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]])
// CHECK-SAME: f = @composite_conv2d_fn_1}
// CHECK: %[[BIASADD_1:.*]] = "tf.BiasAdd"(%[[PARTITIONEDCALL_1]], %[[CONST_0]])
// CHECK: return %[[RELU6_0]], %[[BIASADD_1]]
// CHECK: }

// CHECK-LABEL: private @composite_conv2d_fn_2
// CHECK-NEXT: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// CHECK-SAME: attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations"
// CHECK-SAME: data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
// CHECK-NEXT: return %[[CONV2D_0]]

// CHECK-LABEL: private @composite_conv2d_fn_1
// CHECK-NEXT: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
// CHECK-SAME: attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations"
// CHECK-NEXT: return %[[CONV2D_0]]
}

// -----

// CHECK-LABEL: not_lift_float_conv_with_non_constant_weights
func.func @not_lift_float_conv_with_non_constant_weights(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>

  %3 = "tf.Conv2D"(%arg0, %arg1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>

  func.return %2, %4 : tensor<*xf32>, tensor<*xf32>

// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-NOT: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1)
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %arg1)
}

// -----

// CHECK-LABEL: lift_float_depthwise_conv
func.func @lift_float_depthwise_conv(%arg0: tensor<1x3x4x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst_1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1]
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %1 = "tf.BiasAdd"(%0, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>

  %3 = "tf.DepthwiseConv2dNative"(%arg0, %cst_1) {
    data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [],
    padding = "SAME", strides = [1, 1, 2, 1]
  } : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<*xf32>
  %4 = "tf.BiasAdd"(%3, %cst) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  func.return %2, %4 : tensor<*xf32>, tensor<*xf32>

// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x1xf32>} : () -> tensor<2x3x3x1xf32>
// CHECK: %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]])
// CHECK-SAME: _tfl_quant_trait = "fully_quantizable",
// CHECK-SAME: f = @composite_depthwise_conv2d_fn_2}
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%[[PARTITIONEDCALL_0]], %[[CONST_0]])
// CHECK: %[[RELU6_0:.*]] = "tf.Relu6"(%[[BIASADD_0]])
// CHECK: %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg0, %[[CONST_1]])
// CHECK-SAME: f = @composite_depthwise_conv2d_fn_1}
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%[[PARTITIONEDCALL_1]], %[[CONST_0]])
// CHECK: return %[[RELU6_0]], %[[BIASADD_0]]
// CHECK: }

// CHECK-LABEL: private @composite_depthwise_conv2d_fn_2
// CHECK-NEXT: %[[DEPTHWISECONV2D_0:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %arg1)
// CHECK-SAME: attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations"
// CHECK-NEXT: return %[[DEPTHWISECONV2D_0:.*]]

// CHECK-LABEL: private @composite_depthwise_conv2d_fn_1
// CHECK-NEXT: %[[DEPTHWISECONV2D_0:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %arg1)
// CHECK-SAME: attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations"
// CHECK-NEXT: return %[[DEPTHWISECONV2D_0:.*]]
}
