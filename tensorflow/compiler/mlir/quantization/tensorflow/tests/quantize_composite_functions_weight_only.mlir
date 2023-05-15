// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions='quantization-method=weight_only target-opset=XLA' -quant-quantize-composite-functions='quantization-method=weight_only target-opset=XLA' -symbol-dce | FileCheck --check-prefix=PerTensor %s
// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions='quantization-method=weight_only target-opset=XLA' -quant-quantize-composite-functions='quantization-method=weight_only target-opset=XLA enable-per-channel-quantization=true' -symbol-dce | FileCheck --check-prefix=PerChannel %s

module {
  // TODO(b/260020937): Support transpose_a, transpose_b for matmul.
  func.func @matmul(%arg0: tensor<2x12xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<12x2xf32>} : () -> tensor<12x2xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x12xf32>, tensor<12x2xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x12xf32>, %arg1: tensor<12x2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x12xf32>, tensor<12x2xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}

// PerTensor-LABEL: func @matmul
// PerTensor-DAG: %[[q_w:.*]] = "tf.Const"() {value = dense<0> : tensor<12x2xi8>} : () -> tensor<12x2xi8>
// PerTensor-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<3.93700805E-9> : tensor<f32>} : () -> tensor<f32>
// PerTensor-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// PerTensor: %[[out:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerTensor-SAME: f = @quantized_matmul_fn_0} : (tensor<2x12xf32>, tensor<12x2xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerTensor: return %[[out]]

// PerChannel-LABEL: func @matmul
// PerChannel-DAG: %[[q_w:.*]] = "tf.Const"() {value = dense<0> : tensor<12x2xi8>} : () -> tensor<12x2xi8>
// PerChannel-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<3.93700805E-9> : tensor<f32>} : () -> tensor<f32>
// PerChannel-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// PerChannel: %[[out:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerChannel-SAME: f = @quantized_matmul_fn_0} : (tensor<2x12xf32>, tensor<12x2xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerChannel: return %[[out]]

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

// PerTensor-LABEL: func @conv
// PerTensor-DAG: %[[q_w:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// PerTensor-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<{{[0-9\.Ee\+\-]+}}> : tensor<f32>} : () -> tensor<f32>
// PerTensor-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<i32>} : () -> tensor<i32>
// PerTensor: %[[out_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerTensor-SAME: f = @quantized_conv2d_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerTensor: %[[out_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerTensor-SAME: f = @quantized_conv2d_fn_0} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerTensor: return %[[out_1]], %[[out_2]]

// PerChannel-LABEL: func @conv
// PerChannel-DAG: %[[q_w:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// PerChannel-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<{{[0-9\.Ee\+\-]+}}> : tensor<2xf32>} : () -> tensor<2xf32>
// PerChannel-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<2xi32>} : () -> tensor<2xi32>
// PerChannel: %[[out_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerChannel-SAME: f = @quantized_conv2d_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xi8>, tensor<2xf32>, tensor<2xi32>) -> tensor<*xf32>
// PerChannel: %[[out_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerChannel-SAME: f = @quantized_conv2d_fn_0} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xi8>, tensor<2xf32>, tensor<2xi32>) -> tensor<*xf32>
// PerChannel: return %[[out_1]], %[[out_2]]

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

// PerTensor-LABEL: func @depthwise_conv
// PerTensor-DAG: %[[q_w1:.*]] = "tf.Const"() {value = dense<127> : tensor<2x3x3x1xi8>}
// PerTensor-DAG: %[[q_w2:.*]] = "tf.Const"() {value = dense<127> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// PerTensor-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<0.0236220472> : tensor<f32>} : () -> tensor<f32>
// PerTensor-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// PerTensor-DAG: %[[bias:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3xf32>}
// PerTensor: %[[out_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w1]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerTensor-SAME: f = @quantized_depthwise_conv2d_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerTensor: %[[out_1_add:.*]]  = "tf.BiasAdd"(%[[out_1]], %[[bias]])
// PerTensor: %[[out_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w2]], %[[scale]], %[[zp]]) {config = "", config_proto = "", executor_type = "",
// PerTensor-SAME: f = @quantized_depthwise_conv2d_fn_0} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xi8>, tensor<f32>, tensor<i32>) -> tensor<*xf32>
// PerTensor: return %[[out_1_add]], %[[out_2]]

// PerChannel-LABEL: func @depthwise_conv
// PerChannel-DAG: %[[bias1:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
// PerChannel-DAG: %[[q_w1:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<2x3x3x1xi8>} : () -> tensor<2x3x3x1xi8>
// PerChannel-DAG: %[[q_w2:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// PerChannel-DAG: %[[scale1:.*]] = "tf.Const"() {value = dense<{{[0-9\.Ee\+\-]+}}> : tensor<3xf32>} : () -> tensor<3xf32>
// PerChannel-DAG: %[[scale2:.*]] = "tf.Const"() {value = dense<{{[0-9\.Ee\+\-]+}}> : tensor<6xf32>} : () -> tensor<6xf32>
// PerChannel-DAG: %[[zp1:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<3xi32>} : () -> tensor<3xi32>
// PerChannel-DAG: %[[zp2:.*]] = "tf.Const"() {value = dense<{{[0-9]+}}> : tensor<6xi32>} : () -> tensor<6xi32>
// PerChannel: %[[out_1:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w1]], %[[scale1]], %[[zp1]]) {config = "", config_proto = "", executor_type = "",
// PerChannel-SAME: f = @quantized_depthwise_conv2d_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xi8>, tensor<3xf32>, tensor<3xi32>) -> tensor<*xf32>
// PerChannel: %[[out_1_add:.*]]  = "tf.BiasAdd"(%[[out_1]], %[[bias1]])
// PerChannel: %[[out_2:.*]] = "tf.PartitionedCall"(%arg0, %[[q_w2]], %[[scale2]], %[[zp2]]) {config = "", config_proto = "", executor_type = "",
// PerChannel-SAME: f = @quantized_depthwise_conv2d_fn_0} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xi8>, tensor<6xf32>, tensor<6xi32>) -> tensor<*xf32>
// PerChannel: return %[[out_1_add]], %[[out_2]]
}
