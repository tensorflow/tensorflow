// RUN: tf-quant-opt %s -split-input-file -quant-preprocess-op -tf-quant-prepare-quantize-drq | FileCheck %s

module {
  func.func @matmul(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_matmul_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x1024xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @matmul
// CHECK-DAG: %[[CONST:.*]] = arith.constant dense<0.000000e+00> : tensor<2x1024xf32>
// CHECK: %0 = "quantization.qcast"(%[[CONST]]) : (tensor<2x1024xf32>) -> tensor<2x1024x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>
// CHECK: %1 = "quantization.dcast"(%0) : (tensor<2x1024x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>) -> tensor<2x1024xf32>
// CHECK: %2 = "tf.PartitionedCall"(%arg0, %1) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %2 : tensor<*xf32>

// CHECK-LABEL: func private @composite_matmul_fn
// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %0 : tensor<*xf32>
}

// -----

module {
  func.func @conv2d(%arg0: tensor<1x3x4x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x512xf32>} : () -> tensor<2x3x3x512xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_1) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_fn_1} : (tensor<1x3x4x3xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    %2 = "tf.BiasAdd"(%1, %cst_0) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    func.return %2: tensor<*xf32>
  }
  func.func private @composite_conv2d_fn_1(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x512xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @conv2d
// CHECK-DAG: %[[CONST_0:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = arith.constant dense<3.000000e+00> : tensor<2x3x3x512xf32>
// CHECK: %0 = "quantization.qcast"(%[[CONST_1]]) : (tensor<2x3x3x512xf32>) -> tensor<2x3x3x512x!quant.uniform<i8<-127:127>:f32, 0.023622047244094488>>
// CHECK: %1 = "quantization.dcast"(%0) : (tensor<2x3x3x512x!quant.uniform<i8<-127:127>:f32, 0.023622047244094488>>) -> tensor<2x3x3x512xf32>
// CHECK: %2 = "tf.PartitionedCall"(%arg0, %1) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x3x4x3xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
// CHECK: %3 = "tf.BiasAdd"(%2, %[[CONST_0]])
// CHECK: return %3 : tensor<*xf32>

// CHECK-LABEL: func private @composite_conv2d_fn_1
// CHECK: %0 = "tf.Conv2D"(%arg0, %arg1)
// CHECK: return %0 : tensor<*xf32>
}

// -----

module {
  func.func @depthwise_conv(%arg0: tensor<1x3x4x512xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x512xf32>} : () -> tensor<2x3x3x512xf32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_1) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_depthwise_conv2d_fn} : (tensor<1x3x4x512xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_depthwise_conv2d_fn(%arg0: tensor<1x3x4x512xf32>, %arg1: tensor<2x3x3x512xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {
      attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
    } : (tensor<1x3x4x512xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @depthwise_conv
// CHECK-DAG: %[[CONST_0:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = arith.constant dense<3.000000e+00> : tensor<2x3x1x1536xf32>
// CHECK: %0 = "quantization.qcast"(%[[CONST_1]]) : (tensor<2x3x1x1536xf32>) -> tensor<2x3x1x1536x!quant.uniform<i8<-127:127>:f32, 0.023622047244094488>>
// CHECK: %1 = "quantization.dcast"(%0) : (tensor<2x3x1x1536x!quant.uniform<i8<-127:127>:f32, 0.023622047244094488>>) -> tensor<2x3x1x1536xf32>
// CHECK: %2 = "tf.PartitionedCall"(%arg0, %1) <{config = "", config_proto = "", executor_type = "", f = @composite_depthwise_conv2d_fn_0}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x3x4x512xf32>, tensor<2x3x1x1536xf32>) -> tensor<*xf32>
// CHECK: %3 = "tf.BiasAdd"(%2, %[[CONST_0]])
// CHECK: return %3 : tensor<*xf32>

// CHECK-LABEL: func private @composite_depthwise_conv2d_fn(
// CHECK-SAME:                                             %arg0: tensor<1x3x4x512xf32>,
// CHECK-SAME:                                             %arg1: tensor<2x3x3x512xf32>)

// CHECK-LABEL: func private @composite_depthwise_conv2d_fn_0(
// CHECK-SAME:                                             %arg0: tensor<1x3x4x512xf32>,
// CHECK-SAME:                                             %arg1: tensor<2x3x1x1536xf32>)
// CHECK: %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]}> {attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", device = ""}
// CHECK: return %0 : tensor<*xf32>
}
