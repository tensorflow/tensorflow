// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions='quantization-method=drq target-opset=UNIFORM_QUANTIZED'   -quant-quantize-composite-functions='quantization-method=drq' -symbol-dce | FileCheck %s

module {
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
// CHECK-DAG: %[[q_w:.*]]  = "tf.Const"()
// CHECK-SAME: tensor<512x512x!tf_type.qint8>} : () -> tensor<512x512x!tf_type.qint8>
// CHECK-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<3.93700805E-9> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK: %0 = "tf.PartitionedCall"(%arg0, %[[q_w]], %[[scale]], %[[zp]])
// CHECK-SAME: f = @quantized_matmul_fn

// CHECK-LABEL: func private @quantized_matmul_fn_0
//CHECK:  %0 = "tf.UniformQuantizedDotHybrid"(%arg0, %arg1, %arg2, %arg3)
}

// -----

module {
  func.func @conv(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
  %weight = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %conv = "tf.Conv2D"(%arg0, %weight) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  func.return %conv : tensor<*xf32>
  }

// CHECK-LABEL: func @conv
// CHECK-DAG: %[[w:.*]]  =  "tf.Const"() {value = dense<2.000000e+00> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
// CHECK: %[[conv:.*]] = "tf.Conv2D"(%arg0, %[[w]]) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>

}
