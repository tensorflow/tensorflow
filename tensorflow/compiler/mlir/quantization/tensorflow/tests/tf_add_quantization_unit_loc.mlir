// RUN: tf-quant-opt %s -mlir-print-debuginfo -mlir-print-local-scope -tf-quant-add-quantization-unit-loc | FileCheck %s

func.func @conv2d_unmatching_loc_pattern(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1]}
       : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16> loc("Model/conv2d")
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %3 : tensor<1x3x2x2xf32>
// CHECK: tf.Conv2D
// CHECK-SAME: loc("Model/conv2d")
}

func.func @conv2d_with_valid_loc(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1]}
       : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16> loc(fused["Conv2D:", "Model/conv2d"])
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %3 : tensor<1x3x2x2xf32>
// CHECK: tf.Conv2D
// CHECK-SAME: loc(callsite("Model/conv2d@conv2d_with_valid_loc"("Conv2D") at "QuantizationUnit({{.*}})"))
}

func.func @conv2d_with_callsite_loc(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1]}
       : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16> loc(fused["Conv2D:", callsite("Model/conv2d" at "model.py":10:8)])
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %3 : tensor<1x3x2x2xf32>
// CHECK: tf.Conv2D
// CHECK-SAME: loc(callsite("Model/conv2d@conv2d_with_callsite_loc"("Conv2D") at "QuantizationUnit({{.*}})"))
}

func.func @conv2d_with_func_name(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x3x3x2xbf16>} : () -> tensor<2x3x3x2xbf16>
  %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xbf16>
  %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1]}
       : (tensor<1x3x4x3xbf16>, tensor<2x3x3x2xbf16>) -> tensor<1x3x2x2xbf16> loc(fused["Conv2D:", "Model/conv2d@original_func"])
  %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x2x2xbf16>) -> tensor<1x3x2x2xf32>
  %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  return %3 : tensor<1x3x2x2xf32>
// CHECK: tf.Conv2D
// CHECK-SAME: loc(callsite("Model/conv2d@original_func"("Conv2D") at "QuantizationUnit({{.*}})"))
}

