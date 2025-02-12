// RUN: tf-quant-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope -quant-add-quantization-unit-loc -inline -quant-prepare-lifting -quant-lift-quantizable-spots-as-functions='target-opset=XLA' | FileCheck %s

// This file test the selective quantiation feature in TF Quantizer. In the test
// config, the op named "test_opt_out" will not be quantized.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}} {
  func.func @conv2d_unmatching_unit(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<1.0> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
        : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32> loc(fused["Conv2D:", "Model/conv2d"])
    %2 = "tf.IdentityN"(%1) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %2 : tensor<1x3x2x2xf32>
  }
}

// CHECK-LABEL: func @conv2d_unmatching_unit
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_conv2d_fn_1
// Check that the `_tfl_quant_trait` attribute exists since the unit is not in `unit_wise_quantization_specs`.
// CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: loc(callsite("Model/conv2d@conv2d_unmatching_unit"("Conv2D") at "QuantizationUnit({{.*}})"))

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}} {
  func.func @conv2d_disable_quantization(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<1.0> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
        : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32> loc(fused["Conv2D:", "test_opt_out"])
    %2 = "tf.IdentityN"(%1) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %2 : tensor<1x3x2x2xf32>
  }
}

// CHECK-LABEL: func @conv2d_disable_quantization
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_conv2d_fn_1
// Check that quantization is disabled for this unit.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: loc(callsite("test_opt_out@conv2d_disable_quantization"("Conv2D") at "QuantizationUnit({{.*}})"))

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}} {
  func.func @conv2d_with_bias_disable_quantization(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<1.0> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
        : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32> loc(fused["Conv2D:", "test_opt_out"])
    %2 = "tf.BiasAdd"(%1, %cst_0) : (tensor<1x3x2x2xf32>, tensor<2xf32>) -> tensor<1x3x2x2xf32> loc(fused["BiasAdd:", "model/bias_add"])
    %3 = "tf.IdentityN"(%2) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %3 : tensor<1x3x2x2xf32>
  }
}

// CHECK-LABEL: func @conv2d_with_bias_disable_quantization
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_conv2d_with_bias_fn_1
// Check that quantization is disabled for this unit.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: loc(callsite("test_opt_out@conv2d_with_bias_disable_quantization"("Conv2D") at "QuantizationUnit({{.*}})"))

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}} {
  func.func @matmul_with_reshape_disable_quantization(%arg0: tensor<1x10xf32>, %arg1: tensor<10x10xf32>) -> (tensor<?x10xf32>) {
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32>
    %cst_0 = "tf.Const"() {value = dense<[-1, 10]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tf.MatMul"(%arg0, %arg1) {
      transpose_a = false, transpose_b = false
    } : (tensor<1x10xf32>, tensor<10x10xf32>) -> tensor<1x10xf32>  loc(fused["MatMul:", "test_opt_out"])
    %2 = "tf.Reshape"(%1, %cst_0) : (tensor<1x10xf32>, tensor<2xi32>) -> tensor<?x10xf32> loc(fused["Reshape:", "model/reshape"])
    %3 = "tf.AddV2"(%2, %cst) : (tensor<?x10xf32>, tensor<10xf32>) -> tensor<?x10xf32> loc(fused["AddV2:", "model/add"])
    func.return %3 : tensor<?x10xf32>
  }
}

// CHECK-LABEL: func @matmul_with_reshape_disable_quantization
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_matmul_with_reshape_and_bias_fn_1
// Check that quantization is disabled for this unit.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: loc(callsite("test_opt_out@matmul_with_reshape_disable_quantization"("MatMul") at "QuantizationUnit({{.*}})"))

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1269 : i32}} {
  func.func private @conv2d_with_inliner(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<1.0> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.Cast"(%arg0) {Truncate = false, device = ""} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}
        : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32> loc(fused["Conv2D:", "test_opt_out"])
    %2 = "tf.IdentityN"(%1) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %2 : tensor<1x3x2x2xf32>
  }

  func.func @serving_default(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @conv2d_with_inliner}
        : (tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32>
    return %0 : tensor<1x3x2x2xf32>
  }

// CHECK-LABEL: func @serving_default
// CHECK: "tf.PartitionedCall"
// CHECK-SAME: f = @composite_conv2d_fn_1
// Check that quantization is disabled for this unit.
// CHECK-NOT: _tfl_quant_trait = "fully_quantizable"
// CHECK-SAME: loc(callsite("test_opt_out@conv2d_with_inliner"("Conv2D") at "QuantizationUnit({{.*}})"))
}
