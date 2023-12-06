// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops=insert-at-xla-call-module-op-only=true  -split-input-file | FileCheck %s

// CHECK-NOT: tf.CustomAggregator
module {
  func.func @add_custom_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @no_custom_ops_on_non_f32_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %add : tensor<*xi32>
  }

  func.func @composite_conv2d_with_bias_and_relu6_fn(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<?x100352xf32>, %arg1: tensor<100352x10xf32>) -> tensor<?x10xf32> {
    // CHECK-DAG: %[[ARG0_ID:.*]] = "tf.Identity"(%arg0)
    // CHECK-DAG: %[[ARG1_ID:.*]] = "tf.Identity"(%arg1)
    // CHECK-DAG: %[[ARG0_AGG:.*]] = "tf.CustomAggregator"(%[[ARG0_ID]])
    // CHECK-DAG: %[[ARG1_AGG:.*]] = "tf.CustomAggregator"(%[[ARG1_ID]])
    // CHECK: %[[RES:.*]] = "tf.XlaCallModule"(%[[ARG0_AGG]], %[[ARG1_AGG]])
    // CHECK: %[[RES_AGG:.*]] = "tf.CustomAggregator"(%[[RES]])
    // CHECK-DAG: %[[RES_ID:.*]] = "tf.Identity"(%[[RES_AGG]])
    // CHECK: return %[[RES_ID]] : tensor<?x10xf32>
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<?x100352xf32>) -> tensor<?x100352xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<100352x10xf32>) -> tensor<100352x10xf32>
    %2 = "tf.XlaCallModule"(%0, %1) <{
        Sout = [#tf_type.shape<?x10>], dim_args_spec = [],
        disabled_checks = [], function_list = [],
        has_token_input_output = false, module = "", platforms = [],
        version = 5 : i64
    }> {
        _entry_function = @composite_dot_general_fn_1,
        _original_entry_function = "composite_dot_general_fn_1",
        _tfl_quant_trait = "fully_quantizable"
    } : (tensor<?x100352xf32>, tensor<100352x10xf32>) -> tensor<?x10xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<?x10xf32>) -> tensor<?x10xf32>
    return %3 : tensor<?x10xf32>
  }

  // CHECK-LABEL: func.func private @composite_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<?x100352xf32>, %arg1: tensor<100352x10xf32>) -> tensor<?x10xf32> {
    // CHECK-NOT: tf.CustomAggregator
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x100352xf32>, tensor<100352x10xf32>) -> tensor<?x10xf32>
    return %0 : tensor<?x10xf32>
  }
}
