// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions -quant-quantize-composite-functions | FileCheck --check-prefix=TF %s
// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions -quant-quantize-composite-functions='target-opset=XLA' | FileCheck --check-prefix=XLA %s
// RUN: tf-quant-opt %s -split-input-file -quant-insert-quantized-functions -quant-quantize-composite-functions='target-opset=XLA enable-per-channel-quantization=true' | FileCheck --check-prefix=XLA_PerChannel %s

module {
  func.func @conv_with_int_per_layer(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[-0.0192535277, -5.998660e-03]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-0.282878935, -0.211567819], [-0.248810023, -0.0989695191]], [[0.400888503, 0.0803082585], [-0.0671417713, -0.23301053]]], [[[0.345862567, 0.311298311], [-0.595954239, 0.202630222]], [[-0.606417357, -0.257358253], [-0.3036502, -0.35013032]]]]> : tensor<2x2x2x2xf32>} : () -> tensor<2x2x2x2xf32>
    %cst_1 = "tf.Const"() {device = "", value = dense<[-0.0291469581, 0.0106381178]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_2 = "tf.Const"() {device = "", value = dense<[[[[0.208403707, 0.478067577], [0.593097508, -0.305721074]], [[-0.114346057, 0.583530128], [0.211413622, -0.606618404]]], [[[0.314416587, -0.260997623], [-0.375806928, 0.0813755393]], [[-0.208318114, 0.275989294], [-3.469230e-01, -0.406548172]]]]> : tensor<2x2x2x2xf32>} : () -> tensor<2x2x2x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[1.878980e-02, 0.988373816]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.PartitionedCall"(%0, %cst_0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_20} : (tensor<*xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.36084348]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.DumpTensor"(%2) {device = "", enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"} : (tensor<*xf32>) -> ()
    %3 = "tf.PartitionedCall"(%2, %cst_2, %cst_1) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_10} : (tensor<*xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %4 = "quantization.stats"(%3) {layerStats = dense<[0.000000e+00, 0.18486841]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %6 = "quantization.stats"(%5) {layerStats = dense<[0.000000e+00, 0.18486841]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.PartitionedCall"(%2, %cst_2, %cst_1) {config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_00} : (tensor<*xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %8 = "quantization.stats"(%7) {layerStats = dense<[0.000000e+00, 0.18486841]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %9 = "tf.PartitionedCall"(%0, %cst_0, %cst) {config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_00} : (tensor<*xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %10 = "quantization.stats"(%9) {layerStats = dense<[0.000000e+00, 0.36084348]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.DumpTensor"(%10) {device = "", enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"} : (tensor<*xf32>) -> ()
    "tf.DumpTensor"(%4) {device = "", enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"} : (tensor<*xf32>) -> ()
    "tf.DumpTensor"(%8) {device = "", enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"} : (tensor<*xf32>) -> ()
    func.return %6 : tensor<*xf32>
  }

  func.func private @composite_conv2d_with_bias_and_relu6_fn_10(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_1_00(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_20(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_2_00(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }

// TF-LABEL: @conv_with_int_per_layer
// TF-DAG: %[[w0_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.282878935, -0.211567819
// TF-DAG: %[[b0_float:.*]] = "tf.Const"() <{value = dense<[-0.0192535277, -5.998660e-03]> : tensor<2xf32>}> : () -> tensor<2xf32>
// TF-DAG: %[[w1_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.208403707, 0.478067577
// TF-DAG: %[[b1_float:.*]] = "tf.Const"() <{value = dense<[-0.0291469581, 0.0106381178]> : tensor<2xf32>}> : () -> tensor<2xf32>
// TF-DAG: %[[w0_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-59, -44
// TF-DAG: %[[b0_quantized:.*]] = "tf.Const"() <{value = dense<[-1040, -324]> : tensor<2xi32>}> : () -> tensor<2xi32>
// TF-DAG: %[[w1_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}44, 100
// TF-DAG: %[[b1_quantized:.*]] = "tf.Const"() <{value = dense<[-4312, 1574]> : tensor<2xi32>}> : () -> tensor<2xi32>
// TF-DAG: %[[in_scale:.*]] = "tf.Const"() <{value = dense<0.00387597573> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[in_out_zp:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}> : () -> tensor<i32>
// TF-DAG: %[[w0_scale:.*]] = "tf.Const"() <{value = dense<0.00477493973> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[w_b_zp:.*]]  = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// TF-DAG: %[[b0_scale:.*]] = "tf.Const"() <{value = dense<1.85075514E-5> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[mid_scale:.*]] = "tf.Const"() <{value = dense<0.00141507247> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[w1_scale:.*]] = "tf.Const"() <{value = dense<0.00477652298> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[b1_scale:.*]] = "tf.Const"() <{value = dense<6.75912588E-6> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[out_scale:.*]] = "tf.Const"() <{value = dense<7.24974147E-4> : tensor<f32>}> : () -> tensor<f32>
// TF-DAG: %[[arg_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[in_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}>
// TF-DAG: %[[conv0_quantized:.*]] = "tf.PartitionedCall"(%[[arg_quantized]], %[[w0_quantized]], %[[b0_quantized]], %[[in_scale]], %[[in_out_zp]], %[[w0_scale]], %[[w_b_zp]], %[[b0_scale]], %[[w_b_zp]], %[[mid_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu6_fn_1}>
// TF-DAG: %[[conv0_dequantized:.*]] = "tf.PartitionedCall"(%[[conv0_quantized]], %[[mid_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}>
// TF-DAG: %[[conv1_quantized:.*]] = "tf.PartitionedCall"(%[[conv0_quantized]], %[[w1_quantized]], %[[b1_quantized]], %[[mid_scale]], %[[in_out_zp]], %[[w1_scale]], %[[w_b_zp]], %[[b1_scale]], %[[w_b_zp]], %[[out_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu6_fn_0}>
// TF-DAG: %[[conv1_dequantized_0:.*]] = "tf.PartitionedCall"(%[[conv1_quantized]], %[[out_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}>
// TF-DAG: %[[conv1_dequantized_1:.*]] = "tf.PartitionedCall"(%[[conv1_quantized]], %[[out_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}>
// TF-DAG: %[[identity:.*]] = "tf.Identity"(%[[conv1_dequantized_1]])
// TF-DAG: %[[conv0_float:.*]] = "tf.PartitionedCall"(%arg0, %[[w0_float]], %[[b0_float]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_00}> {device = ""}
// TF-DAG: %[[conv1_float:.*]] = "tf.PartitionedCall"(%[[conv0_dequantized]], %[[w1_float]], %[[b1_float]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_00}> {device = ""}
// TF-DAG: "tf.DumpTensor"(%[[conv0_dequantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}> {device = ""}
// TF-DAG: "tf.DumpTensor"(%[[conv0_float]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}> {device = ""}
// TF-DAG: "tf.DumpTensor"(%[[conv1_dequantized_0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> {device = ""}
// TF-DAG: "tf.DumpTensor"(%[[conv1_float]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> {device = ""}
// TF-DAG: return %[[identity]]

// XLA-LABEL: func @conv_with_int_per_layer
// XLA-DAG: %[[w0_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.282878935, -0.211567819
// XLA-DAG: %[[b0_float:.*]] = "tf.Const"() <{value = dense<[-0.0192535277, -5.998660e-03]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA-DAG: %[[w1_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.208403707, 0.478067577
// XLA-DAG: %[[b1_float:.*]] = "tf.Const"() <{value = dense<[-0.0291469581, 0.0106381178]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA-DAG: %[[w0_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-59, -44
// XLA-DAG: %[[b0_quantized:.*]] = "tf.Const"() <{value = dense<[-1040, -324]> : tensor<2xi32>}> : () -> tensor<2xi32>
// XLA-DAG: %[[w1_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}44, 100
// XLA-DAG: %[[b1_quantized:.*]] = "tf.Const"() <{value = dense<[-4312, 1574]> : tensor<2xi32>}> : () -> tensor<2xi32>
// XLA-DAG: %[[in_scale:.*]] = "tf.Const"() <{value = dense<0.00387597573> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[in_out_zp:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}> : () -> tensor<i32>
// XLA-DAG: %[[w0_scale:.*]] = "tf.Const"() <{value = dense<0.00477493973> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[w_b_zp:.*]]  = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// XLA-DAG: %[[b0_scale:.*]] = "tf.Const"() <{value = dense<1.85075514E-5> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[mid_scale:.*]] = "tf.Const"() <{value = dense<0.00141507247> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[w1_scale:.*]] = "tf.Const"() <{value = dense<0.00477652298> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[b1_scale:.*]] = "tf.Const"() <{value = dense<6.75912588E-6> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[out_scale:.*]] = "tf.Const"() <{value = dense<7.24974147E-4> : tensor<f32>}> : () -> tensor<f32>
// XLA-DAG: %[[quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[in_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}>
// XLA-DAG: %[[conv0_dequantized:.*]] = "tf.PartitionedCall"(%[[quantized]], %[[w0_quantized]], %[[b0_quantized]], %[[in_scale]], %[[in_out_zp]], %[[w0_scale]], %[[w_b_zp]], %[[b0_scale]], %[[w_b_zp]], %[[mid_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu6_float_output_fn_1}>
// XLA-DAG: %[[conv0_quantized:.*]] = "tf.PartitionedCall"(%[[quantized]], %[[w0_quantized]], %[[b0_quantized]], %[[in_scale]], %[[in_out_zp]], %[[w0_scale]], %[[w_b_zp]], %[[b0_scale]], %[[w_b_zp]], %[[mid_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu6_fn_1}>
// XLA-DAG: %[[conv1_dequantized:.*]] = "tf.PartitionedCall"(%[[conv0_quantized]], %[[w1_quantized]], %[[b1_quantized]], %[[mid_scale]], %[[in_out_zp]], %[[w1_scale]], %[[w_b_zp]], %[[b1_scale]], %[[w_b_zp]], %[[out_scale]], %[[in_out_zp]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu6_float_output_fn_0}>
// XLA-DAG: %[[identity:.*]] = "tf.Identity"(%[[conv1_dequantized]])
// XLA-DAG: %[[conv0_float:.*]] = "tf.PartitionedCall"(%arg0, %[[w0_float]], %[[b0_float]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_00}> {device = ""}
// XLA-DAG: %[[conv1_float:.*]] = "tf.PartitionedCall"(%[[conv0_dequantized]], %[[w1_float]], %[[b1_float]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_00}> {device = ""}
// XLA-DAG: "tf.DumpTensor"(%[[conv0_dequantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}> {device = ""}
// XLA-DAG: "tf.DumpTensor"(%[[conv0_float]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}> {device = ""}
// XLA-DAG: "tf.DumpTensor"(%[[conv1_dequantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> {device = ""}
// XLA-DAG: "tf.DumpTensor"(%[[conv1_float]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv_with_dump", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> {device = ""}
// XLA-DAG: return %[[identity]]

// XLA_PerChannel-LABEL: func @conv_with_int_per_layer
// XLA_PerChannel-DAG: %[[PerChannel_w0_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.282878935, -0.211567819
// XLA_PerChannel-DAG: %[[b0_float:.*]] = "tf.Const"() <{value = dense<[-0.0192535277, -5.998660e-03]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[w1_float:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.208403707, 0.478067577
// XLA_PerChannel-DAG: %[[b1_float:.*]] = "tf.Const"() <{value = dense<[-0.0291469581, 0.0106381178]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[w0_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-59, -77
// XLA_PerChannel-DAG: %[[b0_quantized:.*]] = "tf.Const"() <{value = dense<[-1040, -561]> : tensor<2xi32>}> : () -> tensor<2xi32>
// XLA_PerChannel-DAG: %[[w1_quantized:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}45, 100
// XLA_PerChannel-DAG: %[[b1_quantized:.*]] = "tf.Const"() <{value = dense<[-4411, 1574]> : tensor<2xi32>}> : () -> tensor<2xi32>
// XLA_PerChannel-DAG: %[[in_scale:.*]] = "tf.Const"() <{value = dense<0.00387597573> : tensor<f32>}> : () -> tensor<f32>
// XLA_PerChannel-DAG: %[[in_out_zp:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}> : () -> tensor<i32>
// XLA_PerChannel-DAG: %[[w0_scale:.*]] = "tf.Const"() <{value = dense<[0.00477493973, 0.00275693159]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[w_b_zp:.*]]  = "tf.Const"() <{value = dense<0> : tensor<2xi32>}> : () -> tensor<2xi32>
// XLA_PerChannel-DAG: %[[b0_scale:.*]] = "tf.Const"() <{value = dense<[1.85075514E-5, 1.06858006E-5]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[mid_scale:.*]] = "tf.Const"() <{value = dense<0.00141507247> : tensor<f32>}> : () -> tensor<f32>
// XLA_PerChannel-DAG: %[[w1_scale:.*]] = "tf.Const"() <{value = dense<[0.00467005931, 0.00477652298]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[b1_scale:.*]] = "tf.Const"() <{value = dense<[6.60847217E-6, 6.75912588E-6]> : tensor<2xf32>}> : () -> tensor<2xf32>
// XLA_PerChannel-DAG: %[[out_scale:.*]] = "tf.Const"() <{value = dense<7.24974147E-4> : tensor<f32>}> : () -> tensor<f32>
}

// TODO(b/308773062): Add whole_model unit-test

// -----

module {
  func.func @matmul2_with_int_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "tf.PartitionedCall"(%2, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "quantization.stats"(%4) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%5) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %6 = "tf.PartitionedCall"(%2, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%6) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    return %5 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: func @matmul2_with_int_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_8]], %[[cst_9]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_3]], %[[cst_6]], %[[cst_5]], %[[cst_10]], %[[cst_9]], %[[cst_7]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_7]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: return %[[pc_5]]

// XLA-LABEL: func @matmul2_with_int_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// XLA-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_6]], %[[cst_7]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_1}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_6]], %[[cst_7]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_3]], %[[cst_8]], %[[cst_5]], %[[cst_9]], %[[cst_7]], %[[cst_10]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: return %[[pc_4]]
}

// -----

module {
  func.func @matmul2_softmax_with_int_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "tf.Softmax"(%2) {T = "tfdtype$DT_FLOAT"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "quantization.stats"(%4) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = "tf.PartitionedCall"(%5, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = "quantization.stats"(%6) {layerStats = dense<[0.000000e+00, 0.4]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%7) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %8 = "tf.PartitionedCall"(%4, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%8) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    return %7 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: func @matmul2_softmax_with_int_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// TF-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_9]], %[[cst_10]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_2]]) {T = "tfdtype$DT_FLOAT"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_7]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_3]], %[[cst_7]], %[[cst_5]], %[[cst_11]], %[[cst_10]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_5]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_7]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: return %[[pc_6]]

// XLA-LABEL: func @matmul2_softmax_with_int_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// XLA-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_9]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_1}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_1]]) {T = "tfdtype$DT_FLOAT"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_3]], %[[cst_6]], %[[cst_5]], %[[cst_10]], %[[cst_8]], %[[cst_11]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: return %[[pc_4]]
}


// -----

module {
  func.func @matmul2_concat_with_int_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x4xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_1 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "quantization.stats"(%2) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "tf.PartitionedCall"(%4, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = "quantization.stats"(%5) {layerStats = dense<[0.000000e+00, 0.4]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%6) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %7 = "tf.PartitionedCall"(%2, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%7) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %8 = "tf.ConcatV2"(%2, %6, %cst_1) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
    return %8 : tensor<2x4xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: func @matmul2_concat_with_int_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_3]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_2]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[cc_0:.*]] = "tf.ConcatV2"(%[[pc_1]], %[[pc_5]], %[[cst_0]]) : (tensor<2x2xi8>, tensor<2x2xi8>, tensor<i32>) -> tensor<2x4xi8>
// TF-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[cc_0]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: return %[[pc_7]]

// XLA-LABEL: func @matmul2_concat_with_int_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_3]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_3]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_2]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[cc_0:.*]] = "tf.ConcatV2"(%[[pc_2]], %[[pc_5]], %[[cst_0]]) : (tensor<2x2xi8>, tensor<2x2xi8>, tensor<i32>) -> tensor<2x4xi8>
// XLA-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[cc_0]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// XLA-DAG: return %[[pc_7]]
}

// -----

module {
  func.func @matmul2_with_float_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "quantization.stats"(%3) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "tf.PartitionedCall"(%4, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = "quantization.stats"(%5) {layerStats = dense<[0.000000e+00, 0.4]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%6) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %7 = "tf.PartitionedCall"(%3, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%7) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    return %7 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: @matmul2_with_float_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// TF-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_9]], %[[cst_10]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_7]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_3]], %[[cst_7]], %[[cst_5]], %[[cst_11]], %[[cst_10]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_5]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_7]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: return %[[pc_7]]

// XLA-LABEL: @matmul2_with_float_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// XLA-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_9]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_1}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_3]], %[[cst_6]], %[[cst_5]], %[[cst_10]], %[[cst_8]], %[[cst_11]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: return %[[pc_5]]
}


// -----
module {
  func.func @matmul2_softmax_with_float_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "tf.Softmax"(%3) {T = "tfdtype$DT_FLOAT"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "quantization.stats"(%4) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = "tf.PartitionedCall"(%5, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %7 = "quantization.stats"(%6) {layerStats = dense<[0.000000e+00, 0.4]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%7) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %8 = "tf.PartitionedCall"(%4, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%8) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    return %8 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: func @matmul2_softmax_with_float_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// TF-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_9]], %[[cst_10]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_3]]) {T = "tfdtype$DT_FLOAT"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_7]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_3]], %[[cst_7]], %[[cst_5]], %[[cst_11]], %[[cst_10]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_5]], %[[cst_8]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_7]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: return %[[pc_7]]

// XLA-LABEL: func @matmul2_softmax_with_float_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// XLA-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_4]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_2]], %[[cst_4]], %[[cst_5]], %[[cst_7]], %[[cst_8]], %[[cst_9]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_1}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_2]]) {T = "tfdtype$DT_FLOAT"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_6]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_3]], %[[cst_6]], %[[cst_5]], %[[cst_10]], %[[cst_8]], %[[cst_11]], %[[cst_5]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: return %[[pc_5]]
}


// -----
module {
  func.func @matmul2_concat_with_float_per_layer(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> tensor<2x4xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_1 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[0.000000e+00, 0.1]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "tf.PartitionedCall"(%0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 0.2]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%2) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %3 = "tf.PartitionedCall"(%arg0, %cst) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%3) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"} : (tensor<2x2xf32>) -> ()
    %4 = "quantization.stats"(%3) {layerStats = dense<[0.000000e+00, 0.3]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %5 = "tf.PartitionedCall"(%4, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = "quantization.stats"(%5) {layerStats = dense<[0.000000e+00, 0.4]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%6) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %7 = "tf.PartitionedCall"(%3, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "tf.DumpTensor"(%7) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"} : (tensor<2x2xf32>) -> ()
    %8 = "tf.ConcatV2"(%3, %7, %cst_1) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
    return %8 : tensor<2x4xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// TF-LABEL: func @matmul2_concat_with_float_per_layer
// TF-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}
// TF-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// TF-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// TF-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// TF-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// TF-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// TF-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// TF-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// TF-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// TF-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// TF-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// TF-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// TF-DAG: %[[cst_12:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// TF-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_5]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_3]], %[[cst_5]], %[[cst_6]], %[[cst_10]], %[[cst_11]], %[[cst_7]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_1}
// TF-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[pc_1]], %[[cst_7]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_2]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// TF-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_8]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// TF-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_4]], %[[cst_4]], %[[cst_8]], %[[cst_6]], %[[cst_12]], %[[cst_11]], %[[cst_9]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_fn_0}
// TF-DAG: %[[pc_6:.*]] = "tf.PartitionedCall"(%[[pc_5]], %[[cst_9]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @dequantize_i8}
// TF-DAG: "tf.DumpTensor"(%[[pc_6]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[pc_7:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// TF-DAG: "tf.DumpTensor"(%[[pc_7]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// TF-DAG: %[[cc_0:.*]] = "tf.ConcatV2"(%[[pc_3]], %[[pc_7]], %[[cst_0]]) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
// TF-DAG: return %[[cc_0]]

// XLA-LABEL: func @matmul2_concat_with_float_per_layer
// XLA-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}
// XLA-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// XLA-DAG: %[[cst_2:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// XLA-DAG: %[[cst_3:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-105, 91
// XLA-DAG: %[[cst_4:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-28, -94
// XLA-DAG: %[[cst_5:.*]] = "tf.Const"() <{value = dense<3.92156857E-4> : tensor<f32>}
// XLA-DAG: %[[cst_6:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}
// XLA-DAG: %[[cst_7:.*]] = "tf.Const"() <{value = dense<0.00117647066> : tensor<f32>}
// XLA-DAG: %[[cst_8:.*]] = "tf.Const"() <{value = dense<0.00602002116> : tensor<f32>}
// XLA-DAG: %[[cst_9:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}
// XLA-DAG: %[[cst_10:.*]] = "tf.Const"() <{value = dense<7.84313714E-4> : tensor<f32>}
// XLA-DAG: %[[cst_11:.*]] = "tf.Const"() <{value = dense<0.0075123054> : tensor<f32>}
// XLA-DAG: %[[cst_12:.*]] = "tf.Const"() <{value = dense<0.00156862743> : tensor<f32>}
// XLA-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_5]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[pc_0]], %[[cst_3]], %[[cst_5]], %[[cst_6]], %[[cst_8]], %[[cst_9]], %[[cst_10]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_1}
// XLA-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_2]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// XLA-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_7]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantize_i8}
// XLA-DAG: %[[pc_4:.*]] = "tf.PartitionedCall"(%[[pc_3]], %[[cst_4]], %[[cst_7]], %[[cst_6]], %[[cst_11]], %[[cst_9]], %[[cst_12]], %[[cst_6]]) <{config = "", config_proto = "", executor_type = "", f = @quantized_matmul_float_output_fn_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_4]]) <{enabled = false, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[pc_5:.*]] = "tf.PartitionedCall"(%[[pc_2]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// XLA-DAG: "tf.DumpTensor"(%[[pc_5]]) <{enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// XLA-DAG: %[[cc_0:.*]] = "tf.ConcatV2"(%[[pc_2]], %[[pc_5]], %[[cst_0]]) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
// XLA-DAG: return %[[cc_0]]
}
