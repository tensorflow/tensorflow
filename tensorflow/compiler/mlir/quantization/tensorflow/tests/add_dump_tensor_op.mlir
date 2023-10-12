// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=whole_model' | FileCheck --check-prefix=WholeModel %s
// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=per_layer' | FileCheck --check-prefix=PerLayer %s

module {
  func.func @conv(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %cst = "tf.Const"() {value = dense<[[[[1.600000e-01, 1.000000e-01], [5.100000e-01, 5.400000e-01], [-5.000000e-01, 4.100000e-01]], [[-3.500000e-01, 5.000000e-02], [-0.00999999977, 1.600000e-01], [-4.800000e-01, -2.400000e-01]]], [[[-3.500000e-01, -2.100000e-01], [-1.400000e-01, -2.000000e-02], [4.800000e-01, 3.500000e-01]], [[-1.900000e-01, 3.200000e-01], [0.00999999977, -7.000000e-02], [2.000000e-01, -4.000000e-02]]]]> : tensor<2x2x3x2xf32>} : () -> tensor<2x2x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "tf.PartitionedCall"(%arg0, %cst, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>, tensor<2xf32>) -> tensor<*xf32> loc(callsite("test@conv"("Conv2D") at "QuantizationUnit(\12\06Conv2D\1a\04conv)"))
    %1 = "tf.PartitionedCall"(%arg0, %cst, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>, tensor<2xf32>) -> tensor<*xf32> loc(callsite("test@conv"("Conv2D_1") at "QuantizationUnit(\12\08Conv2D_1\1a\04conv)"))
    func.return %0, %1 : tensor<*xf32>, tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_2(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x2x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
  func.func private @composite_conv2d_with_bias_and_relu6_fn_1(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x2x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x2x2x3xf32>, tensor<2x2x3x2xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }

// WholeModel-LABEL: func @conv
// WholeModel-DAG: %[[w:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}1.600000e-01, 1.000000e-01
// WholeModel-DAG: %[[b:.*]] = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00
// WholeModel-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// WholeModel-DAG: %[[output1:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}
// WholeModel-DAG: "tf.DumpTensor"(%[[output1]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"} : (tensor<*xf32>) -> ()
// WholeModel-DAG: return %[[output0]], %[[output1]]

// PerLayer-LABEL: func @conv
// PerLayer-DAG: %[[w:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}1.600000e-01, 1.000000e-01
// PerLayer-DAG: %[[b:.*]] = "tf.Const"() {value = dense<[-2.000000e+00, 3.000000e+00
// PerLayer-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// PerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}
// PerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %cst, %cst_0) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}
// PerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"} : (tensor<*xf32>) -> ()
// PerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"} : (tensor<*xf32>) -> ()
// PerLayer-DAG: return %[[output0]], %[[output1_quantized]]
}

// -----

module {
  func.func @multiple_conv2d(%arg: tensor<?x2x2x2xf32>) -> tensor<?x2x2x2xf32> {
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_1 = "tf.Const"() {value = dense<[[[[0.193340182, 0.285152316], [0.41538316, -0.313452125]], [[0.188379049, 0.0693640113], [-0.199678659, -0.0629909635]]], [[[0.141592324, 0.554834187], [-0.224576354, 0.103607118]], [[0.134974658, -2.952230e-02], [-0.15929231, -0.538676262]]]]> : tensor<2x2x2x2xf32>} : () -> tensor<2x2x2x2xf32>
    %cst_2 = "tf.Const"() {value = dense<[[[[-0.174680978, -0.367524445], [-0.0481151938, -0.154707015]], [[-0.0463985205, 0.457213104], [-0.0713823438, 0.0317451358]]], [[[-0.335502505, 0.00602310896], [0.307939529, 0.49636358]], [[-0.223585874, -0.194682062], [0.0728010535, 0.43586427]]]]> : tensor<2x2x2x2xf32>} : () -> tensor<2x2x2x2xf32>
    %0 = "tf.PartitionedCall"(%arg, %cst_1, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2} : (tensor<?x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<?x2x2x2xf32> loc(callsite("test@multiple_conv2d"("Conv2D") at "QuantizationUnit(\12\06Conv2D\1a\0fmultiple_conv2d)"))
    %1 = "tf.PartitionedCall"(%0, %cst_2, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1} : (tensor<?x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<?x2x2x2xf32> loc(callsite("test@multiple_conv2d"("Conv2D_1") at "QuantizationUnit(\12\08Conv2D_1\1a\0fmultiple_conv2d)"))
    return %1 : tensor<?x2x2x2xf32>
  }

  func.func private @composite_conv2d_with_bias_and_relu6_fn_2(%arg0: tensor<?x2x2x2xf32>, %arg1: tensor<2x2x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2x2x2xf32> {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<?x2x2x2xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<?x2x2x2xf32>, tensor<2xf32>) -> tensor<?x2x2x2xf32>
    %2 = "tf.Relu6"(%1) {device = ""} : (tensor<?x2x2x2xf32>) -> tensor<?x2x2x2xf32>
    return %2 : tensor<?x2x2x2xf32>
  }

  func.func private @composite_conv2d_with_bias_and_relu6_fn_1(%arg0: tensor<?x2x2x2xf32>, %arg1: tensor<2x2x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2x2x2xf32> {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<?x2x2x2xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<?x2x2x2xf32>, tensor<2xf32>) -> tensor<?x2x2x2xf32>
    %2 = "tf.Relu6"(%1) {device = ""} : (tensor<?x2x2x2xf32>) -> tensor<?x2x2x2xf32>
    return %2 : tensor<?x2x2x2xf32>
  }

// WholeModel-LABEL: func @multiple_conv2d
// WholeModel-DAG: %[[b0:.*]] = "tf.Const"() {value = dense<0.000000e+00>
// WholeModel-DAG: %[[b1:.*]] = "tf.Const"() {value = dense<1.000000e+00>
// WholeModel-DAG: %[[w0:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}0.193340182, 0.285152316
// WholeModel-DAG: %[[w1:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}-0.174680978, -0.367524445
// WholeModel-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// WholeModel-DAG: "tf.DumpTensor"(%[[output0]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}
// WholeModel-DAG: %[[output1:.*]] = "tf.PartitionedCall"(%[[output0]], %[[w1]], %[[b1]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}
// WholeModel-DAG: "tf.DumpTensor"(%[[output1]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}
// WholeModel-DAG: return %[[output1]]

// PerLayer-LABEL: func @multiple_conv2d
// PerLayer-DAG: %[[b0:.*]] = "tf.Const"() {value = dense<0.000000e+00>
// PerLayer-DAG: %[[b1:.*]] = "tf.Const"() {value = dense<1.000000e+00>
// PerLayer-DAG: %[[w0:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}0.193340182, 0.285152316
// PerLayer-DAG: %[[w1:.*]] = "tf.Const"() {value = dense<{{\[\[\[\[}}-0.174680978, -0.367524445
// PerLayer-DAG: %[[output0_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// PerLayer-DAG: %[[output0_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_0}
// PerLayer-DAG: "tf.DumpTensor"(%[[output0_quantized]]) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}
// PerLayer-DAG: "tf.DumpTensor"(%[[output0_unquantized]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}
// PerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%[[output0_quantized]], %[[w1]], %[[b1]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}
// PerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%[[output0_quantized]], %[[w1]], %[[b1]]) {config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}
// PerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) {enabled = false, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}
// PerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) {enabled = false, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}
// PerLayer-DAG: return %[[output1_quantized]]
}
