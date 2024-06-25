// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=whole_model' | FileCheck --check-prefix=WholeModel %s
// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=int_per_layer' | FileCheck --check-prefix=IntPerLayer %s
// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=float_per_layer' | FileCheck --check-prefix=FloatPerLayer %s


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
// WholeModel-DAG: %[[w:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}1.600000e-01, 1.000000e-01
// WholeModel-DAG: %[[b:.*]] = "tf.Const"() <{value = dense<[-2.000000e+00, 3.000000e+00
// WholeModel-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}>
// WholeModel-DAG: %[[output1:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[output1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> : (tensor<*xf32>) -> ()
// WholeModel-DAG: return %[[output0]], %[[output1]]

// IntPerLayer-LABEL: func @conv
// IntPerLayer-DAG: %[[w:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}1.600000e-01, 1.000000e-01
// IntPerLayer-DAG: %[[b:.*]] = "tf.Const"() <{value = dense<[-2.000000e+00, 3.000000e+00
// IntPerLayer-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// IntPerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// IntPerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %cst, %cst_0) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> : (tensor<*xf32>) -> ()
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> : (tensor<*xf32>) -> ()
// IntPerLayer-DAG: return %[[output0]], %[[output1_quantized]]

// FloatPerLayer-LABEL: func @conv
// FloatPerLayer-DAG: %[[w:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}1.600000e-01, 1.000000e-01
// FloatPerLayer-DAG: %[[b:.*]] = "tf.Const"() <{value = dense<[-2.000000e+00, 3.000000e+00
// FloatPerLayer-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}
// FloatPerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// FloatPerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w]], %[[b]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> : (tensor<*xf32>) -> ()
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "conv", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}> : (tensor<*xf32>) -> ()
// FloatPerLayer-DAG: return %[[output0]], %[[output1_unquantized]]
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
// WholeModel-DAG: %[[b0:.*]] = "tf.Const"() <{value = dense<0.000000e+00>
// WholeModel-DAG: %[[b1:.*]] = "tf.Const"() <{value = dense<1.000000e+00>
// WholeModel-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.193340182, 0.285152316
// WholeModel-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.174680978, -0.367524445
// WholeModel-DAG: %[[output0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[output0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}>
// WholeModel-DAG: %[[output1:.*]] = "tf.PartitionedCall"(%[[output0]], %[[w1]], %[[b1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[output1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}>
// WholeModel-DAG: return %[[output1]]

// IntPerLayer-LABEL: func @multiple_conv2d
// IntPerLayer-DAG: %[[b0:.*]] = "tf.Const"() <{value = dense<0.000000e+00>
// IntPerLayer-DAG: %[[b1:.*]] = "tf.Const"() <{value = dense<1.000000e+00>
// IntPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.193340182, 0.285152316
// IntPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.174680978, -0.367524445
// IntPerLayer-DAG: %[[output0_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// IntPerLayer-DAG: %[[output0_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_0}>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output0_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output0_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}>
// IntPerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%[[output0_quantized]], %[[w1]], %[[b1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// IntPerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%[[output0_quantized]], %[[w1]], %[[b1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}>
// IntPerLayer-DAG: return %[[output1_quantized]]

// FloatPerLayer-LABEL: func @multiple_conv2d
// FloatPerLayer-DAG: %[[b0:.*]] = "tf.Const"() <{value = dense<0.000000e+00>
// FloatPerLayer-DAG: %[[b1:.*]] = "tf.Const"() <{value = dense<1.000000e+00>
// FloatPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}0.193340182, 0.285152316
// FloatPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[\[\[}}-0.174680978, -0.367524445
// FloatPerLayer-DAG: %[[output0_quantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// FloatPerLayer-DAG: %[[output0_unquantized:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]], %[[b0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_2_0}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output0_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output0_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_2", node_name = "Conv2D"}
// FloatPerLayer-DAG: %[[output1_quantized:.*]] = "tf.PartitionedCall"(%[[output0_unquantized]], %[[w1]], %[[b1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// FloatPerLayer-DAG: %[[output1_unquantized:.*]] = "tf.PartitionedCall"(%[[output0_unquantized]], %[[w1]], %[[b1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_1_0}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output1_quantized]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[output1_unquantized]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "multiple_conv2d", log_dir_path = "/tmp/dumps/composite_conv2d_with_bias_and_relu6_fn_1", node_name = "Conv2D_1"}
// FloatPerLayer-DAG: return %[[output1_unquantized]]
}

// -----

module {
  func.func @matmul2(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "tf.PartitionedCall"(%arg0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2"("MatMul") at "QuantizationUnit(\12\06MatMul\1a\07matmul2)"))
    %1 = "tf.PartitionedCall"(%0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2"("MatMul_1") at "QuantizationUnit(\12\08MatMul_1\1a\07matmul2)"))
    return %1 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// WholeModel-LABEL: func @matmul2
// WholeModel-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// WholeModel-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// WholeModel-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// WholeModel-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// WholeModel-DAG: return %[[m1]]

// IntPerLayer-LABEL: func @matmul2
// IntPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// IntPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// IntPerLayer-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: %[[m0_1:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m0_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: %[[m1_0:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m1_0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: return %[[m1]] : tensor<2x2xf32>

// FloatPerLayer-LABEL: func @matmul2
// FloatPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// FloatPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// FloatPerLayer-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: %[[m0_1:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m0_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0_1]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: %[[m1_0:.*]] = "tf.PartitionedCall"(%[[m0_1]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m1_0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: return %[[m1_0]] : tensor<2x2xf32>
}

// -----

module {
  func.func @matmul2_softmax(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x2xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %0 = "tf.PartitionedCall"(%arg0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2_softmax"("MatMul") at "QuantizationUnit(\12\06MatMul\1a\0fmatmul2_softmax)"))
    %1 = "tf.Softmax"(%0) {T = "tfdtype$DT_FLOAT"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "tf.PartitionedCall"(%1, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2_softmax"("MatMul_1") at "QuantizationUnit(\12\08MatMul_1\1a\0fmatmul2_softmax)"))
    return %2 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// WholeModel-LABEL: func @matmul2_softmax
// WholeModel-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// WholeModel-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// WholeModel-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_0]]) {T = "tfdtype$DT_FLOAT"}
// WholeModel-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[pc_0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// WholeModel-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// WholeModel-DAG: return %[[pc_1]]

// IntPerLayer-LABEL: func @matmul2_softmax
// IntPerLayer-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// IntPerLayer-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// IntPerLayer-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// IntPerLayer-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// IntPerLayer-DAG: "tf.DumpTensor"(%[[pc_0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// IntPerLayer-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// IntPerLayer-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_0]]) {T = "tfdtype$DT_FLOAT"}
// IntPerLayer-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// IntPerLayer-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// IntPerLayer-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// IntPerLayer-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// IntPerLayer-DAG: return %[[pc_2]]

// FloatPerLayer-LABEL: func @matmul2_softmax
// FloatPerLayer-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344, 0.54962182
// FloatPerLayer-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893, -0.708605706
// FloatPerLayer-DAG: %[[pc_0:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// FloatPerLayer-DAG: %[[pc_1:.*]] = "tf.PartitionedCall"(%arg0, %[[cst_0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[pc_0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[pc_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// FloatPerLayer-DAG: %[[sm_0:.*]] = "tf.Softmax"(%[[pc_1]]) {T = "tfdtype$DT_FLOAT"}
// FloatPerLayer-DAG: %[[pc_2:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// FloatPerLayer-DAG: %[[pc_3:.*]] = "tf.PartitionedCall"(%[[sm_0]], %[[cst_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[pc_2]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[pc_3]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_softmax", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// FloatPerLayer-DAG: return %[[pc_3]]
}

// -----

module {
  func.func @matmul2_concat(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x4xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[-0.211145893, -0.708605706], [-0.954062759, -0.614013135]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %cst_1 = "tf.Const"() { value = dense<-1> : tensor<i32> } : () -> tensor<i32>
    %0 = "tf.PartitionedCall"(%arg0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2_concat"("MatMul") at "QuantizationUnit(\12\06MatMul\1a\0ematmul2_concat)"))
    %1 = "tf.PartitionedCall"(%0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32> loc(callsite("test@matmul2_concat"("MatMul_1") at "QuantizationUnit(\12\08MatMul_1\1a\0ematmul2_concat)"))
    %2 = "tf.ConcatV2"(%0, %1, %cst_1) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
  func.func private @composite_matmul_fn_2(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @composite_matmul_fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_b", device = "", transpose_a = false, transpose_b = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// WholeModel-LABEL: func @matmul2_concat
// WholeModel-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// WholeModel-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// WholeModel-DAG: %[[axis:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}
// WholeModel-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"}
// WholeModel-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}
// WholeModel-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}
// WholeModel-DAG: %[[c:.*]] = "tf.ConcatV2"(%[[m0]], %[[m1]], %[[axis]])
// WholeModel-DAG: return %[[c]]

// IntPerLayer-LABEL: func @matmul2_concat
// IntPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// IntPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// IntPerLayer-DAG: %[[axis:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
// IntPerLayer-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: %[[m0_1:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m0_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: %[[m1_0:.*]] = "tf.PartitionedCall"(%[[m0]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: "tf.DumpTensor"(%[[m1_0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// IntPerLayer-DAG: %4 = "tf.ConcatV2"(%[[m0]], %[[m1]], %[[axis]]) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
// IntPerLayer-DAG: return %4 : tensor<2x4xf32>

// FloatPerLayer-LABEL: func @matmul2_concat
// FloatPerLayer-DAG: %[[w0:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.630731344
// FloatPerLayer-DAG: %[[w1:.*]] = "tf.Const"() <{value = dense<{{\[\[}}-0.211145893
// FloatPerLayer-DAG: %[[axis:.*]] = "tf.Const"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
// FloatPerLayer-DAG: %[[m0:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: %[[m0_1:.*]] = "tf.PartitionedCall"(%arg0, %[[w0]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_2_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m0]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m0_1]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_2", node_name = "MatMul"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: %[[m1:.*]] = "tf.PartitionedCall"(%[[m0_1]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: %[[m1_0:.*]] = "tf.PartitionedCall"(%[[m0_1]], %[[w1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn_1_0}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m1]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[m1_0]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "matmul2_concat", log_dir_path = "/tmp/dumps/composite_matmul_fn_1", node_name = "MatMul_1"}> : (tensor<2x2xf32>) -> ()
// FloatPerLayer-DAG: %4 = "tf.ConcatV2"(%1, %[[m1_0]], %[[axis]]) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> tensor<2x4xf32>
// FloatPerLayer-DAG: return %4 : tensor<2x4xf32>
}
