// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=whole_model' | FileCheck --check-prefix=WholeModel %s
// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=int_per_layer' | FileCheck --check-prefix=IntPerLayer %s
// RUN: tf-quant-opt %s -split-input-file -quant-add-dump-tensor-op='debugger_type=float_per_layer' | FileCheck --check-prefix=FloatPerLayer %s

module {
  func.func @matmul2(%arg0: tensor<?x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x2xf32>) {
    %0 = stablehlo.constant dense<[-0.211145893, -0.708605706]> : tensor<2xf32>
    %1 = stablehlo.constant dense<[[-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
    %2 = "tf.XlaCallModule"(%arg0, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
    %3 = "tf.XlaCallModule"(%2, %1, %0) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
    return %3 : tensor<?x2xf32>
  }
  func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
    %3 = shape.shape_of %2 : tensor<?x2xf32> -> tensor<2xindex>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [1] : (tensor<2xf32>, tensor<2xindex>) -> tensor<?x2xf32>
    %5 = stablehlo.add %2, %4 : tensor<?x2xf32>
    %6 = stablehlo.clamp %0, %5, %1 : (tensor<f32>, tensor<?x2xf32>, tensor<f32>) -> tensor<?x2xf32>
    return %6 : tensor<?x2xf32>
  }
  func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1(%arg0: tensor<?x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2xf32>) -> tensor<?x2xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
    %3 = shape.shape_of %2 : tensor<?x2xf32> -> tensor<2xindex>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [1] : (tensor<2xf32>, tensor<2xindex>) -> tensor<?x2xf32>
    %5 = stablehlo.add %2, %4 : tensor<?x2xf32>
    %6 = stablehlo.clamp %0, %5, %1 : (tensor<f32>, tensor<?x2xf32>, tensor<f32>) -> tensor<?x2xf32>
    return %6 : tensor<?x2xf32>
  }

// WholeModel-LABEL: func @matmul2
// WholeModel-DAG: %[[b0:.*]] = stablehlo.constant dense<[-0.211145893
// WholeModel-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344, 0.54962182], [0.180364341, -0.764542698]]> : tensor<2x2xf32>
// WholeModel-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// WholeModel-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_2", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// WholeModel-DAG: %[[matmul1_q:.*]] = "tf.XlaCallModule"(%[[matmul0_q]], %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// WholeModel-DAG: "tf.DumpTensor"(%[[matmul1_q]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_1", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// WholeModel-DAG: return %[[matmul1_q]] : tensor<?x2xf32>
// WholeModel-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2
// WholeModel-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1

// IntPerLayer-LABEL: func @matmul2
// IntPerLayer-DAG: %[[b0:.*]] = stablehlo.constant dense<[-0.211145893
// IntPerLayer-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344
// IntPerLayer-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_2", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// IntPerLayer-DAG: %[[matmul0_uq:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0", _stablehlo_version = "1.0.0"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_2", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// IntPerLayer-DAG: %[[matmul1_q:.*]] = "tf.XlaCallModule"(%[[matmul0_q]], %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul1_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_1", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// IntPerLayer-DAG: %[[matmul1_uq:.*]] = "tf.XlaCallModule"(%[[matmul0_q]], %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0", _stablehlo_version = "1.0.0"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul1_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_1", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// IntPerLayer-DAG: return %[[matmul1_q]] : tensor<?x2xf32>
// IntPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2
// IntPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1
// IntPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0
// IntPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0

// FloatPerLayer-LABEL: func @matmul2
// FloatPerLayer-DAG: %[[b0:.*]] = stablehlo.constant dense<[-0.211145893
// FloatPerLayer-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344
// FloatPerLayer-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_2", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// FloatPerLayer-DAG: %[[matmul0_uq:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0", _stablehlo_version = "1.0.0"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_2", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_2", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// FloatPerLayer-DAG: %[[matmul1_q:.*]] = "tf.XlaCallModule"(%[[matmul0_uq]], %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul1_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_1", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// FloatPerLayer-DAG: %[[matmul1_uq:.*]] = "tf.XlaCallModule"(%[[matmul0_uq]], %[[w0]], %[[b0]]) <{Sout = [#tf_type.shape<?x2>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0, _original_entry_function = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0", _stablehlo_version = "1.0.0"} : (tensor<?x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul1_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_with_bias_and_relu6_dynamic_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_with_bias_and_relu6_dynamic_fn_1", node_name = "_empty_node"}> : (tensor<?x2xf32>) -> ()
// FloatPerLayer-DAG: return %[[matmul1_uq]] : tensor<?x2xf32>
// FloatPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2
// FloatPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1
// FloatPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1_0
// FloatPerLayer-DAG: func.func private @composite_dot_general_with_bias_and_relu6_dynamic_fn_2_0
}

// -----

module {
  func.func @matmul_concat(%arg0: tensor<1x2xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<1x3xf32>
    %2 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %3 = stablehlo.concatenate %2, %1, dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

// WholeModel-LABEL: func @matmul_concat
// WholeModel-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344
// WholeModel-DAG: %[[c0:.*]] = stablehlo.constant dense<1.000000e+00
// WholeModel-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]]) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// WholeModel-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_fn_1", node_name = "_empty_node"}> : (tensor<1x3xf32>) -> ()
// WholeModel-DAG: %[[concat:.*]] = stablehlo.concatenate %[[matmul0_q]], %[[c0]], dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
// WholeModel-DAG: return %[[concat]] : tensor<2x3xf32>
// WholeModel-DAG: func.func private @composite_dot_general_fn_1

// IntPerLayer-LABEL: func @matmul_concat
// IntPerLayer-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344
// IntPerLayer-DAG: %[[c0:.*]] = stablehlo.constant dense<1.000000e+00
// IntPerLayer-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]]) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_fn_1", node_name = "_empty_node"}> : (tensor<1x3xf32>) -> ()
// IntPerLayer-DAG: %[[matmul0_uq:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]]) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1_0, _original_entry_function = "composite_dot_general_fn_1_0", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "1.0.0"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// IntPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_fn_1", node_name = "_empty_node"}> : (tensor<1x3xf32>) -> ()
// IntPerLayer-DAG: %[[concat:.*]] = stablehlo.concatenate %[[matmul0_q]], %[[c0]], dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
// IntPerLayer-DAG: return %[[concat]] : tensor<2x3xf32>
// IntPerLayer-DAG: func.func private @composite_dot_general_fn_1
// IntPerLayer-DAG: func.func private @composite_dot_general_fn_1_0

// FloatPerLayer-LABEL: func @matmul_concat
// FloatPerLayer-DAG: %[[w0:.*]] = stablehlo.constant dense<{{\[\[}}-0.630731344
// FloatPerLayer-DAG: %[[c0:.*]] = stablehlo.constant dense<1.000000e+00
// FloatPerLayer-DAG: %[[matmul0_q:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]]) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_q]]) <{enabled = true, file_name = "quantized_tensor_data.pb", func_name = "composite_dot_general_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_fn_1", node_name = "_empty_node"}> : (tensor<1x3xf32>) -> ()
// FloatPerLayer-DAG: %[[matmul0_uq:.*]] = "tf.XlaCallModule"(%arg0, %[[w0]]) <{Sout = [#tf_type.shape<1x3>], module = "", version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1_0, _original_entry_function = "composite_dot_general_fn_1_0", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "1.0.0"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// FloatPerLayer-DAG: "tf.DumpTensor"(%[[matmul0_uq]]) <{enabled = true, file_name = "unquantized_tensor_data.pb", func_name = "composite_dot_general_fn_1", log_dir_path = "/tmp/dumps/composite_dot_general_fn_1", node_name = "_empty_node"}> : (tensor<1x3xf32>) -> ()
// FloatPerLayer-DAG: %[[concat:.*]] = stablehlo.concatenate %[[matmul0_uq]], %[[c0]], dim = 0 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
// FloatPerLayer-DAG: return %[[concat]] : tensor<2x3xf32>
// FloatPerLayer-DAG: func.func private @composite_dot_general_fn_1
// FloatPerLayer-DAG: func.func private @composite_dot_general_fn_1_0
}
