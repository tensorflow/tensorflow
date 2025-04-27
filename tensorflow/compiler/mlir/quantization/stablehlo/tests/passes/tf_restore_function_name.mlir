// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-restore-function-name | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1646 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: @serving_default
  // CHECK-SAME: %[[ARG0:[^:[:space:]]+]]
  // CHECK-SAME: %[[ARG1:[^:[:space:]]+]]
  func.func private @serving_default(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x3xf32> {
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<1x3>], _entry_function = @main, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
    // CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: _entry_function = @composite_dot_general_fn_1
    // CHECK-SAME: _original_entry_function = "composite_dot_general_fn_1"
    // CHECK: return %[[CALL]]
  }

  // CHECK: @composite_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:[^:[:space:]]+]]
  // CHECK-SAME: %[[ARG3:[^:[:space:]]+]]
  func.func private @main(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
    // CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK: return %[[DOT]]
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1646 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: @serving_default
  // CHECK-SAME: %[[ARG0:[^:[:space:]]+]]
  // CHECK-SAME: %[[ARG1:[^:[:space:]]+]]
  func.func private @serving_default(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x3xf32> {
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<1x3>], _entry_function = @main, _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
    // CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: _entry_function = @main
    // CHECK-NOT: _original_entry_function = "composite_dot_general_fn_1"
    // CHECK: return %[[CALL]]
  }

  // CHECK: @main
  // CHECK-NOT: @composite_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:[^:[:space:]]+]]
  // CHECK-SAME: %[[ARG3:[^:[:space:]]+]]
  func.func private @main(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
    // CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK: return %[[DOT]]
  }
}
