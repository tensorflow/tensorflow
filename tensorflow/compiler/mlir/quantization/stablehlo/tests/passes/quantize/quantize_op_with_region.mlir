// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize -verify-each=false | FileCheck %s

// Tests if reduce_window op following quantized function is quantized.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1722 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: main_00
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x3x1x1024xf32>
  func.func private @main_00(%arg0: tensor<2x3x1x1024xf32>) -> tensor<2x3x1x3xf32> attributes {tf._original_func_name = "main_0"} {
    // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
    // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    // CHECK: %[[Q0:.*]] = "quantfork.qcast"(%[[CST0]])
    // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[CST1]])
    // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[ARG0]])
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q2]], %[[Q1]])

    // CHECK: %[[REDUCE:.*]] = "stablehlo.reduce_window"(%[[CALL]], %[[Q0]])
    // CHECK: %[[ARG1:.*]]: tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>, %[[ARG2:.*]]: tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK: stablehlo.return %[[MAX]] : tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK{LITERAL}: padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
    // CHECK-SAME: window_dimensions = array<i64: 1, 3, 3, 1>
    // CHECK-SAME: (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>

    // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[REDUCE]])
    // CHECK: return %[[DQ]]

    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    %2 = "quantfork.qcast"(%0) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    %3 = "quantfork.dcast"(%2) : (tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<f32>
    %4 = "quantfork.qcast"(%1) {volatile} : (tensor<2x3x1024x3xf32>) -> tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>
    %5 = "quantfork.dcast"(%4) : (tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>) -> tensor<2x3x1024x3xf32>
    %6 = "quantfork.qcast"(%arg0) {volatile} : (tensor<2x3x1x1024xf32>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %7 = "quantfork.dcast"(%6) : (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024xf32>
    %8 = "tf.XlaCallModule"(%7, %5) <{Sout = [#tf_type.shape<2x3x1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    %9 = "quantfork.qcast"(%8) {volatile} : (tensor<2x3x1x3xf32>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %10 = "quantfork.dcast"(%9) : (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3xf32>
    %11 = "stablehlo.reduce_window"(%10, %3) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %14 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %14 : tensor<f32>
    }) {padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>} : (tensor<2x3x1x3xf32>, tensor<f32>) -> tensor<2x3x1x3xf32>
    %12 = "quantfork.qcast"(%11) {volatile} : (tensor<2x3x1x3xf32>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %13 = "quantfork.dcast"(%12) : (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3xf32>
    return %13 : tensor<2x3x1x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<2x3x1x1024xf32>, %arg1: tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general
    // CHECK: %[[RQ:.*]] = stablehlo.uniform_quantize %[[DOT]]
    // CHECK: return %[[RQ]]

    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    return %0 : tensor<2x3x1x3xf32>
  }
}

// -----

// Tests if reduce_window op preceding quantized function is quantized.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1722 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: main_00
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x3x1x1024xf32>
  func.func private @main_00(%arg0: tensor<2x3x1x1024xf32>) -> tensor<2x3x1x3xf32> attributes {tf._original_func_name = "main_0"} {
    // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
    // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    // CHECK: %[[Q0:.*]] = "quantfork.qcast"(%[[CST0]])
    // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[ARG0]])

    // CHECK: %[[REDUCE:.*]] = "stablehlo.reduce_window"(%[[Q1]], %[[Q0]])
    // CHECK: %[[ARG1:.*]]: tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>, %[[ARG2:.*]]: tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK: stablehlo.return %[[MAX]] : tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK{LITERAL}: padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
    // CHECK-SAME: window_dimensions = array<i64: 1, 3, 3, 1>
    // CHECK-SAME: (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>, tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>

    // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[CST1]])
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[REDUCE]], %[[Q2]])

    // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[CALL]])
    // CHECK: return %[[DQ]]

    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    %2 = "quantfork.qcast"(%0) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    %3 = "quantfork.dcast"(%2) : (tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<f32>
    %4 = "quantfork.qcast"(%arg0) {volatile} : (tensor<2x3x1x1024xf32>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %5 = "quantfork.dcast"(%4) : (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024xf32>
    %6 = "stablehlo.reduce_window"(%5, %3) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %14 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %14 : tensor<f32>
    }) {padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>} : (tensor<2x3x1x1024xf32>, tensor<f32>) -> tensor<2x3x1x1024xf32>
    %7 = "quantfork.qcast"(%6) {volatile} : (tensor<2x3x1x1024xf32>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %8 = "quantfork.dcast"(%7) : (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024xf32>
    %9 = "quantfork.qcast"(%1) {volatile} : (tensor<2x3x1024x3xf32>) -> tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>
    %10 = "quantfork.dcast"(%9) : (tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>) -> tensor<2x3x1024x3xf32>
    %11 = "tf.XlaCallModule"(%8, %10) <{Sout = [#tf_type.shape<2x3x1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    %12 = "quantfork.qcast"(%11) {volatile} : (tensor<2x3x1x3xf32>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %13 = "quantfork.dcast"(%12) : (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3xf32>
    return %13 : tensor<2x3x1x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<2x3x1x1024xf32>, %arg1: tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general
    // CHECK: %[[RQ:.*]] = stablehlo.uniform_quantize %[[DOT]]
    // CHECK: return %[[RQ]]

    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    return %0 : tensor<2x3x1x3xf32>
  }
}

// -----

// Tests if reduce_window op following quantized same-scale op is quantized.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1722 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: main_00
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x3x1x1024xf32>
  func.func private @main_00(%arg0: tensor<2x3x1x1024xf32>) -> tensor<2x3x3xf32> attributes {tf._original_func_name = "main_0"} {
    // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
    // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    // CHECK: %[[Q0:.*]] = "quantfork.qcast"(%[[CST0]])
    // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[CST1]])
    // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[ARG0]])
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q2]], %[[Q1]])
    // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[CALL]]

    // CHECK: %[[REDUCE:.*]] = "stablehlo.reduce_window"(%[[RESHAPE]], %[[Q0]])
    // CHECK: %[[ARG1:.*]]: tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>, %[[ARG2:.*]]: tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK: stablehlo.return %[[MAX]] : tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    // CHECK{LITERAL}: padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>
    // CHECK-SAME: window_dimensions = array<i64: 1, 3, 1>
    // CHECK-SAME: (tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>

    // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[REDUCE]])
    // CHECK: return %[[DQ]]

    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    %2 = "quantfork.qcast"(%0) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>
    %3 = "quantfork.dcast"(%2) : (tensor<!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<f32>
    %4 = "quantfork.qcast"(%1) {volatile} : (tensor<2x3x1024x3xf32>) -> tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>
    %5 = "quantfork.dcast"(%4) : (tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>) -> tensor<2x3x1024x3xf32>
    %6 = "quantfork.qcast"(%arg0) {volatile} : (tensor<2x3x1x1024xf32>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %7 = "quantfork.dcast"(%6) : (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024xf32>
    %8 = "tf.XlaCallModule"(%7, %5) <{Sout = [#tf_type.shape<2x3x1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    %9 = "quantfork.qcast"(%8) {volatile} : (tensor<2x3x1x3xf32>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %10 = "quantfork.dcast"(%9) : (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3xf32>
    %11 = stablehlo.reshape %10 : (tensor<2x3x1x3xf32>) -> tensor<2x3x3xf32>
    %12 = "quantfork.qcast"(%11) {volatile} : (tensor<2x3x3xf32>) -> tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %13 = "quantfork.dcast"(%12) : (tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x3xf32>
    %14 = "stablehlo.reduce_window"(%13, %3) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %17 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %17 : tensor<f32>
    }) {padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>, window_dimensions = array<i64: 1, 3, 1>} : (tensor<2x3x3xf32>, tensor<f32>) -> tensor<2x3x3xf32>
    %15 = "quantfork.qcast"(%14) {volatile} : (tensor<2x3x3xf32>) -> tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %16 = "quantfork.dcast"(%15) : (tensor<2x3x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x3xf32>
    return %16 : tensor<2x3x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<2x3x1x1024xf32>, %arg1: tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general
    // CHECK: %[[RQ:.*]] = stablehlo.uniform_quantize %[[DOT]]
    // CHECK: return %[[RQ]]

    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    return %0 : tensor<2x3x1x3xf32>
  }
}

// -----

// Tests if reduce_window op preceding quantized same-scale op is quantized.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1722 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: main_00
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x3x1024xf32>
  func.func private @main_00(%arg0: tensor<2x3x1024xf32>) -> tensor<2x3x1x3xf32> attributes {tf._original_func_name = "main_0"} {
    // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
    // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    // CHECK: %[[Q0:.*]] = "quantfork.qcast"(%[[CST0]])
    // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[ARG0]])

    // CHECK: %[[REDUCE:.*]] = "stablehlo.reduce_window"(%[[Q1]], %[[Q0]])
    // CHECK: %[[ARG1:.*]]: tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>, %[[ARG2:.*]]: tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ARG1]], %[[ARG2]] : tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK: stablehlo.return %[[MAX]] : tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    // CHECK{LITERAL}: padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>
    // CHECK-SAME: window_dimensions = array<i64: 1, 3, 1>
    // CHECK-SAME: (tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>, tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>

    // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[REDUCE]]
    // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[CST1]])
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[RESHAPE]], %[[Q2]])

    // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[CALL]])
    // CHECK: return %[[DQ]]

    %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.constant dense<0xFF80000E> : tensor<2x3x1024x3xf32>
    %2 = "quantfork.qcast"(%0) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>
    %3 = "quantfork.dcast"(%2) : (tensor<!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<f32>
    %4 = "quantfork.qcast"(%arg0) {volatile} : (tensor<2x3x1024xf32>) -> tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %5 = "quantfork.dcast"(%4) : (tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1024xf32>
    %6 = "stablehlo.reduce_window"(%5, %3) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %17 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %17 : tensor<f32>
    }) {padding = dense<[[0, 0], [1, 1], [0, 0]]> : tensor<3x2xi64>, window_dimensions = array<i64: 1, 3, 1>} : (tensor<2x3x1024xf32>, tensor<f32>) -> tensor<2x3x1024xf32>
    %7 = "quantfork.qcast"(%6) {volatile} : (tensor<2x3x1024xf32>) -> tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %8 = "quantfork.dcast"(%7) : (tensor<2x3x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1024xf32>
    %9 = stablehlo.reshape %8 : (tensor<2x3x1024xf32>) -> tensor<2x3x1x1024xf32>
    %10 = "quantfork.qcast"(%9) {volatile} : (tensor<2x3x1x1024xf32>) -> tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>
    %11 = "quantfork.dcast"(%10) : (tensor<2x3x1x1024x!quant.uniform<i8:f32, 5.000000e-01:2>>) -> tensor<2x3x1x1024xf32>
    %12 = "quantfork.qcast"(%1) {volatile} : (tensor<2x3x1024x3xf32>) -> tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>
    %13 = "quantfork.dcast"(%12) : (tensor<2x3x1024x3x!quant.uniform<i8<-127:127>:f32, 4.000000e-01>>) -> tensor<2x3x1024x3xf32>
    %14 = "tf.XlaCallModule"(%11, %13) <{Sout = [#tf_type.shape<2x3x1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    %15 = "quantfork.qcast"(%14) {volatile} : (tensor<2x3x1x3xf32>) -> tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>
    %16 = "quantfork.dcast"(%15) : (tensor<2x3x1x3x!quant.uniform<i8:f32, 3.000000e-01:1>>) -> tensor<2x3x1x3xf32>
    return %16 : tensor<2x3x1x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<2x3x1x1024xf32>, %arg1: tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general
    // CHECK: %[[RQ:.*]] = stablehlo.uniform_quantize %[[DOT]]
    // CHECK: return %[[RQ]]

    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<2x3x1x1024xf32>, tensor<2x3x1024x3xf32>) -> tensor<2x3x1x3xf32>
    return %0 : tensor<2x3x1x3xf32>
  }
}
