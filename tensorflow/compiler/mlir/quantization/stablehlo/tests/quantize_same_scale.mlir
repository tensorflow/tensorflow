// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize -verify-each=false | FileCheck %s

// CHECK-LABEL: same_scale_after_composite
func.func @same_scale_after_composite() -> tensor<3x1xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[RESHAPE:.*]] = "stablehlo.reshape"(%[[CALL]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[RESHAPE]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  return %5 : tensor<3x1xf32>
}

// -----

// CHECK-LABEL: same_scale_indirectly_connected
func.func @same_scale_indirectly_connected() -> tensor<1x3xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[RESHAPE:.*]] = "stablehlo.reshape"(%[[CALL]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[TRANSPOSE:.*]] = "stablehlo.transpose"(%[[RESHAPE]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[TRANSPOSE]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  %6 = "stablehlo.transpose"(%5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x3xf32>
  %7 = "quantfork.qcast"(%6) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %8 = "quantfork.dcast"(%7) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  return %8 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: same_scale_not_connected_to_composite
func.func @same_scale_not_connected_to_composite() -> tensor<3x1xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[CST]]) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ1:.*]] = "quantfork.dcast"(%[[Q1]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[DQ1]]
  // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[RESHAPE]]) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ2:.*]] = "quantfork.dcast"(%[[Q2]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  // CHECK: return %[[DQ2]]

  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  return %5 : tensor<3x1xf32>
}

// -----

// CHECK-LABEL: concatenate_and_composite
// CHECK: %[[ARG0:.*]]: tensor<3x2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x2xf32>
func.func @concatenate_and_composite(%arg0: tensor<3x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<4x5xf32> {
  // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[ARG0]]) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[ARG1]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[PAD:.*]] = "stablehlo.concatenate"(%[[Q1]], %[[Q2]]) {dimension = 0 : i64}
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>, tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"(%[[PAD]])
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[CALL]]) : (tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<4x5xf32>
  // CHECK:  "func.return"(%[[DQ]])

  %0 = "quantfork.qcast"(%arg0) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %1 = "quantfork.dcast"(%0) : (tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<3x2xf32>
  %2 = "quantfork.qcast"(%arg1) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %3 = "quantfork.dcast"(%2) : (tensor<1x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x2xf32>
  %4 = "stablehlo.concatenate"(%1, %3) {
    dimension = 0 : i64
  } : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
  %5 = "quantfork.qcast"(%4) {volatile} : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %6 = "quantfork.dcast"(%5) : (tensor<4x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<4x2xf32>
  %7 = "tf.XlaCallModule"(%6) {Sout = [#tf_type.shape<4x5>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<4x2xf32>) -> tensor<4x5xf32>
  %8 = "quantfork.qcast"(%7) {volatile} : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %9 = "quantfork.dcast"(%8) : (tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<4x5xf32>
  return %9 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: composite_and_convert
func.func @composite_and_convert() -> tensor<1x3xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[CONVERT:.*]] = "stablehlo.convert"(%[[CALL]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[CONVERT]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK:  "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.convert %2 : (tensor<1x3xf32>) -> (tensor<1x3xf32>)
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  return %5 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: pad_and_composite
// CHECK: %[[ARG0:.*]]: tensor<2x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<f32>
func.func @pad_and_composite(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<5x6xf32> {
  // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[ARG0]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[ARG1]]) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[PAD:.*]] = "stablehlo.pad"(%[[Q1]], %[[Q2]])
  // CHECK-SAME: {edge_padding_high = dense<[2, 1]> : tensor<2xi64>, edge_padding_low = dense<[0, 1]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>}
  // CHECK-SAME: (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>, tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"(%[[PAD]])
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: (tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<5x6x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[CALL]]) : (tensor<5x6x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<5x6xf32>
  // CHECK:  "func.return"(%[[DQ]])

  %0 = "quantfork.qcast"(%arg0) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %1 = "quantfork.dcast"(%0) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<2x3xf32>
  %2 = "quantfork.qcast"(%arg1) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %3 = "quantfork.dcast"(%2) : (tensor<!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<f32>
  %4 = "stablehlo.pad"(%1, %3) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[2, 1]> : tensor<2xi64>,
    interior_padding = dense<[1, 2]> : tensor<2xi64>
  }: (tensor<2x3xf32>, tensor<f32>) -> tensor<5x9xf32>
  %5 = "quantfork.qcast"(%4) {volatile} : (tensor<5x9xf32>) -> tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %6 = "quantfork.dcast"(%5) : (tensor<5x9x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<5x9xf32>
  %7 = "tf.XlaCallModule"(%6) {Sout = [#tf_type.shape<5x6>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<5x9xf32>) -> tensor<5x6xf32>
  %8 = "quantfork.qcast"(%7) {volatile} : (tensor<5x6xf32>) -> tensor<5x6x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %9 = "quantfork.dcast"(%8) : (tensor<5x6x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<5x6xf32>
  return %9 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: composite_and_select
// CHECK: %[[ARG0:.*]]: tensor<1x3xi1>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x3xf32>
func.func @composite_and_select(%arg0: tensor<1x3xi1>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[ARG1]]) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[SELECT:.*]] = "stablehlo.select"(%[[ARG0]], %[[CALL]], %[[Q1]]) : (tensor<1x3xi1>, tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%2) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK:  "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = "quantfork.qcast"(%arg1) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %4 = "quantfork.dcast"(%3) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %7 = stablehlo.select %arg0, %2, %4 : (tensor<1x3xi1>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %8 = "quantfork.qcast"(%7) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %9 = "quantfork.dcast"(%8) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  return %9 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: composite_and_broadcast_in_dim
func.func @composite_and_broadcast_in_dim() -> tensor<2x3x2xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[BROADCAST:.*]] = "stablehlo.broadcast_in_dim"(%[[CALL]])
  // CHECK-SAME: (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[BROADCAST]]) : (tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = "stablehlo.broadcast_in_dim"(%2) {
    broadcast_dimensions = dense<[2, 1]>: tensor<2xi64>
  } : (tensor<1x3xf32>) -> tensor<2x3x2xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<2x3x2xf32>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2xf32>
  return %5 : tensor<2x3x2xf32>
}

// -----

// CHECK-LABEL: composite_and_gather
// CHECK: %[[ARG0:.*]]: tensor<2x3x2xi64>
func.func @composite_and_gather(%arg0: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[CALL]], %[[ARG0]])
  // CHECK-SAME: (tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<2x3x2xi64>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[GATHER]]) : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x2xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<3x4x2>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<3x4x2xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<3x4x2xf32>) -> tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x4x2xf32>
  %3 = "stablehlo.gather"(%2, %arg0) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = dense<[1, 2, 2]> : tensor<3xi64>,
    indices_are_sorted = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x2xf32>
  return %5 : tensor<2x3x2x2xf32>
}

// -----

// CHECK-LABEL: composite_and_slice
func.func @composite_and_slice() -> tensor<2x2xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[SLICE:.*]] = "stablehlo.slice"(%[[CALL]])
  // CHECK-SAME: (tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[SLICE]]) : (tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<3x4>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<3x4xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x4xf32>
  %3 = "stablehlo.slice"(%2) {
    start_indices = dense<[1, 2]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xf32>) -> tensor<2x2xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2xf32>
  return %5 : tensor<2x2xf32>
}
