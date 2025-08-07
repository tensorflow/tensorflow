// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-quantize -verify-each=false | FileCheck %s

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: same_scale_after_composite
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
  func.func private @same_scale_after_composite(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x1xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[CALL]] : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[RESHAPE]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<1x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x3xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x3xf32>) -> tensor<3x1xf32>
    %8 = "quantization.qcast"(%7) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %9 = "quantization.dcast"(%8) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
    return %9 : tensor<3x1xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: same_scale_indirectly_connected
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
  func.func private @same_scale_indirectly_connected(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[CALL]] : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE]], dims = [1, 0] : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[TRANSPOSE]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    // CHECK: return %[[DQ]]

    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<1x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x3xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x3xf32>) -> tensor<3x1xf32>
    %8 = "quantization.qcast"(%7) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %9 = "quantization.dcast"(%8) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
    %10 = stablehlo.transpose %9, dims = [1, 0] : (tensor<3x1xf32>) -> tensor<1x3xf32>
    %11 = "quantization.qcast"(%10) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %12 = "quantization.dcast"(%11) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    return %12 : tensor<1x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

// CHECK-LABEL: same_scale_not_connected_to_composite
func.func @same_scale_not_connected_to_composite() -> tensor<3x1xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[CST]]) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ1:.*]] = "quantization.dcast"(%[[Q1]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[DQ1]]
  // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[RESHAPE]]) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ2:.*]] = "quantization.dcast"(%[[Q2]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  // CHECK: return %[[DQ2]]

  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x3xf32>
  %1 = "quantization.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantization.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantization.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantization.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  return %5 : tensor<3x1xf32>
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: concatenate_and_composite
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG2:.*]]: tensor<2x5xf32>
  func.func private @concatenate_and_composite(%arg0: tensor<3x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<2x5xf32>) -> tensor<4x5xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    // CHECK: %[[CONCAT:.*]] = stablehlo.concatenate %[[Q1]], %[[Q2]], dim = 0
    // CHECK-SAME: (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>, tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>) -> tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    // CHECK: %[[Q3:.*]] = "quantization.qcast"(%[[ARG2]]) {volatile} : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[CONCAT]], %[[Q3]])
    // CHECK-SAME: (tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>, tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[CALL]]) : (tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<4x5xf32>
    // CHECK: return %[[DQ]]

    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    %1 = "quantization.dcast"(%0) : (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>) -> tensor<3x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    %3 = "quantization.dcast"(%2) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>) -> tensor<1x2xf32>
    %4 = "stablehlo.concatenate"(%1, %3) {
      dimension = 0 : i64
    } : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<4x2xf32>) -> tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>) -> tensor<4x2xf32>
    %7 = "quantization.qcast"(%arg2) {volatile} : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
    %8 = "quantization.dcast"(%7) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x5xf32>
    %9 = "tf.XlaCallModule"(%6, %8) {Sout = [#tf_type.shape<4x5>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<4x2xf32>, tensor<2x5xf32>) -> tensor<4x5xf32>
    %10 = "quantization.qcast"(%9) {volatile} : (tensor<4x5xf32>) -> tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %11 = "quantization.dcast"(%10) : (tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<4x5xf32>
    return %11 : tensor<4x5xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG3:.*]]: tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>
  // CHECK-SAME: %[[ARG4:.*]]: tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<4x2xf32>, %arg1: tensor<2x5xf32>) -> tensor<4x5xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG3]], %[[ARG4]]
    // CHECK-SAME: (tensor<4x2x!quant.uniform<i8:f32, 5.000000e-03:-1>>, tensor<2x5x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<4x5x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<4x5x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<4x5x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x2xf32>, tensor<2x5xf32>) -> tensor<4x5xf32>
    return %0 : tensor<4x5xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: composite_and_pad
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
  // CHECK-SAME: %[[ARG2:.*]]: tensor<f32>
  func.func private @composite_and_pad(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<f32>) -> tensor<3x9xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[Q3:.*]] = "quantization.qcast"(%arg2) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[PAD:.*]] = stablehlo.pad %[[CALL]], %[[Q3]]
    // CHECK-SAME: (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x9x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[PAD]]) : (tensor<3x9x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x9xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<1x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x3xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %7 = "quantization.qcast"(%arg2) {volatile} : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %8 = "quantization.dcast"(%7) : (tensor<!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<f32>
    %9 = stablehlo.pad %6, %8, low = [0, 1], high = [2, 1], interior = [0, 2] : (tensor<1x3xf32>, tensor<f32>) -> tensor<3x9xf32>
    %10 = "quantization.qcast"(%9) {volatile} : (tensor<3x9xf32>) -> tensor<3x9x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %11 = "quantization.dcast"(%10) : (tensor<3x9x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x9xf32>
    return %11 : tensor<3x9xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: composite_and_select
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x3xi1>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<1x3xf32>
  func.func private @composite_and_select(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<1x3xi1>, %arg3: tensor<1x3xf32>) -> tensor<1x3xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[Q3:.*]] = "quantization.qcast"(%[[ARG3]]) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[SELECT:.*]] = stablehlo.select %[[ARG2]], %[[CALL]], %[[Q3]] : tensor<1x3xi1>, tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[SELECT]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<1x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x3xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %7 = "quantization.qcast"(%arg3) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %8 = "quantization.dcast"(%7) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %9 = stablehlo.select %arg2, %6, %8 : (tensor<1x3xi1>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %10 = "quantization.qcast"(%9) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %11 = "quantization.dcast"(%10) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    return %11 : tensor<1x3xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: composite_and_broadcast_in_dim
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
  func.func private @composite_and_broadcast_in_dim(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3x2xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CALL]], dims = [2, 1] : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[BROADCAST]]) : (tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<1x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x3xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [2, 1] : (tensor<1x3xf32>) -> tensor<2x3x2xf32>
    %8 = "quantization.qcast"(%7) {volatile} : (tensor<2x3x2xf32>) -> tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %9 = "quantization.dcast"(%8) : (tensor<2x3x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2xf32>
    return %9 : tensor<2x3x2xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<1x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: composite_and_gather
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x4x5xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<3x5x2xf32>
  // CHECK-SAME: %[[ARG2:.*]]: tensor<2x3x2xi64>
  func.func private @composite_and_gather(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x2xf32>, %arg2: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<3x4x5xf32>) -> tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<3x5x2xf32>) -> tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: (tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>) -> tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%[[CALL]], %[[ARG2]])
    // CHECK-SAME: (tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>, tensor<2x3x2xi64>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[GATHER]]) : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x2xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<3x4x5xf32>) -> tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<3x4x5xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<3x5x2xf32>) -> tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>
    %3 = "quantization.dcast"(%2) : (tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>) -> tensor<3x5x2xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<3x4x5xf32>, tensor<3x5x2xf32>) -> tensor<3x4x2xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<3x4x2xf32>) -> tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x4x2xf32>
    %7 = "stablehlo.gather"(%6, %arg2) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
    %8 = "quantization.qcast"(%7) {volatile} : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %9 = "quantization.dcast"(%8) : (tensor<2x3x2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x3x2x2xf32>
    return %9 : tensor<2x3x2x2xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x2xf32>) -> tensor<3x4x2xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<3x4x5x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<3x5x2x!quant.uniform<i8<-127:127>:f32, 6.000000e-03>>) -> tensor<3x4x2x!quant.uniform<i32:f32, 3.000000e-05>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<3x4x2x!quant.uniform<i32:f32, 3.000000e-05>>) -> tensor<3x4x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x4x5xf32>, tensor<3x5x2xf32>) -> tensor<3x4x2xf32>
    return %0 : tensor<3x4x2xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: composite_and_slice
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<2x4xf32>
  func.func private @composite_and_slice(%arg0: tensor<3x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x2xf32> {
    // CHECK: %[[Q1:.*]] = "quantization.qcast"(%[[ARG0]]) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>
    // CHECK: %[[Q2:.*]] = "quantization.qcast"(%[[ARG1]]) {volatile} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
    // CHECK: %[[CALL:.*]] = call @quantized_dot_general_fn_1(%[[Q1]], %[[Q2]])
    // CHECK-SAME: (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[CALL]] [1:3, 2:4] : (tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: %[[DQ:.*]] = "quantization.dcast"(%[[SLICE]]) : (tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2xf32>
    // CHECK: return %[[DQ]]
    %0 = "quantization.qcast"(%arg0) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>
    %1 = "quantization.dcast"(%0) : (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>) -> tensor<3x2xf32>
    %2 = "quantization.qcast"(%arg1) {volatile} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
    %3 = "quantization.dcast"(%2) : (tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<2x4xf32>
    %4 = "tf.XlaCallModule"(%1, %3) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    %5 = "quantization.qcast"(%4) {volatile} : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %6 = "quantization.dcast"(%5) : (tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x4xf32>
    %7 = stablehlo.slice %6 [1:3, 2:4] : (tensor<3x4xf32>) -> tensor<2x2xf32>
    %8 = "quantization.qcast"(%7) {volatile} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    %9 = "quantization.dcast"(%8) : (tensor<2x2x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<2x2xf32>
    return %9 : tensor<2x2xf32>
  }

  // CHECK: quantized_dot_general_fn_1
  // CHECK-SAME: %[[ARG2:.*]]: tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>
  // CHECK-SAME: %[[ARG3:.*]]: tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<3x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<3x4xf32> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[ARG2]], %[[ARG3]]
    // CHECK-SAME: (tensor<3x2x!quant.uniform<i8:f32, 5.000000e-03>>, tensor<2x4x!quant.uniform<i8<-127:127>:f32:1, {6.000000e-03,6.000000e-03,6.000000e-03,6.000000e-03}>>) -> tensor<3x4x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05}>>
    // CHECK: %[[Q3:.*]] = stablehlo.uniform_quantize %0 : (tensor<3x4x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<3x4x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
    // CHECK: return %[[Q3]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
