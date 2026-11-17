// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-opt -convert-tf-quant-types -quant-convert-tf-quant-ops-to-mhlo -split-input-file -verify-diagnostics %s

// Input/kernel spatial dimensions exceed rank - 2 (16 > 2) with SAME padding.
func.func @conv_hybrid_same_oob_spatial_dims(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{input spatial dimensions size must be rank - 2 (2), got 16.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    rhs_dilation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\10\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01 \02(\032\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Fully quantized UniformQuantizedConvolution with extra input_spatial_dimensions.
func.func @conv_same_oob_spatial_dims(%input: tensor<1x2x2x3x!tf_type.qint8>) -> tensor<1x2x1x2x!tf_type.qint32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %output_scales = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{input spatial dimensions size must be rank - 2 (2), got 16.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolution'}}
  %0 = "tf.UniformQuantizedConvolution"(
    %input, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      window_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      padding = "SAME",
      explicit_padding = [],
      lhs_dilation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      rhs_dilation = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      dimension_numbers = "\10\03\1A\10\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01 \02(\032\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\03J\02\01\02",
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64
  } : (
    tensor<1x2x2x3x!tf_type.qint8>, tensor<2x3x3x2x!tf_type.qint8>,
    tensor<f32>, tensor<i32>,
    tensor<f32>, tensor<i32>,
    tensor<f32>, tensor<i32>) -> tensor<1x2x1x2x!tf_type.qint32>
  func.return %0 : tensor<1x2x1x2x!tf_type.qint32>
}

// -----

// Out-of-range spatial dimension index (99 >= rank 4).
func.func @conv_hybrid_oob_spatial_dim_index(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{input dimension numbers must be a permutation of [0, 4), got invalid or duplicate dimension 99.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\63 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Duplicate input dimension number (batch=0, feature=0).
func.func @conv_hybrid_duplicate_dim_number(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{input dimension numbers must be a permutation of [0, 4), got invalid or duplicate dimension 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Empty dimension_numbers string on rank-4 tensor.
func.func @conv_hybrid_empty_dimension_numbers(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{input spatial dimensions size must be rank - 2 (2), got 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Kernel spatial dimensions size mismatch (1 != 2).
func.func @conv_hybrid_kernel_spatial_dims_mismatch(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{kernel spatial dimensions size must be rank - 2 (2), got 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\01\00@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Output spatial dimensions size mismatch (3 != 2).
func.func @conv_hybrid_output_spatial_dims_mismatch(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{output spatial dimensions size must be rank - 2 (2), got 3.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\03\01\02\00",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Operand rank < 3 (rank 1 previously produced negative padding_nums_size).
func.func @conv_hybrid_rank_less_than_3(%input: tensor<4xf32>, %weight: tensor<4x!tf_type.qint8>) -> tensor<4xf32> {
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{lhs and rhs rank must be at least 3, got 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [],
    rhs_dilation = [],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<4xf32>, tensor<4x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// Operand rank 2 violates op minimum rank of 3.
func.func @conv_hybrid_rank_2(%input: tensor<2x3xf32>, %weight: tensor<3x2x!tf_type.qint8>) -> tensor<2x2xf32> {
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{lhs and rhs rank must be at least 3, got 2.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [],
    rhs_dilation = [],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<2x3xf32>, tensor<3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// Mismatched lhs and rhs ranks (4 vs 3).
func.func @conv_hybrid_rank_mismatch(%input: tensor<1x2x2x3xf32>, %weight: tensor<2x3x3x!tf_type.qint8>) -> tensor<1x2x1x2xf32> {
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{lhs and rhs must have the same rank, got 4 and 3.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Unknown padding string.
func.func @conv_hybrid_unknown_padding(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{padding must be one of EXPLICIT, VALID, or SAME, got 'UNKNOWN'.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "UNKNOWN",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// EXPLICIT padding with mismatched/odd explicit_padding length.
func.func @conv_hybrid_explicit_padding_wrong_size(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{explicit_padding must have size 4, got 3.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "EXPLICIT",
    explicit_padding = [1, 2, 3],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// EXPLICIT padding with omitted optional explicit_padding attribute.
func.func @conv_hybrid_explicit_padding_missing(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{explicit_padding must have size 4, got 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "EXPLICIT",
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// EXPLICIT padding with negative explicit_padding element.
func.func @conv_hybrid_explicit_negative_padding(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{explicit_padding elements must be >= 0, got -10.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "EXPLICIT",
    explicit_padding = [-10, 0, 0, 0],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with short window_strides attribute.
func.func @conv_hybrid_same_short_window_strides(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{window_strides must have size 2, got 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with short lhs_dilation attribute.
func.func @conv_hybrid_same_short_lhs_dilation(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{lhs_dilation must have size 2, got 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with zero window stride (previously division by zero).
func.func @conv_hybrid_same_zero_stride(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{window_strides elements must be >= 1, got 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [0, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with non-positive lhs_dilation.
func.func @conv_hybrid_same_zero_lhs_dilation(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{lhs_dilation elements must be >= 1, got 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [0, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with negative rhs_dilation.
func.func @conv_hybrid_same_negative_rhs_dilation(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{rhs_dilation elements must be >= 1, got -1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, -1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with int64 dilated size overflow on rhs_dilation.
func.func @conv_hybrid_same_dilation_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{dilated size overflow for rhs_dilation.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [1, 4611686018427387904],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// VALID padding with int64 dilated size overflow on rhs_dilation.
func.func @conv_hybrid_valid_dilation_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{dilated size overflow for rhs_dilation.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [1, 4611686018427387904],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding where pad_low + lhs_dilated + pad_high overflows int64.
func.func @conv_hybrid_same_padded_lhs_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{padded lhs size overflow.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 1],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [4611686018427387904, 1],
    rhs_dilation = [9223372036854775806, 1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// EXPLICIT padding where pad_low + lhs_dilated + pad_high overflows int64.
func.func @conv_hybrid_explicit_padded_lhs_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{padded lhs size overflow.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 1],
    padding = "EXPLICIT",
    explicit_padding = [9223372036854775807, 0, 0, 0],
    lhs_dilation = [1, 1],
    rhs_dilation = [1, 1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with non-empty explicit_padding.
func.func @conv_hybrid_same_nonempty_explicit_padding(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{explicit_padding must be empty unless padding is EXPLICIT, got size 4.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [1, 2, 3, 4],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Non-positive feature_group_count.
func.func @conv_hybrid_zero_feature_group_count(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{feature_group_count and batch_group_count must be >= 1, got 0 and 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 0 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Out-of-range kernel spatial dimension index (99 >= rank 4).
func.func @conv_hybrid_oob_kernel_spatial_dim_index(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{kernel dimension numbers must be a permutation of [0, 4), got invalid or duplicate dimension 99.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\63@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// SAME padding with int64 dilated size overflow on lhs_dilation.
func.func @conv_hybrid_same_lhs_dilation_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{dilated size overflow for lhs_dilation.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 9223372036854775807],
    rhs_dilation = [1, 1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Non-positive batch_group_count.
func.func @conv_hybrid_zero_batch_group_count(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{feature_group_count and batch_group_count must be >= 1, got 1 and 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 0 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// EXPLICIT padding where high padding overflows pad_low + lhs_dilated + pad_high.
func.func @conv_hybrid_explicit_padded_lhs_high_overflow(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{padded lhs size overflow.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 1],
    padding = "EXPLICIT",
    explicit_padding = [0, 9223372036854775807, 0, 0],
    lhs_dilation = [1, 1],
    rhs_dilation = [1, 1],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Negative feature_group_count (verifies signed IntegerAttr::getInt() read).
func.func @conv_hybrid_negative_feature_group_count(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{feature_group_count and batch_group_count must be >= 1, got -1 and 1.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = -1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// Duplicate output batch dimension in output_spatial_dimensions.
func.func @conv_hybrid_duplicate_output_dim_number(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // expected-error @below {{output dimension numbers must be a permutation of [0, 4), got invalid or duplicate dimension 0.}}
  // expected-error @below {{failed to legalize operation 'tf.UniformQuantizedConvolutionHybrid'}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\00\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}
