// RUN: tf-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// -----

func @enforce_static_shapes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{op operand #0 must be statically shaped tensor}}
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}

// -----

// CHECK-LABEL: func @add_tensors
func @add_tensors(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @add_scalars
func @add_scalars(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0: tensor<i32>
}

// -----

// CHECK-LABEL: func @add_scalar_tensor
func @add_scalar_tensor(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @batch_norm_inference
func @batch_norm_inference(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = "xla_hlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: func @broadcast
func @broadcast(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_nonint_sizes(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[1.0, 2.0]> : tensor<2xf64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_splat_sizes(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<2.0> : tensor<2xf64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_sparse_sizes(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_sizes_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes has rank 2 instead of rank 1}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[[1, 2]]> : tensor<1x2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_result_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result rank (3) does not match operand rank (1) plus size of broadcast_sizes (3)}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_first_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result has shape [1, 3] instead of [2, 3]}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func @broadcast_bad_second_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result has shape [2, 1] instead of [2, 3]}}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<2x1xi32>
  return %0 : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim
func @broadcast_in_dim(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  return %0 : tensor<1x2x2xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_zero_rank
func @broadcast_in_dim_zero_rank(%arg0: tensor<i32>) -> tensor<1x2x3xi32> {
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) : (tensor<i32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_nonint_dimensions(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1.0, 2.0]> : tensor<2xf64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_splat_dimensions(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<2.0> : tensor<2xf64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_sparse_dimensions(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_dimension_rank(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions has rank 2 instead of rank 1}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[[1,1],[1,1]]> : tensor<2x2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_dimension_size(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_rank_decrease(%arg0: tensor<1x2x3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{result rank (1) is less than operand rank (3)}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0,1,2]> : tensor<3xi64>} : (tensor<1x2x3xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func @broadcast_in_dim_dimension_values_too_large(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 9 for result result with rank 3}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[9, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_shape_mismatch(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{size of operand dimension 0 (3) is not equal to 1 or size of result dimension 1 (2)}}
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @comp_eq
func @comp_eq(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// -----

func @comp_bad_direction(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error@+1 {{'comparison_direction' failed to satisfy constraint}}
  %0 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "FOOBAR"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// -----

func @comp_no_direction(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error@+1 {{op requires attribute 'comparison_direction'}}
  %0 = "xla_hlo.compare"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// -----

// CHECK-LABEL: func @conv
func @conv(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "xla_hlo.conv"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  return %0: tensor<3xi32>
}

// -----

// CHECK-LABEL: func @copy
func @copy(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "xla_hlo.copy"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp
func @clamp(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "xla_hlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp_scalar
func @clamp_scalar(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "xla_hlo.clamp"(%arg1, %arg0, %arg1) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_min_element_type(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> tensor<1xi32> {
  // expected-error@+1 {{'xla_hlo.clamp' op requires the same element type for all operands and results}}
  %0 = "xla_hlo.clamp"(%arg1, %arg0, %arg0) : (tensor<1xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_min_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{min shape [2] is not scalar and does not match operand shape [1]}}
  %0 = "xla_hlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_max_element_type(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> tensor<1xi32> {
  // expected-error@+1 {{'xla_hlo.clamp' op requires the same element type for all operands and results}}
  %0 = "xla_hlo.clamp"(%arg0, %arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>, tensor<1xf32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_max_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{max shape [2] is not scalar and does not match operand shape [1]}}
  %0 = "xla_hlo.clamp"(%arg0, %arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @dot_vector
func @dot_vector(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) -> tensor<i32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<i32>
  return %0: tensor<i32>
}

// -----

// CHECK-LABEL: func @dot_matrix
func @dot_matrix(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @dot_precision_config
func @dot_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["HIGH", "HIGHEST"]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

func @dot_bad_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error@+1 {{'precision_config' failed to satisfy constraint}}
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["FOO", "HIGHEST"]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @tanh
func @tanh(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %0: tensor<1xf32>
}

// -----

func @exp_invalid_result_type(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error@+1 {{'xla_hlo.exp' op requires the same type for all operands and results}}
  %0 = "xla_hlo.exp"(%arg0) : (tensor<1xf32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @reshape_same_shape
func @reshape_same_shape(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @reshape_different_shape
func @reshape_different_shape(%arg0: tensor<1x16xi32>) -> tensor<4x4xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x16xi32>) -> tensor<4x4xi32>
  return %0: tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func @reshape_from_scalar
func @reshape_from_scalar(%arg0: tensor<i32>) -> tensor<1xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<i32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @reshape_to_scalar
func @reshape_to_scalar(%arg0: tensor<1xi32>) -> tensor<i32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1xi32>) -> tensor<i32>
  return %0: tensor<i32>
}

// -----

// CHECK-LABEL: func @select
func @select(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_pred
func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_pred_type(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{must be statically shaped tensor of pred (AKA boolean or 1-bit integer)}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_shape_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{on_true type (tensor<2x4xi32>) does not match on_false type (tensor<2x3xi32>)}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_element_type_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{on_true type (tensor<2x3xf32>) does not match on_false type (tensor<2x3xi32>)}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_pred_shape(%arg0: tensor<3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{red shape ([3]) is not scalar and does not match operand shapes ([2, 3])}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @slice
func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @slice_indices_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  // expected-error@+1 {{failed to verify that all of {start_indices, limit_indices} have same type}}
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 2, 3]> : tensor<3xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @slice_operand_result_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @transpose
func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_float(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1.0, 0.0, 3.0, 2.0]> : tensor<4xf64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_splat(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<2.0> : tensor<2xf64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_sparse(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation must be a DenseIntElementsAttr}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_rank(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation has rank 2 instead of rank 1}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[[1]]> : tensor<1x1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation size (1) does not match operand rank (4)}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1]> : tensor<1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2xi32> {
  // expected-error@+1 {{result rank (1) does not match operand rank (4)}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<1x2x3x4xi32> {
  // expected-error@+1 {{result shape is [1, 2, 3, 4] instead of [2, 1, 4, 3]}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32>
  return %0: tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: func @tuple
func @tuple(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>> {
  %0 = "xla_hlo.tuple"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>>
  return %0: tuple<tensor<1xi32>, tensor<1x2xf32>>
}
