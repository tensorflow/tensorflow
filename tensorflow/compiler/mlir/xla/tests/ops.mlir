// RUN: xla-opt %s -verify-diagnostics -split-input-file | xla-opt | FileCheck %s

// Tests for types, ops with custom constraints, verifiers, printer or parser
// methods.

// CHECK-LABEL: func @token_type() -> !xla_hlo.token
func @token_type() -> !xla_hlo.token

// -----

// expected-error@+1 {{unknown xla_hlo type: foobar}}
func @invalid_type() -> !xla_hlo.foobar

// -----

// CHECK-LABEL: func @alltoall
func @alltoall(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "xla_hlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  return %0 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: func @alltoall_unranked_input
func @alltoall_unranked_input(%data: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @alltoall_invalid_split_dim_size(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
// expected-error@+1 {{split dimension has size 16, expected to be a multiple of split_count 5}}
  %0 = "xla_hlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  return %0 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: func @broadcast
func @broadcast(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
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
  // expected-error@+1 {{result rank (3) does not match operand rank (1) plus size of broadcast_sizes (1)}}
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
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
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

func @case_mismatch_num_args(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{expects branch regions to have single argument, but found 2 for branch 1}}
  %0 = "xla_hlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = "xla_hlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_num_results(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 1 returned values do not match op result types}}
  %0 = "xla_hlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1, %arg0) : (tensor<f32>, tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_arg_type(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{expects operand 2 to be of type 'tensor<i32>', but found 'tensor<f32>'}}
  %0 = "xla_hlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<i32>):
      %1 = xla_hlo.constant dense<2.0> : tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_return_type(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 1 returned values do not match op result types}}
  %0 = "xla_hlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = xla_hlo.constant dense<2> : tensor<i32>
      "xla_hlo.return"(%1) : (tensor<i32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "xla_hlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "xla_hlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_empty_region(%index: tensor<i32>, %operand_1: tensor<f32>) -> () {
  // expected-error@+1 {{cannot have empty regions}}
  "xla_hlo.case"(%index, %operand_1) ( {} ) : (tensor<i32>, tensor<f32>) -> tensor<f32>
  return
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

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate sources not allowed}}
  %0 = "xla_hlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [0, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_targets(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate targets not allowed}}
  %0 = "xla_hlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 1]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute to be of rank 2, but got rank 1}}
  %0 = "xla_hlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute of shape (N, 2), but got (2, 3)}}
  %0 = "xla_hlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
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

func @clamp_invalid_clamp_element_type(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> tensor<1xi32> {
  // expected-error@+1 {{'xla_hlo.clamp' op requires the same element type for all operands and results}}
  %0 = "xla_hlo.clamp"(%arg1, %arg0, %arg0) : (tensor<1xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_clamp_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{min shape [2] is not scalar and does not match operand shape [1]}}
  %0 = "xla_hlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
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

func @infeed_invalid_number_of_results(%token: !xla_hlo.token) -> tuple<tuple<tensor<i32>>, !xla_hlo.token, tensor<i32>> {
  // expected-error@+1 {{result is expected to be a tuple of size 2, but got 3}}
  %0 = "xla_hlo.infeed"(%token) {infeed_config = "foobar"} : (!xla_hlo.token) -> tuple<tuple<tensor<i32>>, !xla_hlo.token, tensor<i32>>
  return %0 : tuple<tuple<tensor<i32>>, !xla_hlo.token, tensor<i32>>
}

// -----

func @infeed_non_token_second_result(%token: !xla_hlo.token) -> tuple<tuple<tensor<i32>>, tensor<i32>> {
  // expected-error@+1 {{second element of result tuple is expected to be of token type, but got 'tensor<i32>'}}
  %0 = "xla_hlo.infeed"(%token) {infeed_config = "foobar"} : (!xla_hlo.token) -> tuple<tuple<tensor<i32>>, tensor<i32>>
  return %0 : tuple<tuple<tensor<i32>>, tensor<i32>>
}

// -----

func @iota_scalar() -> tensor<i32> {
  // expected-error@+1 {{does not support scalars}}
  %0 = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @iota_invalid_iota_dimension() -> tensor<4xi32> {
  // expected-error@+1 {{iota dimension cannot go beyond the output rank or be negative}}
  %0 = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

func @map_mismatched_args(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{expects number of operands to match the arity of map computation, but got: 2 and 1}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg: tensor<f32>):
    %1 = xla_hlo.add %arg, %arg {name = "add"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func @map_non_scalar_computation_operand(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation arguments must be 0-rank tensor, but got: arg #1 of type 'tensor<5xf32>'}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<5xf32>):
    %1 = xla_hlo.constant {value = dense<2.0> : tensor<f32>} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_mismatch_operand_and_computation_args(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{element type of operands and computation arguments must match, but got: 'f32' and 'i32'}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = xla_hlo.constant {value = dense<2.0> : tensor<f32>} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_invalid_number_of_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return single output, but got: 0}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.constant {value = dense<2.0> : tensor<f32>} : tensor<f32>
    "xla_hlo.return"() : () -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @main_non_scalar_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return 0-rank tensor, but got: 'tensor<5xf32>'}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.constant {value = dense<2.0> : tensor<f32>} : tensor<5xf32>
    "xla_hlo.return"(%1) : (tensor<5xf32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @mismatch_computation_output_type(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{element type of result and computation output must match, but got: 'f32' and 'i32'}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.constant {value = dense<2> : tensor<i32>} : tensor<i32>
    "xla_hlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_invalid_dimension_numbers(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{requires monotonically increasing dimension numbers, but got: dense<[1, 0]> : tensor<2xi64>}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_mismatch_arguments_and_dimensions(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{applied to a subset of dimensions currently not supported: operand dimensions = 2, requested map dimensions size = 3}}
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func @map_unranked
func @map_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @recv_invalid_number_of_results(%token: !xla_hlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>, !xla_hlo.token> {
  // expected-error@+1 {{result is expected to be a tuple of size 2, but got 3}}
  %0 = "xla_hlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 3 : i64  // Host to device channel
    },
    is_host_transfer = true
  } : (!xla_hlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>, !xla_hlo.token>
  return %0 : tuple<tensor<3x4xi32>, tensor<i32>, !xla_hlo.token>
}

// -----

func @recv_non_token_second_result(%token: !xla_hlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{second element of result tuple is expected to be of token type, but got 'tensor<i32>'}}
  %0 = "xla_hlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 3 : i64  // Host to device channel
    },
    is_host_transfer = true
  } : (!xla_hlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>>
  return %0 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

func @rng_uniform_invalid_type(%mu: tensor<complex<f32>>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = xla_hlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{but got 'tensor<complex<f32>>'}}
  %0 = "xla_hlo.rng_uniform"(%mu, %sigma, %shape) : (tensor<complex<f32>>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: @scalars_to_dimension_tensor
func @scalars_to_dimension_tensor(%arg0: i32, %arg1: i32) -> tensor<2xi32> {
  %0 = "xla_hlo.scalars_to_dimension_tensor"(%arg0, %arg1) : (i32, i32) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @scalars_to_dimension_tensor_index
func @scalars_to_dimension_tensor_index(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = "xla_hlo.scalars_to_dimension_tensor"(%arg0, %arg1) : (index, index) -> tensor<2xindex>
  return %0 : tensor<2xindex>
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

// CHECK-LABEL: func @select_cast_compatible_types
func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<*xi32>, %arg2: tensor<2x3xi32>) -> tensor<*xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<*xi32>, tensor<2x3xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x?xi32>, %arg2: tensor<?x3xi32>) -> tensor<?x?xi32> {
  // TODO(lucyfox): Update once this is supported.
  // expected-error@+1 {{currently unsupported operand types: 'tensor<2x?xi32>' and 'tensor<?x3xi32>'}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x?xi32>, tensor<?x3xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_x_y
func @select_scalar_x_y(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @select_bad_pred_type(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{must be tensor of pred (AKA boolean or 1-bit integer) values}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_shape_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{incompatible operand types: 'tensor<2x4xi32>' and 'tensor<2x3xi32>'}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_element_type_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{incompatible operand types: 'tensor<2x3xf32>' and 'tensor<2x3xi32>'}}
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @slice
func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @slice_indices_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  // expected-error@+1 {{failed to verify that all of {start_indices, limit_indices, strides} have same type}}
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 2, 3]> : tensor<3xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @slice_operand_result_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "xla_hlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_slice
func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "xla_hlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @dynamic_slice_mismatch_indices(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has mismatched number of slice sizes (1) and number of start indices (2)}}
  %0 = "xla_hlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[4]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_slice_different_indice_element_type
func @dynamic_slice_different_indice_element_type(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>) -> tensor<1x4xi32> {
  %0 = "xla_hlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<[4]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @dynamic_slice_mismatch_element_types(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to verify that all of {operand, result} have same element type}}
  %0 = "xla_hlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

func @dynamic_slice_invalid_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{operand #1 must be a 0-dim integer tensor of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got 'tensor<2xi64>'}}
  %0 = "xla_hlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<2xi64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_update_slice
func @dynamic_update_slice(%input: tensor<3x4xi64>, %update: tensor<2xi64>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "xla_hlo.dynamic-update-slice"(%input, %update, %start1, %start2) : (tensor<3x4xi64>, tensor<2xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----

func @dynamic_update_slice_invalid_start(%input: tensor<3x4xi64>, %update: tensor<2xi64>, %start: tensor<2xi64>) -> tensor<3x4xi64> {
  // expected-error@+1 {{operand #2 must be a 0-dim integer tensor of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got 'tensor<2xi64>'}}
  %0 = "xla_hlo.dynamic-update-slice"(%input, %update, %start) : (tensor<3x4xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: func @transpose
func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_ranked(%arg0: tensor<?x?x?x?xi32>) ->  tensor<?x?x?x?xi32> {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0: tensor<?x?x?x?xi32>
}

// -----

func @transpose_unranked(%arg0: tensor<*xi32>) ->  tensor<*xi32> {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<*xi32>) -> tensor<*xi32>
  return %0: tensor<*xi32>
}

// -----

func @transpose_bad_permutations_rank(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation has rank 2 instead of rank 1}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[[1]]> : tensor<1x1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{operand rank (4) does not match permutation size (1)}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1]> : tensor<1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2xi32> {
  // expected-error@+1 {{result rank (1) does not match permutation size (4)}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x?x3x?xi32>) ->  tensor<?x2x?x?xi32> {
  // expected-error@+1 {{result type tensor<?x2x?x?xi32> is incompatible with the expected type tensor<?x1x?x3xi32>}}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32>
  return %0: tensor<?x2x?x?xi32>
}

// -----

func @triangular_solve_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @triangular_solve_rank_less_than_2(%arg0: tensor<4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operand 'a' must have rank >= 2, but got 'tensor<4xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_unequal_minor_dims_a(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{two minor dimensions of operand 'a' must have equal size, but got 'tensor<4x3xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_unequal_rank(%arg0: tensor<10x4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operands must have equal rank, but got 'tensor<10x4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<10x4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_mismatch_shared_dim(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error@+1 {{shared dimension of operands 'a' and 'b' does not match, but got 'tensor<4x4xf32>' and 'tensor<3x4xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

func @triangular_solve_mismatch_leading_dims(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32> {
  // expected-error@+1 {{leading batch dimensions of the operands must be same, but got 'tensor<10x5x4x4xf32>' and 'tensor<10x6x4x3xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32>
  return %0 : tensor<10x6x4x3xf32>
}

// -----

func @triangular_solve_mismatch_result_and_b_type(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{result and operand 'b' must have same shape, but got 'tensor<4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @tuple
func @tuple(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>> {
  %0 = "xla_hlo.tuple"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>>
  return %0: tuple<tensor<1xi32>, tensor<1x2xf32>>
}

// -----

func @tuple_arg_size_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>> {
  // expected-error@+1 {{has return type tuple<tensor<f32>, tensor<f32>, tensor<f32>>, but expected tuple<tensor<f32>, tensor<f32>>}}
  %0 = "xla_hlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>
  return %0 : tuple<tensor<f32>, tensor<f32>, tensor<f32>>
}

// -----

func @tuple_type_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<i32>> {
  // expected-error@+1 {{has return type tuple<tensor<f32>, tensor<i32>>, but expected tuple<tensor<f32>, tensor<f32>>}}
  %0 = "xla_hlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<i32>>
  return %0 : tuple<tensor<f32>, tensor<i32>>
}

// -----

func @get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @get_tuple_element_bad_type(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<i32> {
  // expected-error@+1 {{has return type tensor<i32>, but expected tensor<f32>}}
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @get_tuple_element_index_out_of_bounds(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // expected-error@+1 {{index 2 is out of bounds of operand with size 2}}
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @and_i32_type
func @and_i32_type(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "xla_hlo.and"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: func @or_i1_type
func @or_i1_type(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = "xla_hlo.or"(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

func @or_invalid_f32_type(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{but got 'tensor<4xf32>'}}
  %0 = "xla_hlo.or"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func @floor_invalid_i32_type(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{must be tensor of floating-point values, but got 'tensor<4xi32>'}}
  %0 = "xla_hlo.floor"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Verifiers HLO constant op custom printing and parsing.
// CHECK-LABEL: func @constants
func @constants() -> () {
  // CHECK: xla_hlo.constant dense<0> : tensor<i32>
  %0 = "xla_hlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)

  // CHECK: xla_hlo.constant {extra_attr = 3 : i32} dense<0> : tensor<i32>
  %1 = "xla_hlo.constant"() {extra_attr = 3 : i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  return
}

// -----

func @constant_invalid() -> () {
  // expected-error@+1 {{op failed to verify that all of {value, output} have same type}}
  %0 = "xla_hlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<*xi32>)
  return
}

// -----

func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // CHECK: xla_hlo.sort
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_no_operands() {
  // expected-error @+1 {{op requires at least one input}}
  %0 = "xla_hlo.sort"() ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : () -> tuple<>
  return
}

// -----

func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block argument #0 should be of type 'tensor<f32>' but got 'tensor<i32>'}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_different_dims(%input0: tensor<16x8xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op requires all inputs to have the same dimensions}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x8xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found 10}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 10 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found -3}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -3 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_wrong_block_arg_count(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op comparator block should have 4 arguments}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

func @sort_wrong_block_arg_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op comparator block argument #3 should be of type 'tensor<i32>' but got 'tensor<f32>'}}
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// -----

// CHECK: func @dequantize
func @dequantize(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @dequantize_wrong_shape(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  // expected-error @+1 {{mismatched dimensions.}}
  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = true} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @dequantize_wrong_size(%arg: tensor<16x16xi32>) -> tensor<16x16xbf16> {
  // expected-error @+1 {{last dimension of output should be 4x of the input.}}
  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x16xbf16>
  return %0 : tensor<16x16xbf16>
}

// -----

func @dequantize_wrong_mode(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  // expected-error @+1 {{Dequantization mode. Only MIN_COMBINED is supported.}}
  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "hello", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @reshape_invalid_shapes(%operand: tensor<2x4xf32>) -> tensor<3x3xf32> {
  // expected-error @+1 {{number of output elements (9) doesn't match expected number of elements (8)}}
  %0 = "xla_hlo.reshape"(%operand) : (tensor<2x4xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "xla_hlo.dot_general"(%arg0, %arg1) { dot_dimension_numbers = {
    lhs_batching_dimensions = dense<0> : tensor<1xi64>,
    lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
    rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    rhs_contracting_dimensions = dense<1> : tensor<1xi64>
  }} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of contracting dimensions}}
  %0 = "xla_hlo.dot_general"(%arg0, %arg1) { dot_dimension_numbers = {
    lhs_batching_dimensions = dense<0> : tensor<1xi64>,
    lhs_contracting_dimensions = dense<[]> : tensor<0xi64>,
    rhs_batching_dimensions = dense<0> : tensor<1xi64>,
    rhs_contracting_dimensions = dense<1> : tensor<1xi64>
  }} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

func @compatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "xla_hlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @incompatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?xf32> {
  // expected-error @+1 {{output should have a rank equal to the number of elements in output_shape}}
  %0 = "xla_hlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
