// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file | mlir-hlo-opt | FileCheck %s

// Tests for types, ops with custom constraints, verifiers, printer or parser
// methods.

// CHECK-LABEL: func private @token_type() -> !mhlo.token
func private @token_type() -> !mhlo.token

// -----

// expected-error@+1 {{unknown mhlo type: foobar}}
func private @invalid_type() -> !mhlo.foobar

// -----

// CHECK-LABEL: func @alltoall
func @alltoall(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "mhlo.all_to_all"(%data) {
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
  %0 = "mhlo.all_to_all"(%data) {
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
  %0 = "mhlo.all_to_all"(%data) {
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
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_sizes_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes has rank 2 instead of rank 1}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[[1, 2]]> : tensor<1x2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_result_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result rank (3) does not match operand rank (1) plus size of broadcast_sizes (1)}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_bad_first_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result has shape [1, 3] instead of [2, 3]}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func @broadcast_bad_second_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{result has shape [2, 1] instead of [2, 3]}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<2x1xi32>
  return %0 : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim
func @broadcast_in_dim(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  return %0 : tensor<1x2x2xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_zero_rank
func @broadcast_in_dim_zero_rank(%arg0: tensor<i32>) -> tensor<1x2x3xi32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  return %0 : tensor<?x?x?xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_unknown_dim
func @dynamic_broadcast_in_dim_unknown_dim(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_ok_dim
func @dynamic_broadcast_in_dim_ok_dim(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  return %0 : tensor<7x8x9xf32>
}

// -----

func @dynamic_broadcast_in_dim_shape_mismatch(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{size of operand dimension 0 (32) is not compatible with size of result dimension 2 (9)}}
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  return %0 : tensor<7x8x9xf32>
}

// -----

func @broadcast_in_dim_bad_dimension_rank(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions has rank 2 instead of rank 1}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[[1,1],[1,1]]> : tensor<2x2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_dimension_size(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_rank_decrease(%arg0: tensor<1x2x3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{result rank (1) is less than operand rank (3)}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0,1,2]> : tensor<3xi64>} : (tensor<1x2x3xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func @broadcast_in_dim_dimension_values_too_large(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 9 for result with rank 3}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[9, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

func @broadcast_in_dim_bad_shape_mismatch(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{size of operand dimension 0 (3) is not equal to 1 or size of result dimension 1 (2)}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// -----

// Regression test for b/180052624, where this was improperly marked as an
// invalid mhlo.broadcast_in_dim op.
// CHECK-LABEL: func @broadcast_in_dim_dynamic_shaped_operand
func @broadcast_in_dim_dynamic_shaped_operand(%arg0 : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<?xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// Regression test for b/180052624, where this crashed verification given the
// unranked operand.
// CHECK-LABEL: func @broadcast_in_dim_unranked_operand
func @broadcast_in_dim_unranked_operand(%arg0 : tensor<*xf32>) -> tensor<2xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<*xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @if_nested_different_return_types(
func @if_nested_different_return_types(%pred : tensor<i1>, %branch_operand : tensor<f32>) {
  %0 = "mhlo.if"(%pred, %branch_operand, %branch_operand) ({
    ^bb0(%arg0 : tensor<f32>):
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
  }, {
    ^bb1(%arg1 : tensor<f32>):
      %2 = "mhlo.if"(%pred, %arg1, %arg1) ({
        ^bb0 (%arg2 : tensor<f32>):
          "mhlo.return"(%pred) : (tensor<i1>) -> ()
      }, {
        ^bb1 (%arg3 : tensor<f32>):
          "mhlo.return"(%pred) : (tensor<i1>) -> ()
      }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

func @if_mismatch_arg_type(%pred : tensor<i1>, %branch_operand : tensor<f32>, %wrong_type : tensor<3xf32>) {
  // @expected-error@+1 {{true_arg type ('tensor<3xf32>') does not match true_branch block arg type ('tensor<f32>')}}
  %0 = "mhlo.if"(%pred, %wrong_type, %branch_operand) ({
    ^bb0(%arg0 : tensor<f32>):
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
    }, {
    ^bb0(%arg1 : tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    }) : (tensor<i1>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

func @if_mismatch_return_type(%pred : tensor<i1>, %branch_operand : tensor<f32>, %wrong_type : tensor<3xf32>) {
  // @expected-error@+1 {{true_branch returned types ('tensor<3xf32>') do not match op result types ('tensor<f32>')}}
  %0 = "mhlo.if"(%pred, %branch_operand, %branch_operand) ({
    ^bb0(%arg0 : tensor<f32>):
      "mhlo.return"(%wrong_type) : (tensor<3xf32>) -> ()
    }, {
    ^bb0(%arg1 : tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

func @if_mismatch_num_return_types(%pred : tensor<i1>, %branch_operand : tensor<f32>) {
  // @expected-error@+1 {{true_branch returned types ('tensor<f32>', 'tensor<f32>') do not match op result types ('tensor<f32>')}}
  %0 = "mhlo.if"(%pred, %branch_operand, %branch_operand) ({
    ^bb0(%arg0 : tensor<f32>):
      "mhlo.return"(%branch_operand, %branch_operand) : (tensor<f32>, tensor<f32>) -> ()
    }, {
    ^bb0(%arg1 : tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// CHECK-LABEL: @case_nested_different_return_types(
func @case_nested_different_return_types(%index : tensor<i32>, %branch_operand : tensor<f32>) {
  %0 = "mhlo.case"(%index, %branch_operand, %branch_operand) ({
    ^bb0(%arg0 : tensor<f32>):
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
  }, {
    ^bb1(%arg1 : tensor<f32>):
      %2 = "mhlo.case"(%index, %arg1) ({
        ^bb0 (%arg2 : tensor<f32>):
          "mhlo.return"(%index) : (tensor<i32>) -> ()
      }) : (tensor<i32>, tensor<f32>) -> tensor<i32>
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

func @case_mismatch_num_args(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 1 block should have single argument, but found 2}}
  %0 = "mhlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %1 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_num_results(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 1 returned types ('tensor<f32>', 'tensor<f32>') do not match op result types ('tensor<f32>')}}
  %0 = "mhlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1, %arg0) : (tensor<f32>, tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_arg_type(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch_operand 1 type ('tensor<f32>') does not match branch 1 block arg type ('tensor<i32>')}}
  %0 = "mhlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<i32>):
      %1 = mhlo.constant dense<2.0> : tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @case_mismatch_return_type(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 1 returned types ('tensor<i32>') do not match op result types ('tensor<f32>')}}
  %0 = "mhlo.case"(%index, %operand_1, %operand_2, %operand_3) ( {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = mhlo.constant dense<2> : tensor<i32>
      "mhlo.return"(%1) : (tensor<i32>) -> ()
    },  {
    ^bb0(%arg0: tensor<f32>):
      %1 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @comp_eq
func @comp_eq(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// -----

func @comp_bad_direction(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error@+1 {{'comparison_direction' failed to satisfy constraint}}
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "FOOBAR"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %0 : tensor<3xi1>
}

// -----

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate sources not allowed}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [0, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_targets(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate targets not allowed}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 1]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute to be of rank 2, but got rank 1}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @collective_permute_duplicate_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute of shape (N, 2), but got (2, 3)}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}

// -----

func @concat_0D(%arg0: tensor<i32>, %arg1: tensor<i32>)  -> tensor<2xi32> {
  // expected-error@+1 {{rank-0 values cannot be concatenated}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @concat_1D
func @concat_1D(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @concat_1D
// Verifies that an error is not thrown if the inferred type is compatible with
// the result type.
func @concat_1D(%arg0: tensor<1xi32>, %arg1: tensor<*xi32>)  -> tensor<3xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<*xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

func @concat_1D_type_error(%arg0: tensor<1xi32>, %arg1: tensor<2xf32>)  -> tensor<3xi32> {
  // expected-error@+1 {{'mhlo.concatenate' op requires the same element type for all operands and results}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xf32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @concat_1D_unranked
func @concat_1D_unranked(%arg0: tensor<1xi32>, %arg1: tensor<*xi32>)  -> tensor<*xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

func @concat_1D_error(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<4xi32> {
  // expected-error@+1 {{op inferred type(s) 'tensor<3xi32>' are incompatible with return type(s) of operation 'tensor<4xi32>'}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @clamp
func @clamp(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp_scalar
func @clamp_scalar(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg1) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_clamp_element_type(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> tensor<1xi32> {
  // expected-error@+1 {{'mhlo.clamp' op requires the same element type for all operands and results}}
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg0) : (tensor<1xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

func @clamp_invalid_clamp_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{min shape [2] is not scalar and does not match operand shape [1]}}
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @dot_vector
func @dot_vector(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) -> tensor<i32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<i32>
  return %0: tensor<i32>
}

// -----

// CHECK-LABEL: func @dot_matrix
func @dot_matrix(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @dot_precision_config
func @dot_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["HIGH", "HIGHEST"]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

func @dot_bad_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error@+1 {{'precision_config' failed to satisfy constraint}}
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["FOO", "HIGHEST"]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0: tensor<2x2xi32>
}

// -----

func @infeed_invalid_number_of_results(%token: !mhlo.token) -> tuple<tuple<tensor<i32>>, !mhlo.token, tensor<i32>> {
  // expected-error@+1 {{result is expected to be a tuple of size 2, but got 3}}
  %0 = "mhlo.infeed"(%token) {infeed_config = "foobar", layout = [[[0]], unit, [0]]} : (!mhlo.token) -> tuple<tuple<tensor<i32>>, !mhlo.token, tensor<i32>>
  return %0 : tuple<tuple<tensor<i32>>, !mhlo.token, tensor<i32>>
}

// -----

func @infeed_non_token_second_result(%token: !mhlo.token) -> tuple<tuple<tensor<i32>>, tensor<i32>> {
  // expected-error@+1 {{second element of result tuple is expected to be of token type, but got 'tensor<i32>'}}
  %0 = "mhlo.infeed"(%token) {infeed_config = "foobar", layout = [[[0]], [0]]} : (!mhlo.token) -> tuple<tuple<tensor<i32>>, tensor<i32>>
  return %0 : tuple<tuple<tensor<i32>>, tensor<i32>>
}

// -----

func @iota_scalar() -> tensor<i32> {
  // expected-error@+1 {{does not support scalars}}
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @iota_invalid_iota_dimension() -> tensor<4xi32> {
  // expected-error@+1 {{iota dimension cannot go beyond the output rank or be negative}}
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

func @map_mismatched_args(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{expects number of operands to match the arity of map computation, but got: 2 and 1}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg: tensor<f32>):
    %1 = mhlo.add %arg, %arg {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func @map_non_scalar_computation_operand(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation arguments must be 0-rank tensor, but got: arg #1 of type 'tensor<5xf32>'}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<5xf32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_mismatch_operand_and_computation_args(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{element type of operands and computation arguments must match, but got: 'f32' and 'i32'}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_invalid_number_of_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return single output, but got: 0}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"() : () -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @main_non_scalar_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return 0-rank tensor, but got: 'tensor<5xf32>'}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<5xf32>
    "mhlo.return"(%1) : (tensor<5xf32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @mismatch_computation_output_type(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{element type of result and computation output must match, but got: 'f32' and 'i32'}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2> : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_invalid_dimension_numbers(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{requires monotonically increasing dimension numbers, but got: dense<[1, 0]> : tensor<2xi64>}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

func @map_mismatch_arguments_and_dimensions(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{applied to a subset of dimensions currently not supported: operand dimensions = 2, requested map dimensions size = 3}}
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func @map_unranked
func @map_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @recv_invalid_number_of_results(%token: !mhlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>, !mhlo.token> {
  // expected-error@+1 {{result is expected to be a tuple of size 2, but got 3}}
  %0 = "mhlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 3 : i64  // Host to device channel
    },
    is_host_transfer = true
  } : (!mhlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>, !mhlo.token>
  return %0 : tuple<tensor<3x4xi32>, tensor<i32>, !mhlo.token>
}

// -----

func @recv_non_token_second_result(%token: !mhlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{second element of result tuple is expected to be of token type, but got 'tensor<i32>'}}
  %0 = "mhlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 3 : i64  // Host to device channel
    },
    is_host_transfer = true
  } : (!mhlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>>
  return %0 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

// CHECK-LABEL: func @replica_id
func @replica_id() -> tensor<ui32> {
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  return %0 : tensor<ui32>
}

// -----

func @rng_uniform_invalid_type(%mu: tensor<complex<f32>>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{but got 'tensor<complex<f32>>'}}
  %0 = "mhlo.rng_uniform"(%mu, %sigma, %shape) : (tensor<complex<f32>>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @select
func @select(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_pred
func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<*xi32>, %arg2: tensor<2x3xi32>) -> tensor<*xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<*xi32>, tensor<2x3xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x?xi32>, %arg2: tensor<?x3xi32>) -> tensor<?x?xi32> {
  // TODO(lucyfox): Update once this is supported.
  // expected-error@+1 {{currently unsupported operand types: 'tensor<2x?xi32>' and 'tensor<?x3xi32>'}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x?xi32>, tensor<?x3xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_x_y
func @select_scalar_x_y(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @select_bad_pred_type(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{must be tensor of pred (AKA boolean or 1-bit integer) values}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_shape_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{incompatible operand types: 'tensor<2x4xi32>' and 'tensor<2x3xi32>'}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

func @select_bad_element_type_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{incompatible operand types: 'tensor<2x3xf32>' and 'tensor<2x3xi32>'}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @slice
func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// -----

func @slice_indices_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  // expected-error@+1 {{failed to verify that all of {start_indices, limit_indices, strides} have same type}}
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 2]> : tensor<2xi64>, limit_indices = dense<[2, 4, 1]> : tensor<3xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @slice_operand_result_mismatch(%arg0: tensor<3x4xi32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

func @slice_indices_not_rank_1(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{start_indices has rank 2 instead of required rank 1}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[[1, 0]]> : tensor<1x2xi64>,
    limit_indices = dense<[[2, 4]]> : tensor<1x2xi64>,
    strides = dense<[[1, 2]]> : tensor<1x2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// -----

func @slice_indices_wrong_size(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{the number of elements in start_indices (3) does not match the rank of the operand (2)}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0, 0]> : tensor<3xi64>,
    limit_indices = dense<[2, 4, 0]> : tensor<3xi64>,
    strides = dense<[1, 2, 0]> : tensor<3xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @dynamic_slice
func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "mhlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @dynamic_slice_mismatch_indices(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has mismatched number of slice sizes (1) and number of start indices (2)}}
  %0 = "mhlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[4]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_slice_different_indice_element_type
func @dynamic_slice_different_indice_element_type(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>) -> tensor<1x4xi32> {
  %0 = "mhlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<[4]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i32>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

func @dynamic_slice_mismatch_element_types(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to verify that all of {operand, result} have same element type}}
  %0 = "mhlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

func @dynamic_slice_invalid_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{operand #1 must be 0D tensor of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got 'tensor<2xi64>'}}
  %0 = "mhlo.dynamic-slice"(%arg0, %arg1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<2xi64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_update_slice
func @dynamic_update_slice(%input: tensor<3x4xi64>, %update: tensor<2xi64>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "mhlo.dynamic-update-slice"(%input, %update, %start1, %start2) : (tensor<3x4xi64>, tensor<2xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----

func @dynamic_update_slice_invalid_start(%input: tensor<3x4xi64>, %update: tensor<2xi64>, %start: tensor<2xi64>) -> tensor<3x4xi64> {
  // expected-error@+1 {{operand #2 must be 0D tensor of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got 'tensor<2xi64>'}}
  %0 = "mhlo.dynamic-update-slice"(%input, %update, %start) : (tensor<3x4xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x4xi64>
  return %0 : tensor<3x4xi64>
}

// -----

func @dynamic_update_slice_mismatched_start(%input: tensor<11x3x4xi32>, %update: tensor<1x3x4xi32>, %start1: tensor<i32>, %start2: tensor<i64>, %start3: tensor<i64>) -> tensor<11x3x4xi32> {
  // expected-error@+1 {{start indices must have same element type (encountered mismatch: 'i32' vs 'i64')}}
  %0 = "mhlo.dynamic-update-slice"(%input, %update, %start1, %start2, %start3) : (tensor<11x3x4xi32>, tensor<1x3x4xi32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<11x3x4xi32>
  return %0 : tensor<11x3x4xi32>
}

// -----

// CHECK-LABEL: func @transpose
func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_ranked(%arg0: tensor<?x?x?x?xi32>) ->  tensor<?x?x?x?xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  return %0: tensor<?x?x?x?xi32>
}

// -----

func @transpose_unranked(%arg0: tensor<*xi32>) ->  tensor<*xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<*xi32>) -> tensor<*xi32>
  return %0: tensor<*xi32>
}

// -----

func @transpose_bad_permutations_rank(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation has rank 2 instead of rank 1}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[[1]]> : tensor<1x1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{operand rank (4) does not match permutation size (1)}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1]> : tensor<1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0: tensor<2x1x4x3xi32>
}

// -----

func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2xi32> {
  // expected-error@+1 {{result rank (1) does not match permutation size (4)}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// -----

func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x?x3x?xi32>) ->  tensor<?x2x?x?xi32> {
  // expected-error@+1 {{result type tensor<?x2x?x?xi32> is incompatible with the expected type tensor<?x1x?x3xi32>}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32>
  return %0: tensor<?x2x?x?xi32>
}

// -----

func @triangular_solve_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @triangular_solve_rank_less_than_2(%arg0: tensor<4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operand 'a' must have rank >= 2, but got 'tensor<4xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_unequal_minor_dims_a(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{two minor dimensions of operand 'a' must have equal size, but got 'tensor<4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_unequal_rank(%arg0: tensor<10x4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operands must have equal rank, but got 'tensor<10x4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<10x4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

func @triangular_solve_mismatch_shared_dim(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error@+1 {{shared dimension of operands 'a' and 'b' does not match, but got 'tensor<4x4xf32>' and 'tensor<3x4xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

func @triangular_solve_mismatch_leading_dims(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32> {
  // expected-error@+1 {{leading batch dimensions of the operands must be same, but got 'tensor<10x5x4x4xf32>' and 'tensor<10x6x4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32>
  return %0 : tensor<10x6x4x3xf32>
}

// -----

func @triangular_solve_mismatch_result_and_b_type(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{result and operand 'b' must have same shape, but got 'tensor<4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @tuple
func @tuple(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>>
  return %0: tuple<tensor<1xi32>, tensor<1x2xf32>>
}

// -----

func @tuple_token(%arg0: tensor<f32>, %arg1: !mhlo.token) -> tuple<tensor<f32>, !mhlo.token> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, !mhlo.token) -> tuple<tensor<f32>, !mhlo.token>
  return %0 : tuple<tensor<f32>, !mhlo.token>
}

// -----

func @tuple_arg_size_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>> {
  // expected-error@+1 {{number of operands to tuple expected to match number of types}}
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>
  return %0 : tuple<tensor<f32>, tensor<f32>, tensor<f32>>
}

// -----

func @tuple_type_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<i32>> {
  // expected-error@+1 {{op has return type mismatch at 1th value}}
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<i32>>
  return %0 : tuple<tensor<f32>, tensor<i32>>
}

// -----

func @get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @get_tuple_element_token(%arg0: tuple<tensor<f32>, !mhlo.token>) -> !mhlo.token {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<f32>, !mhlo.token>) -> !mhlo.token
  return %0 : !mhlo.token
}

// -----

func @get_tuple_element_bad_type(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<i32> {
  // expected-error@+1 {{has return type tensor<i32>, but expected tensor<f32>}}
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func @get_tuple_element_index_out_of_bounds(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // expected-error@+1 {{index 2 is out of bounds of operand with size 2}}
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @and_i32_type
func @and_i32_type(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.and"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: func @or_i1_type
func @or_i1_type(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

func @or_invalid_f32_type(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{but got 'tensor<4xf32>'}}
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func @floor_invalid_i32_type(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{must be tensor of floating-point values, but got 'tensor<4xi32>'}}
  %0 = "mhlo.floor"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Verifiers HLO constant op custom printing and parsing.
// CHECK-LABEL: func @constants
func @constants() -> () {
  // CHECK: mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)

  // CHECK: mhlo.constant {extra_attr = 3 : i32} dense<0> : tensor<i32>
  %1 = "mhlo.constant"() {extra_attr = 3 : i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  return
}

// -----

func @constant_invalid() -> () {
  // expected-error@+1 {{op failed to verify that all of {value, output} have same type}}
  %0 = "mhlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<3xi32>)
  return
}

// -----

func @constant_invalid() -> () {
  // expected-error@+1 {{op result #0 must be statically shaped tensor}}
  %0 = "mhlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<?xi32>
  return
}

// -----

func @constant_invalid() -> () {
  // expected-error@+1 {{elements literal type must have static shape}}
  %0 = "mhlo.constant"() {value = dense<1> : tensor<?xi32>} : () -> tensor<?xi32>
  return
}

// -----

func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // CHECK: mhlo.sort
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_no_operands() {
  // expected-error @+1 {{expected named operation to have atleast 1 result}}
  %0:0 = "mhlo.sort"() ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : () -> ()
  return
}

// -----

func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block argument #0 should be of type 'tensor<f32>' but got 'tensor<i32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_different_dims(%input0: tensor<16x8xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op requires the same shape for all operands and results}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x8xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found 10}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 10 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found -3}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -3 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_wrong_block_arg_count(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op comparator block should have 4 arguments}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

func @sort_wrong_block_arg_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op comparator block argument #3 should be of type 'tensor<i32>' but got 'tensor<f32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}

// -----

// CHECK: func @dequantize
func @dequantize(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  %0 = "mhlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @dequantize_wrong_shape(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  // expected-error @+1 {{mismatched dimensions.}}
  %0 = "mhlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = true} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @dequantize_wrong_size(%arg: tensor<16x16xi32>) -> tensor<16x16xbf16> {
  // expected-error @+1 {{last dimension of output should be 4x of the input.}}
  %0 = "mhlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x16xbf16>
  return %0 : tensor<16x16xbf16>
}

// -----

func @dequantize_wrong_mode(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {
  // expected-error @+1 {{Dequantization mode. Only MIN_COMBINED is supported.}}
  %0 = "mhlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "hello", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// -----

func @reshape_invalid_shapes(%operand: tensor<2x4xf32>) -> tensor<3x3xf32> {
  // expected-error @+1 {{number of output elements (9) doesn't match expected number of elements (8)}}
  %0 = "mhlo.reshape"(%operand) : (tensor<2x4xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) { dot_dimension_numbers = {
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
  %0 = "mhlo.dot_general"(%arg0, %arg1) { dot_dimension_numbers = {
    lhs_batching_dimensions = dense<0> : tensor<1xi64>,
    lhs_contracting_dimensions = dense<[]> : tensor<0xi64>,
    rhs_batching_dimensions = dense<0> : tensor<1xi64>,
    rhs_contracting_dimensions = dense<1> : tensor<1xi64>
  }} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

func @compatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @incompatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?xf32> {
  // expected-error @+1 {{output should have a rank equal to the number of elements in output_shape}}
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func @cbrt(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.cbrt"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func @bitcast(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func @bitcast(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.reduce_precision"(%arg) {exponent_bits=2 : i32, mantissa_bits=3 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func @get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %size = "mhlo.get_dimension_size"(%I) {dimension = 3 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  return %size : tensor<i32>
}

// -----

func @get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  %size = "mhlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  return %size : tensor<i32>
}

// -----

func @set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<1xi32>

  // expected-error@+1 {{size operand should be of rank-0}}
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 2 : i64} : (tensor<1x128x512xf32>, tensor<1xi32>) -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}

// -----

func @set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<i32>

  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 3 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}

// -----

// CHECK: func @custom_call_multiple_outputs
func @custom_call_multiple_outputs(%x: tensor<2xf32>) -> tensor<2xf32> {
  %0:2 = "mhlo.custom_call"(%x) {backend_config="", call_target_name = "foo", has_side_effect = false} : (tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %1 = "mhlo.add"(%0#0, %0#1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----

// CHECK: func @reduce_window
func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>, %b1: tensor<i32>):  // no predecessors
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              %4 = "mhlo.tuple"(%2, %3) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
              "mhlo.return"(%4) : (tuple<tensor<f32>, tensor<i32>>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> } : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func @reduce_window_invalid(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xi32>, %init0: tensor<f32>, %init1: tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  // expected-error @+1 {{requires same shape for all inputs}}
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>, %b1: tensor<i32>):  // no predecessors
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              %4 = "mhlo.tuple"(%2, %3) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
              "mhlo.return"(%4) : (tuple<tensor<f32>, tensor<i32>>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> } : (tensor<4x2xf32>, tensor<4x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

func @rng_normal_invalid(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %cst = "mhlo.constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{tensor<7xf32>}}
  %0 = "mhlo.rng_normal"(%arg0, %arg1, %cst) : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<12xf32>
  return
}

// -----

func @rng_uniform_invalid(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<7xi64>) {
  // expected-error @+1 {{tensor<?x?x?x?x?x?x?xf32>}}
  %0 = "mhlo.rng_uniform"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<7xi64>) -> tensor<?xf32>
  return
}

// -----
// CHECK: func @conv2d_generic
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func @conv2d_generic(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers =
       {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>},
       feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  return %0 : tensor<1x8x8x16xf32>
}

// CHECK: func @conv2d
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func @conv2d(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = ["DEFAULT", "DEFAULT"]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  return %0 : tensor<1x8x8x16xf32>
}
