// RUN: mlir-hlo-opt --mhlo-test-infer-shaped-type-methods --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @compare
// CHECK-SAME: (%[[A:.*]]: tensor<2x?xf32>,
func.func @compare(%a : tensor<2x?xf32>, %b : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[A]] : tensor<2x?xf32> -> tensor<2xindex>
  // CHECK: return %[[SHAPE]] : tensor<2xindex>
  %0 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<comparison_direction NE>}
      : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xi1>
  %1 = "mhlo_test.reify_return_type_shapes"(%0)
      : (tensor<2x?xi1>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @select
func.func @select(%pred : tensor<i1>, %a : tensor<?x2x3xf32>, %b : tensor<1x?x3xf32>)
    -> tensor<1x2x3xindex> {
  %0 = "mhlo.select"(%pred, %a, %b)
      : (tensor<i1>, tensor<?x2x3xf32>, tensor<1x?x3xf32>) -> tensor<?x?x?xf32>
  // CHECK: types0 = tensor<1x2x3xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf32>) -> tensor<1x2x3xindex>
  func.return %1 : tensor<1x2x3xindex>
}

// -----

// CHECK-LABEL: @compare
func.func @compare(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xindex> {
  %0 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<comparison_direction NE>}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK: types0 = tensor<2x2xi1>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<2x2xi1>) -> tensor<2x2xindex>
  func.return %1 : tensor<2x2xindex>
}

// -----

// CHECK-LABEL: @broadcast
func.func @broadcast(%a : tensor<3xi32>) -> tensor<1x2x3xindex> {
  %0 = "mhlo.broadcast"(%a) <{broadcast_sizes = dense<[1, 2]> : tensor<2xi64>}>
      : (tensor<3xi32>) -> tensor<1x2x3xi32>
  // CHECK: types0 = tensor<1x2x3xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1x2x3xi32>) -> tensor<1x2x3xindex>
  func.return %1 : tensor<1x2x3xindex>
}

// -----

func.func @broadcast(%a : tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+2 {{'mhlo.broadcast' op failed to infer returned types}}
  // expected-error@+1 {{Broadcast with negative dimension size -2}}
  %0 = "mhlo.broadcast"(%a) <{broadcast_sizes = dense<[1, -2]> : tensor<2xi64>}>
      : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xindex> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: types0 = tensor<1x4xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1x4xi32>) -> tensor<1x4xindex>
  func.return %1 : tensor<1x4xindex>
}

// -----

// CHECK-LABEL: @pad
func.func @pad(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xindex> {
  %0 = "mhlo.pad"(%arg0, %arg1) <{
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  }> : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  // CHECK: types0 = tensor<2x4x7xf16>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<2x4x7xf16>) -> tensor<2x4x7xindex>
  func.return %1 : tensor<2x4x7xindex>
}

// -----

// CHECK-LABEL: @pad_with_bounds
func.func @pad_with_bounds(%arg0: tensor<3x?x?xf16, #mhlo.type_extensions<bounds = [?, 3, ?]>>, %arg1: tensor<f16>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[2, 2, 0]> : tensor<3xi64>,
    edge_padding_high = dense<[0, 0, 0]> : tensor<3xi64>,
    interior_padding = dense<[1, 1, 1]> : tensor<3xi64>
  } : (tensor<3x?x?xf16, #mhlo.type_extensions<bounds = [?, 3, ?]>>, tensor<f16>) -> tensor<?x?x?xf16>
  // CHECK: types0 = tensor<7x?x?xf16, #mhlo.type_extensions<bounds = [?, 7, ?]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf16>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

func.func @pad_with_negative_inferred_bounds(%arg0: tensor<3x?x?xf16, #mhlo.type_extensions<bounds = [?, 3, ?]>>, %arg1: tensor<f16>) -> tensor<?x?x?xindex> {
  // expected-error@+2 {{'mhlo.pad' op failed to infer returned types}}
  // expected-error@+1 {{Padding result in negative bound for dimension 1}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[2, -10, 0]> : tensor<3xi64>,
    edge_padding_high = dense<[0, 0, 0]> : tensor<3xi64>,
    interior_padding = dense<[1, 1, 1]> : tensor<3xi64>
  } : (tensor<3x?x?xf16, #mhlo.type_extensions<bounds = [?, 3, ?]>>, tensor<f16>) -> tensor<?x?x?xf16>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf16>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: @cholesky
func.func @cholesky(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2xindex> {
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  // CHECK: types0 = tensor<1x2x2xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1x2x2xf32>) -> tensor<1x2x2xindex>
  func.return %1: tensor<1x2x2xindex>
}

// -----

// CHECK-LABEL: func @alltoall
func.func @alltoall(%data: tensor<4x16xf32>) -> tensor<16x4xindex> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  // CHECK: types0 = tensor<16x4xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<16x4xf32>) -> tensor<16x4xindex>
  func.return %1 : tensor<16x4xindex>
}

// -----

// CHECK-LABEL: func @alltoall_bounds
func.func @alltoall_bounds(%data: tensor<16x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>) -> tensor<?x?xindex> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 0 : i64,
    concat_dimension = 1 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<16x?xf32, #mhlo.type_extensions<bounds = [?, 5]>>) -> tensor<?x?xf32>
  // CHECK: types0 = tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 20]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: func @abs
func.func @abs(%arg0: tensor<1x2xf32>) -> tensor<1x2xindex> {
  %0 = "mhlo.abs"(%arg0) {} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: types0 = tensor<1x2xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1x2xf32>) -> tensor<1x2xindex>
  func.return %1: tensor<1x2xindex>
}

// -----

// CHECK-LABEL: @concat
func.func @concat(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<3xindex> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  // CHECK: types0 = tensor<3xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<3xi32>) -> tensor<3xindex>
  func.return %1 : tensor<3xindex>
}

// -----

// -----

// Inference rules to concat dimensions with bounds (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              Y               X+Y
//  c1:  X              ?               ?
//  c2:  X              ?, B            ?, X+B
//  c3:  ?              ?               ?
//  c4:  ?              ?, B            ?
//  c5:  ?, B           ?, C            ?, B+C

// CHECK-LABEL: @concat_bounds_c0
func.func @concat_bounds_c0(
  %arg0: tensor<5x1xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
  %arg1: tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x1xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x3xi32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @concat_bounds_c1
func.func @concat_bounds_c1(
  %arg0: tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
  %arg1: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  %result_swap = "mhlo.concatenate"(%arg1, %arg0) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32>
  %2 = "mhlo_test.get_return_types"(%result_swap) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @concat_bounds_c2
func.func @concat_bounds_c2(
  %arg0: tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
  %arg1: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 6]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  %result_swap = "mhlo.concatenate"(%arg1, %arg0) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>,
    tensor<5x2xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 6]>>
  %2 = "mhlo_test.get_return_types"(%result_swap) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @concat_bounds_c3
func.func @concat_bounds_c3(
  %arg0: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
  %arg1: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @concat_bounds_c4
func.func @concat_bounds_c4(
  %arg0: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
  %arg1: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  %result_swap = "mhlo.concatenate"(%arg1, %arg0) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, ?]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32>
  %2 = "mhlo_test.get_return_types"(%result_swap) : (tensor<?x?xi32>) -> tensor<?x?xindex>

  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @concat_bounds_c5
func.func @concat_bounds_c5(
  %arg0: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 3]>>,
  %arg1: tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xindex> {
  %result = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 1 : i64 }> : (
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 3]>>,
    tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<?x?xi32>
  // CHECK: types0 = tensor<5x?xi32, #mhlo.type_extensions<bounds = [?, 7]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xi32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// This test covers all cases (except "error out") of inferBranchedDimAndBound()
// CHECK-LABEL: func @if_bounds
func.func @if_bounds(%pred : tensor<i1>,
    %true_branch_operand : tensor<2x3x4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 6]>>,
    %false_branch_operand : tensor<2x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 5, 7]>>) -> tensor<?x?x?x?x?x?xindex> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%true_branch_operand) : (
        tensor<2x3x4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 6]>>) -> ()
    }, {
      "mhlo.return"(%false_branch_operand) : (
        tensor<2x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 5, 7]>>) -> ()
    }) : (tensor<i1>) -> tensor<?x?x?x?x?x?xf32>
  // CHECK: types0 = tensor<2x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, ?, 7]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?x?x?xindex>
}

// -----

func.func @if_bounds_unranked(%pred : tensor<i1>,
    %true_branch_operand : tensor<2x3x4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 6]>>,
    %false_branch_operand : tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xindex> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%true_branch_operand) : (
        tensor<2x3x4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 6]>>) -> ()
    }, {
      "mhlo.return"(%false_branch_operand) : (
        tensor<?x?x?x?x?x?xf32>) -> ()
    }) : (tensor<i1>) -> tensor<?x?x?x?x?x?xf32>
  // CHECK: types0 = tensor<?x?x?x?x?x?xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?x?x?xindex>
}

// -----

// This test covers only a few cases of inferBranchedDimAndBound() with more branches
// as test "if_bounds" above covers all cases
// CHECK-LABEL: func @case_bounds
func.func @case_bounds(%index : tensor<i32>,
    %branch_0_operand : tensor<2xf32, #mhlo.type_extensions<bounds = [?]>>,
    %branch_2_operand : tensor<?xf32, #mhlo.type_extensions<bounds = [3]>>) -> tensor<?xindex> {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_0_operand) : (tensor<2xf32, #mhlo.type_extensions<bounds = [?]>>) -> ()
  }, {
      "mhlo.return"(%branch_0_operand) : (tensor<2xf32, #mhlo.type_extensions<bounds = [?]>>) -> ()
  }, {
      "mhlo.return"(%branch_2_operand) : (tensor<?xf32, #mhlo.type_extensions<bounds = [3]>>) -> ()
  }) : (tensor<i32>) -> tensor<?xf32>
  // CHECK: types0 = tensor<?xf32, #mhlo.type_extensions<bounds = [3]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?xf32>) -> tensor<?xindex>
  func.return %1 : tensor<?xindex>
}

// -----

// CHECK-LABEL: while_bounds
func.func @while_bounds(
  %while_arg_1: tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>,
  %while_arg_2: tensor<3xf32>) -> tensor<?xindex> {
  %1:2 = "mhlo.while"(%while_arg_1, %while_arg_2) ({
  ^bb0(%arg1: tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>, %arg2: tensor<3xf32>):
    %2 = mhlo.constant dense<1> : tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>, %arg2: tensor<3xf32>):
    "mhlo.return"(%arg1, %arg2) : (tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>, tensor<3xf32>) -> ()
  }) : (tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>, tensor<3xf32>) -> (tensor<?x?xi32>, tensor<?xf32>)
  // CHECK: types0 = tensor<2x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>,
  // CHECK-SAME: types1 = tensor<3xf32>
  %3 = "mhlo_test.get_return_types"(%1) : (tensor<?x?xi32>) -> tensor<?xindex>
  func.return %3 : tensor<?xindex>
}

// -----

// CHECK-LABEL: @gather
func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xindex> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  // CHECK: types0 = tensor<1x5x8xi32>
  %1 = "mhlo_test.get_return_types"(%res) : (tensor<1x5x8xi32>) -> tensor<1x5x8xindex>
  func.return %1 : tensor<1x5x8xindex>
}

// -----

// CHECK-LABEL: @gather_bounds
func.func @gather_bounds(%operand : tensor<?x?x?xi32, #mhlo.type_extensions<bounds = [2, 4, 8]>>,
    %start_indices : tensor<?x?x?xi32, #mhlo.type_extensions<bounds = [16, 32, 64]>>)
    -> tensor<?x?x?xindex> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 0,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<?x?x?xi32, #mhlo.type_extensions<bounds = [2, 4, 8]>>, tensor<?x?x?xi32, #mhlo.type_extensions<bounds = [16, 32, 64]>>)
  -> tensor<?x?x8xi32>

  // CHECK: types0 = tensor<?x?x8xi32, #mhlo.type_extensions<bounds = [32, 64, ?]>>
  %1 = "mhlo_test.get_return_types"(%res) : (tensor<?x?x8xi32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: @rng_normal
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<7xindex> {
  %0 = "mhlo.constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "mhlo.rng"(%arg0, %arg1, %0) <{rng_distribution = #mhlo.rng_distribution<NORMAL>}> : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<7xf32>
  // CHECK: types0 = tensor<7xf32>
  %2 = "mhlo_test.get_return_types"(%1) : (tensor<7xf32>) -> tensor<7xindex>
  func.return %2 : tensor<7xindex>
}

// -----

// CHECK-LABEL: func @rng_uniform
func.func @rng_uniform(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xindex> {
  %0 = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %1 = "mhlo.rng"(%a, %b, %0) <{rng_distribution = #mhlo.rng_distribution<UNIFORM>}> : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  // CHECK: types0 = tensor<2x3x5xf32>
  %2 = "mhlo_test.get_return_types"(%1) : (tensor<2x3x5xf32>) -> tensor<2x3x5xindex>
  func.return %2 : tensor<2x3x5xindex>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xindex> {
  %0 = "mhlo.slice"(%arg0) <{start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}> : (tensor<3x4xi32>) -> tensor<1x2xi32>
  // CHECK: types0 = tensor<1x2xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1x2xi32>) -> tensor<1x2xindex>
  func.return %1 : tensor<1x2xindex>
}

// -----

// CHECK-LABEL: func @slice_with_bounds
func.func @slice_with_bounds(%arg0: tensor<3x?x?xi32, #mhlo.type_extensions<bounds = [?, 4, ?]>>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.slice"(%arg0) <{start_indices = dense<[1, 0, 0]> : tensor<3xi64>, limit_indices = dense<[2, 4, 4]> : tensor<3xi64>, strides = dense<[1, 2, 2]> : tensor<3xi64>}> : (tensor<3x?x?xi32, #mhlo.type_extensions<bounds = [?, 4, ?]>>) -> tensor<?x?x?xi32>
  // CHECK: types0 = tensor<1x2x2xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xi32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

func.func @slice_with_index_larger_than_bound_dim(%arg0: tensor<3x?x?xi32, #mhlo.type_extensions<bounds = [?, 4, ?]>>) -> tensor<?x?x?xindex> {
  // expected-error@+2 {{'mhlo.slice' op failed to infer returned types}}
  // expected-error@+1 {{limit index 5 is larger than dimension bound 4 in dimension 1}}
  %0 = "mhlo.slice"(%arg0) <{start_indices = dense<[1, 0, 0]> : tensor<3xi64>, limit_indices = dense<[2, 5, 4]> : tensor<3xi64>, strides = dense<[1, 2, 2]> : tensor<3xi64>}> : (tensor<3x?x?xi32, #mhlo.type_extensions<bounds = [?, 4, ?]>>) -> tensor<?x?x?xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xi32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1xindex> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: types0 = tensor<1xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<1xi32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----

// CHECK: func @uniform_dequantize
func.func @uniform_dequantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xindex> {
  %0 = mhlo.uniform_dequantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  // CHECK: types0 = tensor<16x16xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<16x16xf32>) -> tensor<16x16xindex>
  func.return %1 : tensor<16x16xindex>
}

// -----

// CHECK-LABEL: func @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xindex> {
  %0 = "mhlo.fft"(%arg0) <{ fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT> }> : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  // CHECK: types0 = tensor<3x9xcomplex<f32>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xindex>
  func.return %1 : tensor<3x9xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_grad
func.func @batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<?x?x?x?xindex> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  // CHECK: types0 = tensor<2x2x2x2xf32>
  // CHECK-SAME: types1 = tensor<2xf32>
  // CHECK-SAME: types2 = tensor<2xf32>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<2x2x2x2xf32>) -> tensor<?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_train
func.func @batch_norm_train(%input: tensor<2x?x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<?x?x?x?xindex> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 1 : i64} : (tensor<2x?x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x?x2x2xf32>, tensor<?xf32>, tensor<?xf32>)
  // CHECK: types0 = tensor<2x?x2x2xf32>
  // CHECK-SAME: types1 = tensor<?xf32>
  // CHECK-SAME: types2 = tensor<?xf32>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<2x?x2x2xf32>) -> tensor<?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?xindex>
}

// -----

// CHECK-LABEL: @batch_norm_inference
func.func @batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<?x?xindex>) {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK: types0 = tensor<4x256xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<4x256xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @batch_norm_inference_bounds
func.func @batch_norm_inference_bounds(
  %input: tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>, %scale: tensor<?xf32>,
  %offset: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>
) -> (tensor<?x?xindex>) {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32, feature_index = 1 : i64
    } : (tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
  // CHECK: types0 = tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<4x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_grad_bounds
func.func @batch_norm_grad_bounds(
  %input: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
  %scale: tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
  %mean: tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
  %variance: tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
  %grad_output: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
) -> tensor<?x?xindex> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {
    epsilon = 0.001 : f32, feature_index = 1 : i64
  } : (
    tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
  ) ->
    (
    tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  )
  // CHECK: types0 = tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
  // CHECK-SAME: types1 = tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  // CHECK-SAME: types2 = tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: func @batch_norm_train_bounds
func.func @batch_norm_train_bounds(
  %input: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
  %scale: tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
  %offset: tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
) -> tensor<?x?xindex> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32, feature_index = 1 : i64
  } : (
    tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  ) ->
    (
    tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  )
  // CHECK: types0 = tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>
  // CHECK-SAME: types1 = tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  // CHECK-SAME: types2 = tensor<?xf32, #mhlo.type_extensions<bounds = [64]>>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 64]>>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: func @map
func.func @map(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xindex> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK: types0 = tensor<4x5xf32>
  %2 = "mhlo_test.get_return_types"(%0) : (tensor<4x5xf32>) -> tensor<4x5xindex>
  func.return %2 : tensor<4x5xindex>
}

// -----

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xindex> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  // CHECK: types0 = tensor<10x5x4x4xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xindex>
  func.return %1 : tensor<10x5x4x4xindex>
}

// -----

// CHECK-LABEL: func @if
func.func @if(%pred : tensor<i1>, %branch_operand : tensor<2xf32>, %wrong_type : tensor<2xf32>) -> tensor<2xindex> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%wrong_type) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  // CHECK: types0 = tensor<2xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<2xf32>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: func @case
func.func @case(%index : tensor<i32>, %branch_operand : tensor<2xf32>) -> tensor<2xindex> {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<i32>) -> tensor<2xf32>
  // CHECK: types0 = tensor<2xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<2xf32>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: func @sort
func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) -> tensor<16x16xindex> {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  // CHECK: types0 = tensor<16x16xf32>, types1 = tensor<16x16xi32>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<16x16xf32>) -> tensor<16x16xindex>
  func.return %1 : tensor<16x16xindex>
}

// -----

// CHECK-LABEL: @sort_bounds_and_unknown_rank
func.func @sort_bounds_and_unknown_rank(%input0: tensor<?x?x?xf32>, %input1: tensor<5x?x?xi32, #mhlo.type_extensions<bounds = [?, 7, 6]>>) {
  %0, %1 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %pred = "mhlo.compare"(%arg0, %arg1) {
      comparison_direction = #mhlo<comparison_direction GT>
    } : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%pred) : (tensor<i1>) -> ()
  }) { dimension = 1 : i64, is_stable = true } : (
    tensor<?x?x?xf32>,
    tensor<5x?x?xi32, #mhlo.type_extensions<bounds = [?, 7, 6]>>
  ) -> (tensor<?x?x?xf32>, tensor<?x?x?xi32>)
  // CHECK: types0 = tensor<?x?x?xf32>
  // CHECK-SAME: types1 = tensor<5x?x?xi32, #mhlo.type_extensions<bounds = [?, 7, 6]>>
  %2 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xindex>
  func.return
}

// -----

// CHECK-LABEL: func @outfeed
func.func @outfeed(%arg0: tensor<3x3x3xi32>, %arg1: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.outfeed"(%arg0, %arg1) {
    outfeed_config = ""
  } : (tensor<3x3x3xi32>, !mhlo.token) -> !mhlo.token
  // CHECK: types0 = !mhlo.token
  %1 = "mhlo_test.get_return_types"(%0) : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

// CHECK-LABEL: func @while
func.func @while(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>) -> tensor<index> {
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %cst_1 = arith.constant dense<0> : tensor<i32>
  %cst_2 = arith.constant dense<1000> : tensor<i32>
  %1:3 = "mhlo.while"(%cst_1, %cst, %cst_2) ({
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %4 = "mhlo.compare"(%arg9, %arg11) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %3 = mhlo.add %arg9, %cst_0 : tensor<i32>
    "mhlo.return"(%3, %arg10, %arg11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK: types0 = tensor<i32>, types1 = tensor<i32>, types2 = tensor<i32>
  %4 = "mhlo_test.get_return_types"(%1#0) : (tensor<i32>) -> tensor<index>
  func.return %4 : tensor<index>
}

// -----

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<4x2xf32>) -> tensor<index> {
  %0 = "mhlo.get_dimension_size"(%arg0) <{dimension = 1 : i64}> : (tensor<4x2xf32>) -> tensor<i32>
  // CHECK: types0 = tensor<i32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<i32>) -> tensor<index>
  func.return %1 : tensor<index>
}

// -----

// CHECK-LABEL: @dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<4x4xi32>, %arg1: tensor<2x2xi32>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<4x4xindex> {
  %0 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
  // CHECK: types0 = tensor<4x4xi32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<4x4xi32>) -> tensor<4x4xindex>
  func.return %1 : tensor<4x4xindex>
}

// -----

// CHECK-LABEL: @dynamic_update_slice_with_bounds
func.func @dynamic_update_slice_with_bounds(%input: tensor<3x?x?xi64, #mhlo.type_extensions<bounds = [?, ?, 5]>>, %update: tensor<1x4x3xi64>, %start1: tensor<i64>, %start2: tensor<i64>, %start3 : tensor<i64>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.dynamic_update_slice"(%input, %update, %start1, %start2, %start3) : (tensor<3x?x?xi64, #mhlo.type_extensions<bounds = [?, ?, 5]>>, tensor<1x4x3xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x?x?xi64>
  // CHECK: types0 = tensor<3x?x?xi64, #mhlo.type_extensions<bounds = [?, ?, 5]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<3x?x?xi64>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: func @create_token
func.func @create_token() -> !mhlo.token {
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  // CHECK: types0 = !mhlo.token
  %1 = "mhlo_test.get_return_types"(%0) : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

// CHECK-LABEL: func @after_all_empty_arg
func.func @after_all_empty_arg() -> !mhlo.token {
  %0 = "mhlo.after_all"() : () -> !mhlo.token
  %1 = "mhlo_test.get_return_types"(%0) : (!mhlo.token) -> !mhlo.token
  // CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = !mhlo.token} : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

// CHECK-LABEL: func @after_all
func.func @after_all(%arg0: !mhlo.token, %arg1: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0, %arg1) : (!mhlo.token, !mhlo.token) -> !mhlo.token
  %1 = "mhlo_test.get_return_types"(%0) : (!mhlo.token) -> !mhlo.token
  // CHECK: %1 = "mhlo_test.return_types"(%0) {types0 = !mhlo.token} : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

// CHECK: func @select_and_scatter
func.func @select_and_scatter(
  %arg0: tensor<10x24x24x64xf32>,
  %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xindex> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>

  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "mhlo.compare"(%arg3, %arg4) {
      compare_type = #mhlo<comparison_type TOTALORDER>,
      comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) ->
        tensor<10x24x24x64xf32>
  // CHECK: types0 = tensor<10x24x24x64xf32>
  %3 = "mhlo_test.get_return_types"(%1) : (tensor<10x24x24x64xf32>) -> tensor<10x24x24x64xindex>
  func.return %3 : tensor<10x24x24x64xindex>
}

// -----

// CHECK-LABEL: func @scatter
func.func @scatter(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xindex> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  // CHECK: types0 = tensor<200x100x300xf32>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<200x100x300xf32>) -> tensor<200x100x300xindex>
  func.return %1 : tensor<200x100x300xindex>
}

// -----

// CHECK-LABEL: func @scatter_bounds
func.func @scatter_bounds(%input_tensor: tensor<200x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 301]>>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<?x?x?xindex> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 301]>>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x?x?xf32>

  // CHECK: types0 = tensor<200x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 301]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<200x?x?xf32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

//===----------------------------------------------------------------------===//
// Sparsity
//===----------------------------------------------------------------------===//

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: @tanh_sparsity
func.func @tanh_sparsity(%arg0: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.tanh"(%arg0) : (tensor<10x10xf32, #CSR>) -> tensor<10x10xf32>
  // CHECK: types0 = tensor<10x10xf32, {{.*}}>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: @abs_sparsity
func.func @abs_sparsity(%arg0: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.abs"(%arg0) : (tensor<10x10xf32, #CSR>) -> tensor<10x10xf32>
  // CHECK: types0 = tensor<10x10xf32, {{.*}}>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: @real_sparsity
func.func @real_sparsity(%arg0: tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.real"(%arg0) : (tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xf32>
  // CHECK: types0 = tensor<10x10xf32, {{.*}}>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: @imag_sparsity
func.func @imag_sparsity(%arg0: tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.imag"(%arg0) : (tensor<10x10xcomplex<f32>, #CSR>) -> tensor<10x10xf32>
  // CHECK: types0 = tensor<10x10xf32, {{.*}}>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x10xf32>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: @complex_sparsity
func.func @complex_sparsity(%arg0: tensor<10x10xf32, #CSR>, %arg1: tensor<10x10xf32, #CSR>) -> tensor<10x10xindex> {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<10x10xf32, #CSR>, tensor<10x10xf32, #CSR>) -> tensor<10x10xcomplex<f32>>
  // CHECK: types0 = tensor<10x10xcomplex<f32>, {{.*}}>}
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<10x10xcomplex<f32>>) -> tensor<10x10xindex>
  func.return %1 : tensor<10x10xindex>
}

// -----

// CHECK-LABEL: func @reduce
func.func @reduce(%arg0: tensor<7x5xf32>, %arg1 : tensor<5xf32>)
    -> (tensor<5xindex>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<5xf32>, %arg3: tensor<5xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    "mhlo.return"(%1) : (tensor<5xf32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<7x5xf32>, tensor<5xf32>) -> tensor<5xf32>

  // CHECK: types0 = tensor<5xf32>
  %2 = "mhlo_test.get_return_types"(%0) : (tensor<5xf32>) -> tensor<5xindex>
  func.return %2: tensor<5xindex>
}

// -----

// CHECK-LABEL: func @reduce_with_bounds
func.func @reduce_with_bounds(%arg0: tensor<?x?x5xf32, #mhlo.type_extensions<bounds = [3, 7, ?]>>, %arg1 : tensor<5xf32>)
    -> (tensor<?x?x?xindex>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<5xf32>, %arg3: tensor<5xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    "mhlo.return"(%1) : (tensor<5xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>}
      : (tensor<?x?x5xf32, #mhlo.type_extensions<bounds = [3, 7, ?]>>, tensor<5xf32>)
          -> tensor<?x5xf32, #mhlo.type_extensions<bounds = [7, ?]>>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [7, ?]>>
  %2 = "mhlo_test.get_return_types"(%0)
      : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [7, ?]>>) -> tensor<?x?x?xindex>

  func.return %2: tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>,
                    %init0: tensor<f32>, %init1: tensor<i32>) -> tensor<2x2xindex> {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> }
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  // CHECK: types0 = tensor<2x2xf32>, types1 = tensor<2x2xi32>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<2x2xf32>) -> tensor<2x2xindex>
  func.return %1 : tensor<2x2xindex>
}

// -----

//===----------------------------------------------------------------------===//
// Bounded Dynamism
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @tensor_bounds
func.func @tensor_bounds(%arg0: tensor<3x5xf32>, %arg1: tensor<i32>) -> tensor<?x?xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x5xf32>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @static_tensor_bounds
func.func @static_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [8, ?]>>) -> tensor<?x?xindex> {
  %bounds = mhlo.constant dense<8> : tensor<i32>
  %result = "mhlo.set_dimension_size"(%arg0, %bounds) {dimension = 0 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [8, ?]>>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: types0 = tensor<8x5xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @edit_tensor_bounds
func.func @edit_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<?x?xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, 5]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @retain_tensor_bounds
func.func @retain_tensor_bounds(%arg0: tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<?x?xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: types0 = tensor<?x5xf32, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: @unknown_bounds
func.func @unknown_bounds(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, %arg1: tensor<i32>) -> tensor<?x?xindex> {
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>

  // CHECK: types0 = tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// This test covers all cases (except the "Error out" case) for type inference
// of binary op with bounds
// See PairwiseSameOperandAndResultType::inferDimWithBound()
// CHECK-LABEL: @add_bounds
func.func @add_bounds(
  %arg0: tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>,
  %arg1: tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>) -> tensor<?x?x?x?x?x?x?xindex> {
  %result1 = "mhlo.add"(%arg0, %arg1) : (
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>,
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>)
    -> tensor<?x?x?x?x?x?x?xf32>
  %result2 = "mhlo.add"(%arg1, %arg0) : (
    tensor<3x?x?x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, ?, 3, 3, 4]>>,
    tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, ?, 3, 3]>>)
    -> tensor<?x?x?x?x?x?x?xf32>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, 3, 3, 3]>>
  %1 = "mhlo_test.get_return_types"(%result1) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?x?xindex>

  // CHECK: types0 = tensor<3x3x3x?x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, ?, 3, 3, 3]>>
  %2 = "mhlo_test.get_return_types"(%result2) : (tensor<?x?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?x?x?x?xindex>
}

// -----

// This test covers "Error out" case for type inference of binary op with bounds
// See PairwiseSameOperandAndResultType::inferDimWithBound()
func.func @add_bounds_mismatch(
  %arg0: tensor<3xf32, #mhlo.type_extensions<bounds = [?]>>,
  %arg1: tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> tensor<?xindex> {
  // expected-error@+1 {{requires compatible types for all operands and results}}
  %result = "mhlo.add"(%arg0, %arg1) : (
    tensor<3xf32, #mhlo.type_extensions<bounds = [?]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> tensor<?xf32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<?xf32>) -> tensor<?xindex>
  func.return %1 : tensor<?xindex>
}

// -----

// CHECK-LABEL: @partition_id
func.func @partition_id() -> tensor<?xindex> {
  %result = "mhlo.partition_id"() : () -> tensor<ui32>
  // CHECK: types0 = tensor<ui32>
  %1 = "mhlo_test.get_return_types"(%result) : (tensor<ui32>) -> tensor<?xindex>
  func.return %1 : tensor<?xindex>
}

// -----

// CHECK-LABEL: @send
func.func @send(%arg0: !mhlo.token) -> !mhlo.token {
  %result = "mhlo.send"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  } : (!mhlo.token) -> !mhlo.token
  // CHECK: types0 = !mhlo.token
  %1 = "mhlo_test.get_return_types"(%result) : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

// CHECK-LABEL: func @gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x4x2xi32>, %[[ARG1:.*]]: tensor<?x3x2xi64>
func.func @gather(%operand : tensor<3x4x2xi32>, %start_indices : tensor<?x3x2xi64>) -> tensor<4xindex> {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x3x2xi64>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[DIM]], %[[C3]], %[[C2]], %[[C2]] : tensor<4xindex>
  // CHECK: return %[[RES]] : tensor<4xindex>
  %result = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
      slice_sizes = dense<[1, 2, 2]> : tensor<3xi64>,
      indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<?x3x2xi64>) -> tensor<?x3x2x2xi32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result) : (tensor<?x3x2x2xi32>) -> tensor<4xindex>
  func.return %1 : tensor<4xindex>
}

// -----

// CHECK-LABEL: func @pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x48x48x32xf32>
func.func @pad(%arg0: tensor<?x48x48x32xf32>) -> tensor<4xindex> {
  // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[CST1:.*]] = arith.constant 48 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[CST0]] : tensor<?x48x48x32xf32>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[DIM]], %[[CST1]], %[[CST1]], %[[CST1]] : tensor<4xindex>
  // CHECK: return %[[RES]] : tensor<4xindex>
  %0 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %result = "mhlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[0, 0, 0, 16]> : tensor<4xi64>,
    edge_padding_low = dense<0> : tensor<4xi64>,
    interior_padding = dense<0> : tensor<4xi64>
  } : (tensor<?x48x48x32xf32>, tensor<f32>) -> tensor<?x48x48x48xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result) : (tensor<?x48x48x48xf32>) -> tensor<4xindex>
  func.return %1 : tensor<4xindex>
}

// -----

// CHECK-LABEL: func @cholesky_bounds
func.func @cholesky_bounds(%input: tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, ?]>>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.cholesky"(%input) { lower = true } : (tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, ?]>>) -> tensor<?x?x?xf32>
  // CHECK: types0 = tensor<2x?x?xf32, #mhlo.type_extensions<bounds = [?, 5, ?]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// CHECK-LABEL: func @concatenate
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>
func.func @concatenate(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<2xindex> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xi32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xi32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xi32>
  // CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xi32>
  // CHECK: %[[V0:.*]] = arith.addi %[[DIM]], %[[DIM1]] : index
  // CHECK: %[[V1:.*]] = arith.addi %[[V0]], %[[DIM2]] : index
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[V1]], %[[DIM0]] : tensor<2xindex>
  // CHECK: return %[[RES]] : tensor<2xindex>
  %result = "mhlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 0 : i64
  } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result) : (tensor<?x?xi32>) -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// -----

// CHECK-LABEL: func @reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x?xf32>,
func.func @reduce(%arg0: tensor<4x?xf32>, %arg1 : tensor<4xf32>)-> (tensor<1xindex>) {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<4x?xf32>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[DIM]] : tensor<1xindex>
  // CHECK: return %[[RES]] : tensor<1xindex>
  %result = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x?xf32>, tensor<4xf32>) -> tensor<?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?xf32>) -> tensor<1xindex>
  func.return %1: tensor<1xindex>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<1xindex>, %[[ARG2:.*]]: tensor<1xindex>, %[[ARG3:.*]]: tensor<1xindex>
func.func @real_dynamic_slice(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[EXTD:.*]] = tensor.extract %[[ARG1]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[EXTD0:.*]] = tensor.extract %[[ARG2]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[EXTD1:.*]] = tensor.extract %[[ARG3]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[V0:.*]] = arith.subi %[[EXTD0]], %[[EXTD]] : index
  // CHECK: %[[V1:.*]] = arith.addi %[[EXTD1]], %[[V0]] : index
  // CHECK: %[[V2:.*]] = arith.subi %[[V1]], %[[C1]] : index
  // CHECK: %[[V3:.*]] = arith.divsi %[[V2]], %[[EXTD1]] : index
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[V3]] : tensor<1xindex>
  // CHECK: return %[[RES]] : tensor<1xindex>
  %result = "mhlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?xf32>) -> tensor<1xindex>
  func.return %1: tensor<1xindex>
}

// -----

// CHECK-LABEL: func @dot_general
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>
func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<3xindex> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<?x?x?xf32>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[DIM]], %[[DIM0]], %[[DIM1]] : tensor<3xindex>
  // CHECK: return %[[RES]] : tensor<3xindex>
  %result = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?x?x?xf32>) -> tensor<3xindex>
  func.return %1: tensor<3xindex>
}

// -----

// CHECK-LABEL: func @dynamic_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<1xindex>, %[[ARG3:.*]]: tensor<1xindex>, %[[ARG4:.*]]: tensor<1xindex>
func.func @dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xf32>
  // CHECK: %[[EXTD:.*]] = tensor.extract %[[ARG2]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[EXTD0:.*]] = tensor.extract %[[ARG3]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[EXTD1:.*]] = tensor.extract %[[ARG4]][%[[C0]]] : tensor<1xindex>
  // CHECK: %[[V0:.*]] = arith.cmpi slt, %[[DIM]], %[[C1]] : index
  // CHECK: %[[V1:.*]] = arith.subi %[[DIM]], %[[C1]] : index
  // CHECK: %[[V2:.*]] = arith.select %[[V0]], %[[C0]], %[[V1]] : index
  // CHECK: %[[V3:.*]] = arith.muli %[[EXTD1]], %[[V2]] : index
  // CHECK: %[[V4:.*]] = arith.addi %[[V3]], %[[DIM]] : index
  // CHECK: %[[V5:.*]] = arith.addi %[[V4]], %[[EXTD]] : index
  // CHECK: %[[V6:.*]] = arith.addi %[[V5]], %[[EXTD0]] : index
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[V6]] : tensor<1xindex>
  // CHECK: return %[[RES]] : tensor<1xindex>
  %result = "mhlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?xf32>) -> tensor<1xindex>
  func.return %1: tensor<1xindex>
}

// -----

// CHECK-LABEL: func @broadcast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xi32>
func.func @broadcast(%arg0: tensor<?xi32>) -> tensor<3xindex> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?xi32>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[C1]], %[[C2]], %[[DIM]] : tensor<3xindex>
  // CHECK: return %[[RES]] : tensor<3xindex>
  %result = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[1, 2]> : tensor<2xi64>}> : (tensor<?xi32>) -> tensor<1x2x?xi32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<1x2x?xi32>) -> tensor<3xindex>
  func.return %1: tensor<3xindex>
}

// -----

// CHECK-LABEL: func @transpose
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?x?x?xi32>
func.func @transpose(%arg0: tensor<?x?x?x?xi32>) -> tensor<4xindex> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xi32>
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xi32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xi32>
  // CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xi32>
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[DIM0]], %[[DIM]], %[[DIM2]], %[[DIM1]] : tensor<4xindex>
  // CHECK: return %[[RES]] : tensor<4xindex>
  %result = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}> : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?x?x?x?xi32>) -> tensor<4xindex>
  func.return %1: tensor<4xindex>
}

// -----

// CHECK-LABEL: func @dynamic_iota
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xindex>
func.func @dynamic_iota(%arg0: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: return %[[ARG0]] : tensor<1xindex>
  %result = "mhlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%result): (tensor<?xf32>) -> tensor<1xindex>
  func.return %1: tensor<1xindex>
}

// -----

// CHECK: func @select_and_scatter_bound
func.func @select_and_scatter_bound(
    %arg0: tensor<?x24x24x64xf32, #mhlo.type_extensions<bounds = [10, ?, ?, ?]>>,
    %arg1: tensor<?x12x12x64xf32, #mhlo.type_extensions<bounds = [10, ?, ?, ?]>>) -> tensor<?x?x?x?xindex> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "mhlo.compare"(%arg3, %arg4) {
      compare_type = #mhlo<comparison_type TOTALORDER>,
      comparison_direction = #mhlo<comparison_direction GE>
    } : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<?x24x24x64xf32, #mhlo.type_extensions<bounds = [10, ?, ?, ?]>>,
       tensor<?x12x12x64xf32, #mhlo.type_extensions<bounds = [10, ?, ?, ?]>>,
       tensor<f32>) -> tensor<?x?x?x?xf32>
  // CHECK: types0 = tensor<?x24x24x64xf32, #mhlo.type_extensions<bounds = [10, ?, ?, ?]>>
  %3 = "mhlo_test.get_return_types"(%1) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xindex>
  func.return %3 : tensor<?x?x?x?xindex>
}

// -----

// CHECK-LABEL: func @reduce_window_bound
func.func @reduce_window_bound(%arg0: tensor<4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, 2]>>,
    %init0: tensor<f32>) -> (tensor<?x?x?x?xindex>) {
  %0:1 = "mhlo.reduce_window"(%arg0, %init0) ({
  ^bb0(%a0: tensor<f32>, %b0: tensor<f32>):
    %2 = mhlo.add %a0, %b0 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    padding = dense<[[0, 0], [0, 0], [2, 2], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = dense<[1, 1, 5, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 1, 3, 1]> : tensor<4xi64>
  } : (tensor<4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 4, 2]>>,
       tensor<f32>) -> (tensor<?x?x?x?xf32>)
  // CHECK: types0 = tensor<4x?x?x?xf32, #mhlo.type_extensions<bounds = [?, ?, 2, 2]>>
  %1 = "mhlo_test.get_return_types"(%0#0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xindex>
  func.return %1: tensor<?x?x?x?xindex>
}

// -----

// CHECK-LABEL: func @triangular_solve_bounds
func.func @triangular_solve_bounds(
    %arg0: tensor<10x5x?x4xf32, #mhlo_test.type_extensions<bounds = [?, ?, 5, ?]>>,
    %arg1: tensor<10x5x?x?xf32, #mhlo_test.type_extensions<bounds = [?, ?, ?, 7]>>) -> tensor<?x?x?x?xindex> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {
    left_side = false,
    lower = true,
    transpose_a = #mhlo<transpose NO_TRANSPOSE>,
    unit_diagonal = true
  } : (tensor<10x5x?x4xf32, #mhlo_test.type_extensions<bounds = [?, ?, 5, ?]>>,
       tensor<10x5x?x?xf32, #mhlo_test.type_extensions<bounds = [?, ?, ?, 7]>>) -> tensor<?x?x?x?xf32>
  // CHECK: types0 = tensor<10x5x?x?xf32, #mhlo_test.type_extensions<bounds = [?, ?, ?, 7]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?xindex>
}

//-----

// CHECK-LABEL: func @fft_bound
func.func @fft_bound(%arg0: tensor<?x9xcomplex<f32>, #mhlo.type_extensions<bounds = [3, ?]>>) -> tensor<?x?xindex> {
  %0 = "mhlo.fft"(%arg0) {
    fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT>
  } : (tensor<?x9xcomplex<f32>, #mhlo.type_extensions<bounds = [3, ?]>>) -> tensor<?x?xcomplex<f32>>
  // CHECK: types0 = tensor<?x9xcomplex<f32>, #mhlo.type_extensions<bounds = [3, ?]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xindex>
  func.return %1 : tensor<?x?xindex>
}

// -----

// CHECK-LABEL: func @rfft_with_bound
func.func @rfft_with_bound(%arg0: tensor<3x?x?xf32, #mhlo.type_extensions<bounds = [?, 3, 10]>>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.fft"(%arg0) {
    fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>
  } : (tensor<3x?x?xf32, #mhlo.type_extensions<bounds = [?, 3, 10]>>) -> tensor<?x?x?xcomplex<f32>>
  // CHECK: types0 = tensor<3x?x5xcomplex<f32>, #mhlo.type_extensions<bounds = [?, 3, ?]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: func @irfft_with_bound
func.func @irfft_with_bound(%arg0: tensor<3x?x?xcomplex<f32>, #mhlo.type_extensions<bounds = [?, 3, 17]>>) -> tensor<?x?x?xindex> {
  %0 = "mhlo.fft"(%arg0) {
    fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT>
  } : (tensor<3x?x?xcomplex<f32>, #mhlo.type_extensions<bounds = [?, 3, 17]>>) -> tensor<?x?x?xf32>
  // CHECK: types0 = tensor<3x?x9xf32, #mhlo.type_extensions<bounds = [?, 3, ?]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xindex>
  func.return %1 : tensor<?x?x?xindex>
}

// -----

// CHECK-LABEL: @select
func.func @select(%pred : tensor<i1>,
    %a : tensor<?x2x3x?xf32, #mhlo.type_extensions<bounds = [5, ?, ?, 7]>>,
    %b : tensor<1x?x3x?xf32, #mhlo.type_extensions<bounds = [?, 6, ?, 8]>>) -> tensor<?x?x?x?xindex> {
  %0 = "mhlo.select"(%pred, %a, %b) : (tensor<i1>,
      tensor<?x2x3x?xf32, #mhlo.type_extensions<bounds = [5, ?, ?, 7]>>,
      tensor<1x?x3x?xf32, #mhlo.type_extensions<bounds = [?, 6, ?, 8]>>) -> tensor<?x?x?x?xf32>
  // CHECK: types0 = tensor<1x2x3x?xf32, #mhlo.type_extensions<bounds = [?, ?, ?, 7]>>
  %1 = "mhlo_test.get_return_types"(%0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xindex>
  func.return %1 : tensor<?x?x?x?xindex>
}

// -----

// CHECK-LABEL: func @dynamic_gather
func.func @dynamic_gather(%arg0: tensor<?x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xindex> {
  %0 = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "mhlo.dynamic_gather"(%arg0, %arg1, %0) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [0, 1],
      start_index_map = [1]
    >,
    indices_are_sorted = true
  } : (tensor<?x4xf32>, tensor<1xi64>, tensor<2xi32>) -> tensor<?x?xf32>
  // CHECK: types0 = tensor<1x2xf32>
  %2 = "mhlo_test.get_return_types"(%1) : (tensor<?x?xf32>) -> tensor<?x?xindex>
  func.return %2 : tensor<?x?xindex>
}
