// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// Folding this case would explode the IR
func.func @scatter_fold_explosion() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<1> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<511x1x6400x6400xf32>
  // CHECK: mhlo.scatter
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [3]>, unique_indices = true} : (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<511x1x6400x6400xf32>) -> tensor<512x1x6400x6400xf32>

  func.return %scatter :  tensor<512x1x6400x6400xf32>
}

// -----

// Verify that a full overwrite of the "base" with a scatter is not folded
// if the type mismatch.
// TODO(mhlo): this would be nice to handle: the update could be broadcasted
// to the type of the base here.
func.func @scatter_full_overwrite_type_mismatch(%base : tensor<1x1x1xf64>) ->  tensor<1x1x1xf64> {
  %0 = mhlo.constant dense<0.28209479177387814> : tensor<1xf64>
  %1 = mhlo.constant dense<0> : tensor<2xi32>
  %scatter = "mhlo.scatter"(%base, %1, %0) ({
  ^bb0(%arg11: tensor<f64>, %arg12: tensor<f64>):
    "mhlo.return"(%arg12) : (tensor<f64>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1]>, unique_indices = true} : (tensor<1x1x1xf64>, tensor<2xi32>, tensor<1xf64>) -> tensor<1x1x1xf64>

  // CHECK: %[[SCATTER:.*]] = "mhlo.scatter
  // CHECK: return %[[SCATTER]]
  func.return %scatter :  tensor<1x1x1xf64>
}

// -----

// Verify that a full overwrite of the "base" with a scatter is correctly folded
// even if the tensor is huge.
func.func @scatter_full_overwrite() ->  tensor<512x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<512x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [3]>, unique_indices = true} : (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) -> tensor<512x1x6400x6400xf32>

  // CHECK: %[[FOLD:.*]] = mhlo.constant dense<1.000000e+00> : tensor<512x1x6400x6400xf32>
  // CHECK: return %[[FOLD]]
  func.return %scatter :  tensor<512x1x6400x6400xf32>
}

// -----

// Verify that a full overwrite of the "base" with a batched scatter is
// correctly folded.
func.func @scatter_batching_dims_full_overwrite() ->  tensor<3x1x6400x6400xf32> {
  %base = mhlo.constant dense<0.000000e+00> : tensor<3x1x6400x6400xf32>
  %index = mhlo.constant dense<0> : tensor<3x1xi32>
  %update = mhlo.constant dense<1.000000e+00> : tensor<3x1x6400x6400xf32>
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1, 2, 3], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [3], index_vector_dim = 1>, unique_indices = true} : (tensor<3x1x6400x6400xf32>, tensor<3x1xi32>, tensor<3x1x6400x6400xf32>) -> tensor<3x1x6400x6400xf32>

  // CHECK: %[[FOLD:.*]] = mhlo.constant dense<1.000000e+00> : tensor<3x1x6400x6400xf32>
  // CHECK: return %[[FOLD]]
  func.return %scatter :  tensor<3x1x6400x6400xf32>
}

// -----

// Verify that a full overwrite of the "base" with a scatter is correctly folded
// even if the base and update are not constant values.
func.func @scatter_full_overwrite_non_const(
        %base : tensor<512x1x6400x6400xf32>,
        %update : tensor<512x1x6400x6400xf32>) ->  tensor<512x1x6400x6400xf32> {
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      "mhlo.return"(%arg6) : (tensor<f32>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [3]>, unique_indices = true} : (tensor<512x1x6400x6400xf32>, tensor<1xi32>, tensor<512x1x6400x6400xf32>) -> tensor<512x1x6400x6400xf32>

  // CHECK: return %arg1
  func.return %scatter :  tensor<512x1x6400x6400xf32>
}

// -----

// CHECK-LABEL: func @scatter_full_overwrite_add(
//  CHECK-SAME: %[[ARG0:[A-Za-z0-9]+]]: tensor<1xbf16>,
//  CHECK-SAME: %[[ARG1:[A-Za-z0-9]+]]: tensor<0xi32>,
//  CHECK-SAME: %[[ARG2:[A-Za-z0-9]+]]: tensor<1xbf16>
func.func @scatter_full_overwrite_add(
        %base : tensor<1xbf16>,
        %index : tensor<0xi32>,
        %update : tensor<1xbf16>) ->  tensor<1xbf16> {
  %scatter = "mhlo.scatter"(%base, %index, %update) ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      %2 = mhlo.add %arg3, %arg4 : tensor<bf16>
      mhlo.return %2 : tensor<bf16>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0]>, unique_indices = true} : (tensor<1xbf16>, tensor<0xi32>, tensor<1xbf16>) -> tensor<1xbf16>

  // CHECK: "mhlo.map"(%[[ARG0]], %[[ARG2]]) <{dimensions = dense<0> : tensor<1xi64>}> ({
  // CHECK:  ^bb0(%[[ARG3:.*]]: tensor<bf16>, %[[ARG4:.*]]: tensor<bf16>):
  // CHECK:    %[[ADD:.*]] = mhlo.add %[[ARG3]], %[[ARG4]] : tensor<bf16>
  // CHECK:    mhlo.return %[[ADD]] : tensor<bf16>
  // CHECK:  }) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
  func.return %scatter : tensor<1xbf16>
}

// -----

// Verify that a full overwrite of the "base" with a scatter is not folded when
// there is a non-identity computation.
func.func public @scatter_non_identity(%arg0: tensor<12xbf16>, %arg1: tensor<12xbf16>) -> tensor<12xbf16> {
  %0 = mhlo.constant dense<0> : tensor<1xi32>
  %1 = "mhlo.scatter"(%arg0, %0, %arg1) ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %2 = mhlo.add %arg2, %arg3 : tensor<bf16>
    "mhlo.return"(%2) : (tensor<bf16>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<12xbf16>, tensor<1xi32>, tensor<12xbf16>) -> tensor<12xbf16>
  // CHECK: %[[SCATTER:.*]] = "mhlo.scatter
  // CHECK: return %[[SCATTER]]
  func.return %1 : tensor<12xbf16>
}
