// RUN: mlir-hlo-opt %s --split-input-file --hlo-canonicalize-scatter | FileCheck %s

func.func @insert_index_vector_and_window_dims(%dst1: tensor<3x3xf32>,
    %dst2: tensor<3x3xf32>, %indices: tensor<2xi32>, %update1: tensor<2x3xf32>,
    %update2: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %0, %1 = "mhlo.scatter"(%dst1, %dst2, %indices, %update1, %update2) ({
  ^bb0(%u1: tensor<f32>, %d1: tensor<f32>, %u2: tensor<f32>, %d2: tensor<f32>):
    "mhlo.return"(%u1, %u2) : (tensor<f32>, tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<2xi32>,
       tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func.func @insert_index_vector_and_window_dims(
// CHECK-SAME:      %[[DST1:.*]]: tensor<3x3xf32>, %[[DST2:.*]]: tensor<3x3xf32>,
// CHECK-SAME:      %[[IND:.*]]: tensor<2xi32>,
// CHECK-SAME:      %[[UPD1:.*]]: tensor<2x3xf32>, %[[UPD2:.*]]: tensor<2x3xf32>)

// CHECK:         %[[IND_:.*]] = tensor.expand_shape %[[IND]] [
// CHECK-SAME:      [0, 1]] output_shape [2, 1] : tensor<2xi32> into tensor<2x1xi32>
// CHECK:         %[[UPD1_:.*]] = tensor.expand_shape %[[UPD1]] [
// CHECK-SAME:      [0], [1, 2]] output_shape [2, 1, 3] : tensor<2x3xf32> into tensor<2x1x3xf32>
// CHECK:         %[[UPD2_:.*]] = tensor.expand_shape %[[UPD2]] [
// CHECK-SAME:      [0], [1, 2]] output_shape [2, 1, 3] : tensor<2x3xf32> into tensor<2x1x3xf32>

// CHECK:         "mhlo.scatter"(%[[DST1]], %[[DST2]], %[[IND_]], %[[UPD1_]], %[[UPD2_]])
// CHECK:           update_window_dims = [1, 2],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0],
// CHECK-SAME:      index_vector_dim = 1
// CHECK-SAME:      unique_indices = false

// -----

func.func @collapse_scatter_dims(%dst: tensor<3x3xf32>,
    %indices: tensor<2x1x2xi32>, %update: tensor<2x1x1x3xf32>) -> tensor<3x3xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%u: tensor<f32>,  %d: tensor<f32>):
    "mhlo.return"(%u) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3x3xf32>, tensor<2x1x2xi32>, tensor<2x1x1x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func.func @collapse_scatter_dims(
// CHECK-SAME:      %[[DST:.*]]: tensor<3x3xf32>, %[[IND:.*]]: tensor<2x1x2xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<2x1x1x3xf32>)

// CHECK:         %[[IND_:.*]] = tensor.collapse_shape %[[IND]] {{\[\[}}0, 1], [2]] : tensor<2x1x2xi32> into tensor<2x2xi32>
// CHECK:         %[[UPD_:.*]] = tensor.collapse_shape %[[UPD]] {{\[\[}}0, 1], [2], [3]] : tensor<2x1x1x3xf32> into tensor<2x1x3xf32>
// CHECK:         "mhlo.scatter"(%[[DST]], %[[IND_]], %[[UPD_]])
// CHECK-SAME:      update_window_dims = [1, 2],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0, 1],
// CHECK-SAME:      index_vector_dim = 1

// -----

func.func @move_index_vector_dim(%dst: tensor<3x3xf32>,
    %indices: tensor<2x1xi32>, %update: tensor<1x3x3xf32>) -> tensor<3x3xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%u: tensor<f32>,  %d: tensor<f32>):
    "mhlo.return"(%u) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 0,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3x3xf32>, tensor<2x1xi32>, tensor<1x3x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func.func @move_index_vector_dim(
// CHECK-SAME:      %[[DST:.*]]: tensor<3x3xf32>,
// CHECK-SAME:      %[[IND:.*]]: tensor<2x1xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<1x3x3xf32>

// CHECK:         %[[IND_:.*]] = "mhlo.transpose"(%[[IND]]) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2x1xi32>) -> tensor<1x2xi32>
// CHECK:         "mhlo.scatter"(%[[DST]], %[[IND_]], %[[UPD]])
// CHECK:           update_window_dims = [1, 2],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0, 1],
// CHECK-SAME:      index_vector_dim = 1

// -----

func.func @transform_updates_and_operands_using_scatter_dims(%dst: tensor<3x4x5xf32>,
    %indices: tensor<2x2xi32>, %update: tensor<2x1x1x3xf32>) -> tensor<3x4x5xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%u: tensor<f32>,  %d: tensor<f32>):
    "mhlo.return"(%u) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1, 2, 3],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [2, 0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3x4x5xf32>, tensor<2x2xi32>, tensor<2x1x1x3xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
// CHECK-LABEL: func.func @transform_updates_and_operands_using_scatter_dims(
// CHECK-SAME:      %[[DST:.*]]: tensor<3x4x5xf32>,
// CHECK-SAME:      %[[IND:.*]]: tensor<2x2xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<2x1x1x3xf32>) -> tensor<3x4x5xf32> {

// CHECK:         %[[DST_:.*]] = "mhlo.transpose"(%[[DST]]) <{
// CHECK-SAME:      permutation = dense<[2, 0, 1]> : tensor<3xi64>
// CHECK-SAME:    }> : (tensor<3x4x5xf32>) -> tensor<5x3x4xf32>
// CHECK:         %[[UPD_:.*]] = "mhlo.transpose"(%[[UPD]]) <{
// CHECK-SAME:      permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>
// CHECK-SAME:    }> : (tensor<2x1x1x3xf32>) -> tensor<2x3x1x1xf32>

// CHECK:         %[[NEW_OP:.*]] = "mhlo.scatter"(%[[DST_]], %[[IND]], %[[UPD_]])
// CHECK-SAME:       update_window_dims = [1, 2, 3],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0, 1],
// CHECK-SAME:      index_vector_dim = 1

// CHECK:        "mhlo.transpose"(%[[NEW_OP:.*]]) <{
// CHECK-SAME:      permutation = dense<[1, 2, 0]> : tensor<3xi64>
// CHECK-SAME:    }> : (tensor<5x3x4xf32>) -> tensor<3x4x5xf32>

// -----

func.func @make_scatter_dims_leading_in_updates(%dst: tensor<3xf32>,
    %indices: tensor<1x1xi32>, %update: tensor<2x1xf32>) -> tensor<3xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%u: tensor<f32>,  %d: tensor<f32>):
    "mhlo.return"(%u) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3xf32>, tensor<1x1xi32>, tensor<2x1xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}
// CHECK-LABEL: func.func @make_scatter_dims_leading_in_updates(
// CHECK-SAME:      %[[DST:.*]]: tensor<3xf32>,
// CHECK-SAME:      %[[IND:.*]]: tensor<1x1xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<2x1xf32>

// CHECK:         %[[UPD_:.*]] = "mhlo.transpose"(%[[UPD]]) <{
// CHECK-SAME:      permutation = dense<[1, 0]> : tensor<2xi64>
// CHECK-SAME:    }> : (tensor<2x1xf32>) -> tensor<1x2xf32>

// CHECK:         "mhlo.scatter"(%[[DST]], %[[IND]], %[[UPD_]]
// CHECK-SAME:      update_window_dims = [1],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0],
// CHECK-SAME:      index_vector_dim = 1

// -----

func.func @zero_dim_scatter_indices(%dst: tensor<4x4xf32>,
    %indices: tensor<2xi32>, %update: tensor<3x3xf32>) -> tensor<4x4xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%u: tensor<f32>,  %d: tensor<f32>):
    "mhlo.return"(%u) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0, 1],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 0,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<4x4xf32>, tensor<2xi32>, tensor<3x3xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @zero_dim_scatter_indices(
// CHECK-SAME:      %[[DST:.*]]: tensor<4x4xf32>, %[[IND:.*]]: tensor<2xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<3x3xf32>

// CHECK:         %[[IND_:.*]] = tensor.expand_shape %[[IND]] [
// CHECK-SAME:      [0, 1]] output_shape [1, 2] : tensor<2xi32> into tensor<1x2xi32>
// CHECK:         %[[UPD_:.*]] = tensor.expand_shape %[[UPD]] [
// CHECK-SAME:      [0, 1], [2]] output_shape [1, 3, 3] : tensor<3x3xf32> into tensor<1x3x3xf32>
// CHECK:         "mhlo.scatter"(%[[DST]], %[[IND_]], %[[UPD_]])
// CHECK-SAME:      update_window_dims = [1, 2],
// CHECK-SAME:      scatter_dims_to_operand_dims = [0, 1]
// CHECK-SAME:      index_vector_dim = 1

// -----

func.func @multiple_window_and_scatter_dims(
    %dst: tensor<1x2x3x4x5xf32>, %indices: tensor<6x7x2xi32>,
    %updates: tensor<2x6x4x7xf32>) -> tensor<1x2x3x4x5xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %updates) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    mhlo.return %arg3 : tensor<f32>
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      inserted_window_dims = [0, 2, 4],
      update_window_dims = [0, 2],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2
    >, unique_indices = false
  } : (tensor<1x2x3x4x5xf32>, tensor<6x7x2xi32>, tensor<2x6x4x7xf32>) ->
      tensor<1x2x3x4x5xf32>
  return %0 : tensor<1x2x3x4x5xf32>
}

// CHECK-LABEL: @multiple_window_and_scatter_dims(
// CHECK-SAME:      %[[DST:.*]]: tensor<1x2x3x4x5xf32>,
// CHECK-SAME:      %[[IND:.*]]: tensor<6x7x2xi32>,
// CHECK-SAME:      %[[UPD:.*]]: tensor<2x6x4x7xf32>
// CHECK:         %[[IND0:.*]] = tensor.collapse_shape %[[IND]] {{.*}} into tensor<42x2xi32>
// CHECK:         %[[UPD0:.*]] = "mhlo.transpose"(%[[UPD]]) {{.*}} -> tensor<6x7x2x4xf32>
// CHECK:         %[[UPD1:.*]] = tensor.collapse_shape %[[UPD0]] {{.*}} into tensor<42x2x4xf32>
// CHECK:         %[[UPD2:.*]] = tensor.expand_shape %[[UPD1]] {{.*}} into tensor<42x1x2x1x4x1xf32>
// CHECK:         "mhlo.scatter"(%[[DST]], %[[IND0]], %[[UPD2]])
