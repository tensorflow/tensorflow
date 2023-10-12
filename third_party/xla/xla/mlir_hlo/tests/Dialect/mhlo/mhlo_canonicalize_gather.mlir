// RUN: mlir-hlo-opt %s --split-input-file --hlo-canonicalize-gather | FileCheck %s --dump-input=always

func.func @transform_start_indices(%operand: tensor<33x34xf32>,
    %indices: tensor<42x43xi32>) -> tensor<42x43x7x8xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 2,
      offset_dims = [2, 3],
      start_index_map = [0]
    >,
    slice_sizes = dense<[7, 8]> : tensor<2xi64>
  } : (tensor<33x34xf32>, tensor<42x43xi32>) -> tensor<42x43x7x8xf32>
  return %gather : tensor<42x43x7x8xf32>
}

// CHECK-LABEL: func @transform_start_indices
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<33x34xf32>
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<42x43xi32>
//       CHECK:   %[[WITH_IVD:.*]] = tensor.expand_shape %[[INDICES]] 
//  CHECK-SAME:     into tensor<42x43x1xi32>
//       CHECK:   %[[FLATTENED:.*]] = tensor.collapse_shape %[[WITH_IVD]]
//  CHECK-SAME:     into tensor<1806x1xi32>
//       CHECK:   %[[GATHER:.*]] = "mhlo.gather"(%[[OPERAND]], %[[FLATTENED]])
//  CHECK-SAME:     offset_dims = [1, 2]
//  CHECK-SAME:     index_vector_dim = 1
//       CHECK:   %[[RESULT:.*]] = tensor.expand_shape %[[GATHER]]
//  CHECK-SAME:     into tensor<42x43x7x8xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @remove_collapsed_slice_dims(%operand: tensor<33x34xf32>,
    %indices: tensor<42x1xi32>) -> tensor<42xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [],
      start_index_map = [0]
    >,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<33x34xf32>, tensor<42x1xi32>) -> tensor<42xf32>
  return %gather : tensor<42xf32>
}

// CHECK-LABEL: func @remove_collapsed_slice_dims
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<33x34xf32>
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<42x1xi32>
//       CHECK:   %[[GATHER:.*]] = "mhlo.gather"(%[[OPERAND]], %[[INDICES]])
//  CHECK-SAME:     offset_dims = [1, 2]
//   CHECK-NOT:     collapsed_slice_dims
//       CHECK:   %[[RESULT:.*]] = tensor.collapse_shape %[[GATHER]]
//  CHECK-SAME:     into tensor<42xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @permuted_start_index_map(%operand: tensor<33x34x35xf32>,
    %indices: tensor<42x3xi32>) -> tensor<42x1x2x3xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 1,
      offset_dims = [1, 2, 3],
      start_index_map = [2, 0, 1]
    >,
    slice_sizes = dense<[1, 2, 3]> : tensor<3xi64>
  } : (tensor<33x34x35xf32>, tensor<42x3xi32>) -> tensor<42x1x2x3xf32>
  return %gather : tensor<42x1x2x3xf32>
}

// CHECK-LABEL: func @permuted_start_index_map
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<33x34x35xf32>
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<42x3xi32>
//       CHECK:   %[[TRANSPOSED:.*]] = "mhlo.transpose"(%[[OPERAND]])
//  CHECK-SAME:     -> tensor<35x33x34xf32>
//       CHECK:   %[[GATHER:.*]] = "mhlo.gather"(%[[TRANSPOSED]], %[[INDICES]])
//  CHECK-SAME:     start_index_map = [0, 1, 2]
//  CHECK-SAME:     slice_sizes = dense<[3, 1, 2]>
//       CHECK:   %[[RESULT:.*]] = "mhlo.transpose"(%[[GATHER]])
//  CHECK-SAME:     -> tensor<42x1x2x3xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @collapse_some_dims(%operand: tensor<33x34x35xf32>,
    %indices: tensor<42x1xi32>) -> tensor<7x42xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 1,
      offset_dims = [0],
      collapsed_slice_dims = [0, 2],
      start_index_map = [0]
    >,
    slice_sizes = dense<[1, 7, 1]> : tensor<3xi64>
  } : (tensor<33x34x35xf32>, tensor<42x1xi32>) -> tensor<7x42xf32>
  return %gather : tensor<7x42xf32>
}

// CHECK-LABEL: func @collapse_some_dims
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<33x34x35x
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<42x1x
//       CHECK:   %[[GATHER:.*]] = "mhlo.gather"(%[[OPERAND]], %[[INDICES]])
//  CHECK-SAME:     -> tensor<42x1x7x1xf32>
//       CHECK:   %[[COLLAPSED:.*]] = tensor.collapse_shape %[[GATHER]]
//  CHECK-SAME:     into tensor<42x7xf32>
//       CHECK:   %[[RESULT:.*]] = "mhlo.transpose"(%[[COLLAPSED]])
//  CHECK-SAME:     -> tensor<7x42xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @no_batch_dims(%operand: tensor<8x16xf32>, %indices: tensor<2xi32>)
    -> tensor<8x16xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 1]
    >,
    slice_sizes = dense<[8, 16]> : tensor<2xi64>
  } : (tensor<8x16xf32>, tensor<2xi32>) -> tensor<8x16xf32>
  return %gather : tensor<8x16xf32>
}

// CHECK-LABEL: func @no_batch_dims
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<8x16x
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<2x
//       CHECK:   %[[EXPANDED:.*]] = tensor.expand_shape %[[INDICES]]
//  CHECK-SAME:     into tensor<1x2xi32>
//       CHECK:   %[[GATHER:.*]] = "mhlo.gather"(%[[OPERAND]], %[[EXPANDED]])
//  CHECK-SAME:     offset_dims = [1, 2]
//  CHECK-SAME:     index_vector_dim = 1
//       CHECK:   %[[RESULT:.*]] = tensor.collapse_shape %[[GATHER]]
//  CHECK-SAME:     into tensor<8x16xf32>
//       CHECK:   return %[[RESULT]]

// -----

func.func @zero_slice(%operand: tensor<8x16xf32>, %indices: tensor<2xi32>)
    -> tensor<8x0xf32> {
  %gather = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 1]
    >,
    slice_sizes = dense<[8, 0]> : tensor<2xi64>
  } : (tensor<8x16xf32>, tensor<2xi32>) -> tensor<8x0xf32>
  return %gather : tensor<8x0xf32>
}

// CHECK-LABEL: func @zero_slice
//  CHECK-SAME:   %[[OPERAND:.*]]: tensor<8x16x
//  CHECK-SAME:   %[[INDICES:.*]]: tensor<2x
//   CHECK-DAG:   %[[CST:.*]] = arith.constant dense<0
//       CHECK:   return %[[CST]] : tensor<8x0xf32>