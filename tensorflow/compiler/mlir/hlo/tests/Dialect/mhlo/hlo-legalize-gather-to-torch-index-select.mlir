// RUN: mlir-hlo-opt -mhlo-legalize-gather-to-torch-index-select %s -o - | FileCheck %s

// CHECK-LABEL: @gather_to_index_select
func.func @gather_to_index_select(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x4xf32> {
  // CHECK: [[TIS:%.+]] = "mhlo.torch_index_select"(%arg0, %arg1) {
  // CHECK-SAME:   batch_dims = 0 : i64,
  // CHECK-SAME:   dim = 0 : i64
  // CHECK-SAME: } : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
  // CHECK: [[RES:%.+]] = "mhlo.reshape"([[TIS]])
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x4xf32>

  // CHECK: return [[RES]]
  func.return %0 : tensor<1x3x4xf32>
}

// CHECK-LABEL: @gather_no_lowering_subslice
func.func @gather_no_lowering_subslice(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x3xf32> {
  // CHECK: "mhlo.gather"
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 3]> : tensor<2xi64>
  } : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x3xf32>
  func.return %0 : tensor<1x3x3xf32>
}

// CHECK-LABEL: @gather_no_lowering_multidim
func.func @gather_no_lowering_multidim(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x2xi32>) -> tensor<1x3x4xf32> {
  // CHECK: "mhlo.gather"
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<5x4xf32>, tensor<1x3x2xi32>) -> tensor<1x3x4xf32>
  func.return %0 : tensor<1x3x4xf32>
}
