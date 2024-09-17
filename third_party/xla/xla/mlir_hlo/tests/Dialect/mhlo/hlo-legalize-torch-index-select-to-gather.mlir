// RUN: mlir-hlo-opt -mhlo-legalize-torch-index-select-to-gather -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @index_select_to_gather_convert_index_type
func.func @index_select_to_gather_convert_index_type(%arg0 : tensor<5x1x5xi64>, %arg1 : tensor<2xi64>) -> tensor<2x1x5xi64> {
  // CHECK: [[ARG1:%.+]] = mhlo.convert %arg1 : (tensor<2xi64>) -> tensor<2xui32>
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, [[ARG1]]) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [1, 2],
  // CHECK-SAME:     collapsed_slice_dims = [0],
  // CHECK-SAME:     start_index_map = [0],
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
  // CHECK-SAME: }> : (tensor<5x1x5xi64>, tensor<2xui32>) -> tensor<2x1x5xi64>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x1x5xi64>, tensor<2xi64>) -> tensor<2x1x5xi64>
  // CHECK: return [[RES]] : tensor<2x1x5xi64>
  func.return %0 : tensor<2x1x5xi64>
}

// -----

// CHECK-LABEL: @index_select_to_gather_multi_offset_dims
func.func @index_select_to_gather_multi_offset_dims(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [1, 2],
  // CHECK-SAME:     collapsed_slice_dims = [0],
  // CHECK-SAME:     start_index_map = [0],
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
  // CHECK-SAME: }> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  // CHECK: return [[RES]] : tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

// CHECK-LABEL: @index_select_to_gather_larger_output
func.func @index_select_to_gather_larger_output(%arg0 : tensor<5x4xf32>, %arg1 : tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32> {
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [3],
  // CHECK-SAME:     collapsed_slice_dims = [0],
  // CHECK-SAME:     start_index_map = [0],
  // CHECK-SAME:     index_vector_dim = 3
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 4]> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x4xf32>, tensor<1x3x1xi32>) -> tensor<1x3x1x4xf32>
  // CHECK: return [[RES]] : tensor<1x3x1x4xf32>
  func.return %0 : tensor<1x3x1x4xf32>
}

// -----

// CHECK-LABEL: @index_select_to_gather_regular_map
func.func @index_select_to_gather_regular_map(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi32>) -> tensor<2x4xi32> {
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [1],
  // CHECK-SAME:     collapsed_slice_dims = [0],
  // CHECK-SAME:     start_index_map = [0],
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 4]> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<2x4xi32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<2x4xi32>
  // CHECK: return [[RES]] : tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: @index_select_to_gather_reverse_map
func.func @index_select_to_gather_reverse_map(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi32>) -> tensor<3x2xi32> {
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, %arg1) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [0],
  // CHECK-SAME:     collapsed_slice_dims = [1],
  // CHECK-SAME:     start_index_map = [1],
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[3, 1]> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 1 : i64,
    batch_dims = 0 : i64
  }> : (tensor<3x4xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  // CHECK: return [[RES]] : tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: @index_select_to_gather_batch_dim_greater_than_1
func.func @index_select_to_gather_batch_dim_greater_than_1(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x5xi32> {
  // CHECK: [[ARG0:%.+]] = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2x1xi32>
  // CHECK: [[ARG1:%.+]] = mhlo.reshape %arg1 : (tensor<2xi32>) -> tensor<2x1xi32>
  // CHECK: [[ARG2:%.+]] = "mhlo.concatenate"([[ARG0]], [[ARG1]]) <{dimension = 1 : i64}> : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>
  // CHECK: [[RES:%.+]] = "mhlo.gather"(%arg0, [[ARG2]]) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [1],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 1
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
  // CHECK-SAME: }> : (tensor<5x1x5xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 1 : i64,
    batch_dims = 1 : i64
  }> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

func.func @index_select_to_gather_non_static_operand(%arg0 : tensor<5x1x?xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  // CHECK: mhlo.torch_index_select
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x1x?xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

func.func @index_select_to_gather_non_static_index(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<?xi32>) -> tensor<2x1x5xi32> {
  // CHECK: mhlo.torch_index_select
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x1x5xi32>, tensor<?xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

func.func @index_select_to_gather_dim_less_than_batch_dims(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xi32>) -> tensor<2x1x5xi32> {
  // CHECK: mhlo.torch_index_select
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 1 : i64
  }> : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}

// -----

func.func @index_select_to_gather_non_integer_index(%arg0 : tensor<5x1x5xi32>, %arg1 : tensor<2xf32>) -> tensor<2x1x5xi32> {
  // CHECK: mhlo.torch_index_select
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) <{
    dim = 0 : i64,
    batch_dims = 0 : i64
  }> : (tensor<5x1x5xi32>, tensor<2xf32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}
