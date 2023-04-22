// RUN: mlir-hlo-opt %s -pass-pipeline='func(mhlo-test-optimize)' | FileCheck %s

// CHECK-LABEL: @gather_is_slice_no_rank
func @gather_is_slice_no_rank(%arg0: tensor<2x1x2xi32>, %arg1: tensor<i64>) -> tensor<1x2xi32> {
  // CHECK: [[CST:%.+]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK: [[SLICE:%.+]] = "mhlo.dynamic-slice"(%arg0, %arg1, [[CST]], [[CST]]) {slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>}
  // CHECK: [[RESHAPE:%.+]] = "mhlo.reshape"([[SLICE]])
   %res = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>
  } : (tensor<2x1x2xi32>, tensor<i64>) -> tensor<1x2xi32>

  // CHECK: return [[RESHAPE]]
  return %res : tensor<1x2xi32>
}

// CHECK-LABEL: @gather_is_slice
func @gather_is_slice(%arg0: tensor<2x1x2xi32>, %arg1: tensor<1xi64>) -> tensor<1x2xi32> {
   // CHECK: [[CST:%.+]] = mhlo.constant dense<0> : tensor<i64>
   // CHECK: [[RESHAPE:%.+]] = "mhlo.reshape"(%arg1)
   // CHECK: [[SLICE:%.+]] = "mhlo.dynamic-slice"(%arg0, [[RESHAPE]], [[CST]], [[CST]]) {slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>}
   // CHECK: [[RES:%.+]] = "mhlo.reshape"([[SLICE]])

   %res = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>
  } : (tensor<2x1x2xi32>, tensor<1xi64>) -> tensor<1x2xi32>

  // CHECK: return [[RES]]
  return %res : tensor<1x2xi32>
}

// CHECK-LABEL: @gather_is_slice_multiple_start_indices
func @gather_is_slice_multiple_start_indices(%arg0: tensor<2x1x2xi32>, %arg1: tensor<2xi64>) -> tensor<1x2xi32> {
  // CHECK-DAG: [[CST:%.+]] = mhlo.constant dense<0>
  // CHECK-DAG: [[SLICE1:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[RESHAPE1:%.+]] = "mhlo.reshape"([[SLICE1]])
  // CHECK-DAG: [[SLICE2:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[RESHAPE2:%.+]] = "mhlo.reshape"([[SLICE2]])
  // CHECK-DAG: [[DSLICE:%.+]] = "mhlo.dynamic-slice"(%arg0, [[RESHAPE1]], [[RESHAPE2]], [[CST]]) {slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>}
  // CHECK-DAG: [[RES:%.+]] = "mhlo.reshape"([[DSLICE]])
   %res = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>
    },
    slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>
  } : (tensor<2x1x2xi32>, tensor<2xi64>) -> tensor<1x2xi32>

  // CHECK: return [[RES]]
  return %res : tensor<1x2xi32>
}
