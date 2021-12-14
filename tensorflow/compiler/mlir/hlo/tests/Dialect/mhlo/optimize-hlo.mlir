// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.func(mhlo-test-optimize)' | FileCheck %s --dump-input-context=30

// CHECK-LABEL: @gather_is_slice_no_rank
func @gather_is_slice_no_rank(%arg0: tensor<2x1x2xi32>, %arg1: tensor<i64>) -> tensor<1x2xi32> {
  // CHECK: [[CST:%.+]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK: [[SLICE:%.+]] = "mhlo.dynamic-slice"(%arg0, %arg1, [[CST]], [[CST]]) {slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>}
  // CHECK: [[RESHAPE:%.+]] = "mhlo.reshape"([[SLICE]])
   %res = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
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
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>
  } : (tensor<2x1x2xi32>, tensor<1xi64>) -> tensor<1x2xi32>

  // CHECK: return [[RES]]
  return %res : tensor<1x2xi32>
}
