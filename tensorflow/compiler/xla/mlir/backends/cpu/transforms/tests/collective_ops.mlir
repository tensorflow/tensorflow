// RUN: xla-cpu-opt %s -xla-legalize-collective-ops | FileCheck %s

func.func @max_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
   func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @max_reduce
//  CHECK-SAME: %[[ARG0:.*]]: tensor<10xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<10xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.all_reduce"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:   channel_handle = 5 : i64,
//  CHECK-SAME:   reduction_kind = 3 : i32,
//  CHECK-SAME:   replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
//  CHECK-SAME:   use_global_device_ids = 1
//       CHECK: return %[[RET]]

func.func @and_reduce(%arg0: tensor<1xi1>) -> tensor<1xi1> {
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<i1>, %rhs: tensor<i1>):
    %1 = mhlo.and %lhs, %rhs : tensor<i1>
    mhlo.return %1 : tensor<i1>
  }) {
    replica_groups = dense<> : tensor<0x0xi64>
  } : (tensor<1xi1>) -> tensor<1xi1>
  func.return %0 : tensor<1xi1>
}

// CHECK-LABEL: @and_reduce
//       CHECK:   reduction_kind = 2 : i32,

func.func @or_reduce(%arg0: tensor<1xi1>) -> tensor<1xi1> {
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<i1>, %rhs: tensor<i1>):
    %1 = mhlo.or %lhs, %rhs : tensor<i1>
    mhlo.return %1 : tensor<i1>
  }) {
    replica_groups = dense<> : tensor<0x0xi64>
  } : (tensor<1xi1>) -> tensor<1xi1>
  func.return %0 : tensor<1xi1>
}

// CHECK-LABEL: @or_reduce
//       CHECK:   reduction_kind = 3 : i32,

func.func @min_reduce_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.minimum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<> : tensor<0x0xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<?xf32>) -> tensor<?xf32>
   func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @min_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[DIM]])
//       CHECK: "xla_cpu.all_reduce"(%[[ARG0]], %[[DST]])
//  CHECK-SAME:   reduction_kind = 2
//  CHECK-SAME:   use_global_device_ids = 0

func.func @partition_id() -> tensor<ui32> {
  %0 = "mhlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// CHECK-LABEL: @partition_id
// CHECK: %[[ID:.*]] = "xla_cpu.partition_id"() : () -> i32
// CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ID]] : tensor<i32>
// CHECK: %[[CAST:.*]] = mhlo.convert %[[TENSOR]] : (tensor<i32>) -> tensor<ui32>
// CHECK: return %[[CAST]]

func.func @replica_id() -> tensor<ui32> {
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// CHECK-LABEL: @replica_id
// CHECK: %[[ID:.*]] = "xla_cpu.replica_id"() : () -> i32
// CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ID]] : tensor<i32>
// CHECK: %[[CAST:.*]] = mhlo.convert %[[TENSOR]] : (tensor<i32>) -> tensor<ui32>
// CHECK: return %[[CAST]]

func.func @collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: @collective_permute
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16x8xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<16x8xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.collective_permute"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:    channel_handle = 1
//  CHECK-SAME:    source_target_pairs = dense<
//       CHECK: return %[[RET]]

func.func @collective_permute_dynamic(%arg0: tensor<16x?xf32>)
    -> tensor<16x?xf32> {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x?xf32>) -> tensor<16x?xf32>
  func.return %0 : tensor<16x?xf32>
}

// CHECK-LABEL: @collective_permute_dynamic
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16x?xf32>
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[DIM]]) : tensor<16x?xf32>
//       CHECK: "xla_cpu.collective_permute"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    channel_handle = #mhlo.channel_handle<handle = 2, type = 0>,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// CHECK-LABEL: @all_to_all
//  CHECK-SAME: %[[ARG0:.*]]: tensor<4x16xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<16x4xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:    channel_id_present = 1
//  CHECK-SAME:    concat_dimension = 0
//  CHECK-SAME:    op_id = 2
//  CHECK-SAME:    replica_groups = dense<
//  CHECK-SAME:    split_count = 4
//  CHECK-SAME:    split_dimension = 1
//       CHECK: return %[[RET]]

func.func @all_to_all_dynamic_concat_dim(%arg0: tensor<?x16xf32>)
    -> tensor<?x4xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<?x16xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @all_to_all_dynamic_concat_dim
//  CHECK-SAME: %[[ARG0:.*]]: tensor<?x16xf32>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK: %[[CONCAT_DIM:.*]] = arith.muli %[[DIM]], %[[C4]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[CONCAT_DIM]]) : tensor<?x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all_dynamic_split_dim(%arg0: tensor<4x?xf32>)
    -> tensor<16x?xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x?xf32>) -> tensor<16x?xf32>
  func.return %0 : tensor<16x?xf32>
}

// CHECK-LABEL: @all_to_all_dynamic_split_dim
//  CHECK-SAME: %[[ARG0:.*]]: tensor<4x?xf32>
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK: %[[CONCAT_DIM:.*]] = arith.divui %[[DIM]], %[[C4]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[CONCAT_DIM]]) : tensor<16x?xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>)
    -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

// CHECK-LABEL: @all_to_all_tuple
//  CHECK-SAME: %[[ARG0:.*]]: tensor<128x4xf32>,
//  CHECK-SAME: %[[ARG1:.*]]: tensor<128x4xf32>
//       CHECK: %[[DST0:.*]] = tensor.empty() : tensor<128x4xf32>
//       CHECK: %[[DST1:.*]] = tensor.empty() : tensor<128x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[ARG1]], %[[DST0]], %[[DST1]])

func.func @outfeed_0_input(%token: !mhlo.token) -> !mhlo.token {
  %res = "mhlo.outfeed"(%token) {outfeed_config = "foobar"} : (!mhlo.token) -> !mhlo.token
  func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_0_input
//       CHECK: "xla_cpu.outfeed"() {config = "foobar", result_type = []} : () -> ()

func.func @outfeed_1_input(%data: tensor<2xui32>, %token: !mhlo.token)
  -> !mhlo.token attributes {xlaframework.result_mapping = 1 : i32} {
    %res = "mhlo.outfeed"(%data, %token) {
      outfeed_config = "", xla_shape = "token[]"
      } : (tensor<2xui32>, !mhlo.token) -> !mhlo.token
    func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_1_input
//  CHECK-SAME: %[[DATA:.*]]: tensor<2xui32>
//  CHECK-SAME: %[[TOKEN:.*]]: !mhlo.token
//       CHECK: "xla_cpu.outfeed"(%[[DATA]]) {config = "", result_type = [ui32]} : (tensor<2xui32>) -> ()
//       CHECK: return %[[TOKEN]] : !mhlo.token

func.func @outfeed_2_input(%data1: tensor<3xui32>, %data2: tensor<3xi32>, %token: !mhlo.token) -> !mhlo.token {
  %res = "mhlo.outfeed"(%data1, %data2,  %token) {outfeed_config = "foobar"}
    : (tensor<3xui32>, tensor<3xi32>, !mhlo.token) -> !mhlo.token
  func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_2_input
//  CHECK-SAME: %[[ARG0:.*]]: tensor<3xui32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<3xi32>
//       CHECK: "xla_cpu.outfeed"(%[[ARG0]], %[[ARG1]]) {config = "foobar", result_type = [ui32, i32]}
//  CHECK-SAME: (tensor<3xui32>, tensor<3xi32>)

func.func @add_dependency(%arg0: tensor<16xf32>, %arg1: !mhlo.token) -> tensor<16xf32> {
  %0 = "mhlo.add_dependency"(%arg0, %arg1) : (tensor<16xf32>, !mhlo.token) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// CHECK-LABEL: @add_dependency
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>
//  CHECK-SAME: %[[ARG1:.*]]: !mhlo.token
//       CHECK: %[[RES:.*]] = "xla_cpu.add_dependency"
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<16xf32>