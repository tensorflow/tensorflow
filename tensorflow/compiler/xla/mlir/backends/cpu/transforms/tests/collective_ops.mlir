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
//       CHECK: %[[RET:.*]] = "xla_cpu.all_reduce"(%[[ARG0]]) {
//  CHECK-SAME:   channel_handle = 5 : i64,
//  CHECK-SAME:   reduction_kind = 3 : i32,
//  CHECK-SAME:   replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
//  CHECK-SAME:   use_global_device_ids
//       CHECK: return %[[RET]]

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
