// RUN: xla-cpu-opt %s -split-input-file -empty-tensor-to-alloc-tensor \
// RUN:   -one-shot-bufferize | FileCheck %s

func.func @max_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = "xla_cpu.all_reduce"(%arg0, %0) {
    channel_handle = 5 : i64,
    reduction_kind = 3 : i32,
    replica_groups = dense<[]> : tensor<0xi64>
  } : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @max_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<10xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<10xf32>
//       CHECK: "xla_cpu.all_reduce"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   channel_handle = 5
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]