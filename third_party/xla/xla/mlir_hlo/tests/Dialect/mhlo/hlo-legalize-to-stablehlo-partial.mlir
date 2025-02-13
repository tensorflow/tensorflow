// RUN: mlir-hlo-opt %s -hlo-legalize-to-stablehlo=allow-xla-features --split-input-file | FileCheck %s

func.func @async_ops(%arg1: tensor<128x32xf32>) -> tensor<128x128xf32> attributes {execution_thread = "main"} {
  // CHECK: stablehlo.all_gather
  %0 = "mhlo.all_gather"(%arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  // CHECK: mhlo.async_start
  %0 = "mhlo.async_start"(%arg0) {called_computation = @async_ops, execution_thread = "main"} : (tensor<128x32xf32>) -> !mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>
  // CHECK: mhlo.async_done
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// -----

// CHECK-LABEL: func @copy
func.func @copy() -> tensor<2x1xi32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant dense<{{.*}}> : tensor<2x1xi32>
  // CHECK: %[[COPY:.*]] = mhlo.copy %[[CST]] : tensor<2x1xi32>
  %0 = mhlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %1 = "mhlo.copy"(%0) : (tensor<2x1xi32>) -> tensor<2x1xi32>

  // CHECK: return %[[COPY]]
  func.return %1 : tensor<2x1xi32>
}
