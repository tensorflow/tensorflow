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

// -----

// CHECK-LABEL: func @all_to_all_tuple
func.func @all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  // CHECK: mhlo.all_to_all
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_schedule
func.func @custom_call_schedule(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: mhlo.custom_call
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    custom_call_schedule = #mhlo<custom_call_schedule EARLIEST>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Tokens flow between StableHLO and MHLO ops, so need to have special converson
// logic. AddDependencyOp is the only op that doesn't exist in StableHLO but
// uses token types, so it can have either StableHLO or MHLO token types as
// input.

// CHECK-LABEL: func @add_dependency
func.func @add_dependency(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK:      %[[TOK:.*]] = stablehlo.create_token {{.*}} : !stablehlo.token
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.add_dependency %arg0, %[[TOK]] : (tensor<3x4xf32>, !stablehlo.token) -> tensor<3x4xf32>
  %0 = mhlo.create_token {xla_shape = "token[]"} : !mhlo.token
  %1 = mhlo.add_dependency %arg0, %0 : (tensor<3x4xf32>, !mhlo.token) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}
