// RUN: mlir-hlo-opt %s -hlo-legalize-to-stablehlo=allow-xla-features --split-input-file | FileCheck %s

// CHECK-LABEL: @recv
func.func @recv(%arg0: !mhlo.token) -> (tensor<3x4xi32>, !mhlo.token) attributes {execution_thread = "main"} {
  // CHECK: stablehlo.recv{{.*}}!stablehlo.token
  %0:2 = "mhlo.recv"(%arg0) <{channel_handle = #mhlo.channel_handle<handle = 5, type = 3>, is_host_transfer = true}> : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
  return %0#0, %0#1 : tensor<3x4xi32>, !mhlo.token
}

// CHECK-LABEL: @async_ops_with_token
func.func @async_ops_with_token(%arg0: !mhlo.token) -> (tensor<3x4xi32>, !mhlo.token) {
  // CHECK: mhlo.async_start{{.*}} !mhlo.async_bundle<!stablehlo.token, tuple<tensor<3x4xi32>, !stablehlo.token>, tensor<i32>>
  %0 = "mhlo.async_start"(%arg0) <{called_computation = @recv, execution_thread = "main"}> : (!mhlo.token) -> !mhlo.async_bundle<!mhlo.token, tuple<tensor<3x4xi32>, !mhlo.token>, tensor<i32>>
  // CHECK: mhlo.async_done{{.*}} -> (tensor<3x4xi32>, !stablehlo.token)
  %1:2 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<!mhlo.token, tuple<tensor<3x4xi32>, !mhlo.token>, tensor<i32>>) -> (tensor<3x4xi32>, !mhlo.token)
  return %1#0, %1#1 : tensor<3x4xi32>, !mhlo.token
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
