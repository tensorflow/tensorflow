// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-export='enable-hlo-sharding-v3=true' 2>&1 | FileCheck %s

// Tests exporting `sdy.manual_computation` under HloShardingV3 (`enable-hlo-sharding-v3=true`).
// Verifies that:
// 1. Shard map boundaries (GlobalToLocalShape and LocalToGlobalShape) receive native
//    `sdy.sharding` and typed `xla.sdy.manual_axes` attributes directly.
// 2. Legacy string `mhlo.frontend_attributes` (xla.sdy.in_shardings, etc.) are NOT emitted.

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

// Ensure string frontend attributes are completely absent in V3 export:
// CHECK-NOT: xla.sdy.in_shardings
// CHECK-NOT: xla.sdy.out_shardings

// Tests basic export: GlobalToLocalShape gets local shardings and manual axes; LocalToGlobalShape gets global shardings.
// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]]:2 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0, %arg1)

  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{}, {}]>, <mesh<["a"=4, "b"=2]>, [{}, {}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>
  // CHECK-SAME:   (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK-NEXT: %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body(%[[GLOBAL_TO_LOCAL]]#0, %[[GLOBAL_TO_LOCAL]]#1)
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{"a"}, {}], replicated={"b"}>]>
  // CHECK-SAME:   (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// Tests chained manual computations where the output of one feeds the input of another.
// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"b"}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL_0:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{}, {}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"a"}>
  // CHECK-SAME:   (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP_0:.*]] = call @xla.sdy.manual_computation_body_0(%[[GLOBAL_TO_LOCAL_0]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL_0:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP_0]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{"a"}, {}]>]>
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL_1:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%[[LOCAL_TO_GLOBAL_0]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{}, {}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"b"}>
  // CHECK-SAME:   (tensor<8x8xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[SHMAP_1:.*]] = call @xla.sdy.manual_computation_body_1(%[[GLOBAL_TO_LOCAL_1]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL_1:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP_1]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{}, {"b"}]>]>
  // CHECK-SAME:   (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL_1]] : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"b"}]>] out_shardings=[<@mesh_0, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Tests nested manual computations: an inner shard_map inside an outer shard_map body.
// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"a"}>
  // CHECK-SAME:   (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body_3(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{"a"}, {}]>]>
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    sdy.return %1 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Tests nested manual computation where the outer body performs additional math on the inner result.
// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"a"}>
  // CHECK-SAME:   (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body_5(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{"a"}, {}]>]>
  // CHECK-SAME:   (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %3 = stablehlo.add %1, %1 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Edge case: Manual computation producing an output but taking no inputs.
// CHECK-LABEL: func @manual_computation_no_inputs
func.func @manual_computation_no_inputs() -> tensor<4xi64> {
  // CHECK-NEXT: %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body_6()
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   () -> tensor<2xi64>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{"b"}]>]>
  // CHECK-SAME:   (tensor<2xi64>) -> tensor<4xi64>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4xi64>
  %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh_0, [{"b"}]>] manual_axes={"b"} () {
    %1 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
    sdy.return %1 : tensor<2xi64>
  } : () -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// Edge case: Manual computation consuming an input but producing no outputs.
// CHECK-LABEL: func @manual_computation_no_outputs
func.func @manual_computation_no_outputs(%arg0: tensor<4xi64>) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, [{}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"b"}>
  // CHECK-SAME:   (tensor<4xi64>) -> tensor<2xi64>
  // CHECK-NEXT: call @xla.sdy.manual_computation_body_7(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (tensor<2xi64>) -> ()
  // CHECK-NEXT: return
  sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"b"}]>] out_shardings=[] manual_axes={"b"} (%arg1: tensor<2xi64>) {
    stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<2xi64>) -> ()
    sdy.return
  } : (tensor<4xi64>) -> ()
  func.return
}

// Edge case: Degenerate manual computation with no inputs and no outputs.
// CHECK-LABEL: func @manual_computation_no_inputs_no_outputs
func.func @manual_computation_no_inputs_no_outputs() {
  // CHECK-NEXT: call @xla.sdy.manual_computation_body_8() {mhlo.frontend_attributes = {inlineable = "xla_late"}} : () -> ()
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    sdy.return
  } : () -> ()
  func.return
}

// Type preservation: Token arguments should not have manual axes attached in V3.
// CHECK-LABEL: func @manual_computation_tokens
func.func @manual_computation_tokens(%arg0: !stablehlo.token, %arg1: tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]]:2 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0, %arg1)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, []>, <mesh<["a"=4, "b"=2]>, [{}]>]>
  // CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"b"}>
  // CHECK-SAME:   (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<1xi64>)
  // CHECK-NEXT: %[[SHMAP:.*]]:2 = call @xla.sdy.manual_computation_body_9(%[[GLOBAL_TO_LOCAL]]#0, %[[GLOBAL_TO_LOCAL]]#1)
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "xla_late"}}
  // CHECK-SAME:   (!stablehlo.token, tensor<1xi64>) -> (!stablehlo.token, tensor<1xi64>)
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]]:2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]]#0, %[[SHMAP]]#1)
  // CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4, "b"=2]>, []>, <mesh<["a"=4, "b"=2]>, [{"b"}]>]>
  // CHECK-SAME:   (!stablehlo.token, tensor<1xi64>) -> (!stablehlo.token, tensor<2xi64>)
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]]#0, %[[LOCAL_TO_GLOBAL]]#1 : !stablehlo.token, tensor<2xi64>
  %0:2 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>] out_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>] manual_axes={"b"} (%arg2: !stablehlo.token, %arg3: tensor<1xi64>) {
    sdy.return %arg2, %arg3 : !stablehlo.token, tensor<1xi64>
  } : (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>)
  func.return %0#0, %0#1 : !stablehlo.token, tensor<2xi64>
}


// Tests that outlined body functions for nested manual computations have correct HloShardingV3 attributes on inner custom calls.
// CHECK-LABEL: func @xla.sdy.manual_computation_body_3(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
// CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {}]>]>
// CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"b"}>
// CHECK-SAME:     (tensor<2x8xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body_2(%[[GLOBAL_TO_LOCAL]])
// CHECK-SAME:     {mhlo.frontend_attributes = {inlineable = "xla_late"}}
// CHECK-SAME:     (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
// CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {"b"}]>]>
// CHECK-SAME:     (tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   return %[[LOCAL_TO_GLOBAL]] : tensor<2x8xf32>

// CHECK-LABEL: func @xla.sdy.manual_computation_body_5(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0)
// CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {}]>]>
// CHECK-SAME{LITERAL}: xla.sdy.manual_axes = #sdy<manual_axes{"b"}>
// CHECK-SAME:     (tensor<2x8xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[SHMAP:.*]] = call @xla.sdy.manual_computation_body_4(%[[GLOBAL_TO_LOCAL]])
// CHECK-SAME:     {mhlo.frontend_attributes = {inlineable = "xla_late"}}
// CHECK-SAME:     (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[SHMAP]])
// CHECK-SAME{LITERAL}: sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=2, "b"=2, "c"=2, "d"=2]>, [{}, {"b"}]>]>
// CHECK-SAME:     (tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LOCAL_TO_GLOBAL]], %[[LOCAL_TO_GLOBAL]] : tensor<2x8xf32>
// CHECK-NEXT:   return %[[ADD]] : tensor<2x8xf32>
