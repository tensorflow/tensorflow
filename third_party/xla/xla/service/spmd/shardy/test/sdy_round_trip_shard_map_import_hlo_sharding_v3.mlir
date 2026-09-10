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
// RUN: sdy_opt %s --split-input-file -xla-sdy-test-flatten-call-graph -xla-sdy-round-trip-shard-map-import 2>&1 | FileCheck %s

// Tests importing shard map custom calls back into `sdy.manual_computation` under
// HloShardingV3 (where frontend_attributes are omitted).
// Verifies that:
// 1. Shardings and manual axes are reconstructed from custom call attributes and shape ratios.
// 2. Outlined manual computation call bodies are inlined back into `sdy.manual_computation`.

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

// Tests manual computation with multiple inputs/outputs, all-reduce, and sharding reconstruction.
// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  // CHECK-NOT: call @xla.sdy.manual_computation_body
  // CHECK:               %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>]

  // CHECK-SAME{LITERAL}:     manual_axes={"a", "b"}
  // CHECK-SAME:              (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            %[[ADD_0:.*]] = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
  // CHECK-NEXT:            %[[DOT:.*]] = stablehlo.dot %[[ADD_0]], %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            %[[REDUCE:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  // CHECK-NEXT:            ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
  // CHECK-NEXT:              %[[ADD_1:.*]] = stablehlo.add %arg4, %arg5 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD_1]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            sdy.return %[[REDUCE]] : tensor<2x32xf32>
  // CHECK-NEXT:          } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:          return %[[MAN_COMP]] : tensor<8x32xf32>
  %0:2 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0, %arg1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>, <@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  %1 = call @xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}], replicated={"b"}>]>} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %2 : tensor<8x32xf32>
}

// Tests manual computation inlining when function name does not have a standard prefix or suffix.
// CHECK-LABEL: func @single_manual_comp_name_is_not_prefix_nor_suffix
func.func @single_manual_comp_name_is_not_prefix_nor_suffix(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
  // CHECK-NOT: call @my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234
  // CHECK:               %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            sdy.return %arg1 : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8x8xf32>) -> tensor<2x8xf32>
  %1 = call @my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// Tests chained manual computations: outputs of first shard map feed second shard map.
// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NOT: call @xla.sdy.manual_computation_body_0
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            sdy.return %arg1 : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NOT: call @xla.sdy.manual_computation_body_1
  // CHECK-NEXT:          %[[MAN_COMP_1:.*]] = sdy.manual_computation(%[[MAN_COMP_0]])
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME:              (%arg1: tensor<8x4xf32>) {
  // CHECK-NEXT:            sdy.return %arg1 : tensor<8x4xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_1]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8x8xf32>) -> tensor<2x8xf32>
  %1 = call @xla.sdy.manual_computation_body_0(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<8x8xf32>
  %3 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%2) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<8x8xf32>) -> tensor<8x4xf32>
  %4 = call @xla.sdy.manual_computation_body_1(%3) : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %5 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%4) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"b"}]>]>} : (tensor<8x4xf32>) -> tensor<8x8xf32>
  return %5 : tensor<8x8xf32>
}

// Tests nested manual computations: reconstructing inner and outer sdy.manual_computation regions.
// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: call @xla.sdy.manual_computation_body_2
  // CHECK-NOT: call @xla.sdy.manual_computation_body_3
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            %[[MAN_COMP_1:.*]] = sdy.manual_computation(%arg1)
  // CHECK-SAME{LITERAL}:       in_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       out_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       manual_axes={"b"}
  // CHECK-SAME:                (%arg2: tensor<2x4xf32>) {
  // CHECK-NEXT:              %[[MUL:.*]] = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
  // CHECK-NEXT:              sdy.return %[[MUL]] : tensor<2x4xf32>
  // CHECK-NEXT:            } : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT:            sdy.return %[[MAN_COMP_1]] : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_0]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  %1 = call @xla.sdy.manual_computation_body_3(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

// Tests nested manual computation with operations in the outer region around the inner region.
// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: call @xla.sdy.manual_computation_body_4
  // CHECK-NOT: call @xla.sdy.manual_computation_body_5
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            %[[MAN_COMP_1:.*]] = sdy.manual_computation(%arg1)
  // CHECK-SAME{LITERAL}:       in_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       out_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       manual_axes={"b"}
  // CHECK-SAME:                (%arg2: tensor<2x4xf32>) {
  // CHECK-NEXT:              %[[MUL:.*]] = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
  // CHECK-NEXT:              sdy.return %[[MUL]] : tensor<2x4xf32>
  // CHECK-NEXT:            } : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %[[MAN_COMP_1]], %[[MAN_COMP_1]] : tensor<2x8xf32>
  // CHECK-NEXT:            sdy.return %[[ADD]] : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_0]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  %1 = call @xla.sdy.manual_computation_body_5(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}

// Tests manual computation with no inputs: manual axes are preserved on LocalToGlobalShape.
// CHECK-LABEL: func @manual_computation_no_inputs
func.func @manual_computation_no_inputs() -> tensor<4xi64> {

  // CHECK-NOT: call @xla.sdy.manual_computation_body_6
  // CHECK:               %[[SHMAP:.*]] = sdy.manual_computation()
  // CHECK-SAME{LITERAL}:     in_shardings=[]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"b"}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME{LITERAL}:     () {
  // CHECK-NEXT:            %[[C:.*]] = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
  // CHECK-NEXT:            sdy.return %[[C]] : tensor<2xi64>
  // CHECK-NEXT:          } : () -> tensor<4xi64>
  // CHECK-NEXT:          return %[[SHMAP]] : tensor<4xi64>
  %0 = call @xla.sdy.manual_computation_body_6() : () -> tensor<2xi64>
  %1 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : (tensor<2xi64>) -> tensor<4xi64>
  return %1 : tensor<4xi64>
}

// Edge case: Manual computation consuming an input but producing no outputs.
// CHECK-LABEL: func @manual_computation_no_outputs
func.func @manual_computation_no_outputs(%arg0: tensor<4xi64>) {
  // CHECK-NOT: call @xla.sdy.manual_computation_body_7
  // CHECK:               sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"b"}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME{LITERAL}:     (%arg1: tensor<2xi64>) {
  // CHECK-NEXT:            stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<2xi64>) -> ()
  // CHECK-NEXT:            sdy.return
  // CHECK-NEXT:          } : (tensor<4xi64>) -> ()
  // CHECK-NEXT:          return
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<4xi64>) -> tensor<2xi64>
  call @xla.sdy.manual_computation_body_7(%0) : (tensor<2xi64>) -> ()
  return
}

// Edge case: Degenerate manual computation with no inputs and no outputs.
// CHECK-LABEL: func @manual_computation_no_inputs_no_outputs
func.func @manual_computation_no_inputs_no_outputs() {
  // CHECK-NEXT: sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
  // CHECK-NEXT:   sdy.return
  // CHECK-NEXT: } : () -> ()
  // CHECK-NEXT: return
  call @xla.sdy.manual_computation_body_8() {mhlo.frontend_attributes = {inlineable = "false"}} : () -> ()
  return
}

// Edge case: Manual computation with 0-sized dimension tensors.
// CHECK-LABEL: func @manual_computation_some_zero_dim_inputs_outputs
func.func @manual_computation_some_zero_dim_inputs_outputs(%arg0: tensor<0x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<0x32xf32>, tensor<16x32xf32>) {

  // CHECK-NOT: call @xla.sdy.manual_computation_body
  // CHECK:               %[[CONST_0_32:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0x32xf32>
  // CHECK:               %[[MAN_COMP:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{}, {"b"}]>, <@mesh_0, [{"b"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{}, {}], replicated={"b"}>, <@mesh_0, [{"b"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME:              (%arg2: tensor<0x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            %[[DOT:.*]] = stablehlo.dot %arg2, %arg3
  // CHECK-NEXT:            sdy.return %[[DOT]], %arg3
  // CHECK-NEXT:          } : (tensor<0x16xf32>, tensor<16x32xf32>) -> (tensor<0x32xf32>, tensor<16x32xf32>)
  // CHECK-NEXT:          return %[[CONST_0_32]], %[[MAN_COMP]]#1
  %c1 = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
  %c2 = stablehlo.constant dense<0.000000e+00> : tensor<0x32xf32>
  %0:2 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0, %arg1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>, <@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<0x16xf32>, tensor<16x32xf32>) -> (tensor<0x8xf32>, tensor<8x32xf32>)
  %1:2 = call @xla.sdy.manual_computation_body_9(%c1, %0#1) : (tensor<0x8xf32>, tensor<8x32xf32>) -> (tensor<0x32xf32>, tensor<8x32xf32>)
  %2:2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%c2, %1#1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}], replicated={"b"}>, <@mesh_0, [{"b"}, {}]>]>} : (tensor<0x32xf32>, tensor<8x32xf32>) -> (tensor<0x32xf32>, tensor<16x32xf32>)
  return %c2, %2#1 : tensor<0x32xf32>, tensor<16x32xf32>
}

// Type preservation: Token arguments are preserved with empty sharding and not assigned manual axes.
// CHECK-LABEL: func @manual_computation_tokens
func.func @manual_computation_tokens(%arg0: !stablehlo.token, %arg1: tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>) {
  // CHECK-NOT: call @xla.sdy.manual_computation_body_11
  // CHECK:               %[[MAN_COMP:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME:              (%arg2: !stablehlo.token, %arg3: tensor<1xi64>) {
  // CHECK-NEXT:            sdy.return %arg2, %arg3 : !stablehlo.token, tensor<1xi64>
  // CHECK-NEXT:          } : (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>)
  // CHECK-NEXT:          return %[[MAN_COMP]]#0, %[[MAN_COMP]]#1 : !stablehlo.token, tensor<2xi64>
  %0:2 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0, %arg1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>, <@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<1xi64>)
  %1:2 = call @xla.sdy.manual_computation_body_11(%0#0, %0#1) : (!stablehlo.token, tensor<1xi64>) -> (!stablehlo.token, tensor<1xi64>)
  %2:2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1#0, %1#1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>, <@mesh_0, [{"b"}]>]>} : (!stablehlo.token, tensor<1xi64>) -> (!stablehlo.token, tensor<2xi64>)
  return %2#0, %2#1 : !stablehlo.token, tensor<2xi64>
}

// Tests stray/dangling boundary custom calls where results are unused: safely stripped.
// CHECK-LABEL: func @stray_unused_manual_computation_custom_calls
func.func @stray_unused_manual_computation_custom_calls(%arg0: tensor<0x16xf32>) -> tensor<0x16xf32> {
  // CHECK-NEXT: %[[CONST_0_8:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
  // CHECK-NEXT: %[[CONST_0_16:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<0x16xf32>
  // CHECK-NOT: call @xla.sdy.manual_computation_body
  // CHECK-NOT: call @xla.sdy.GlobalToLocalShape
  // CHECK-NOT: call @xla.sdy.LocalToGlobalShape
  // CHECK-NEXT: return %[[CONST_0_16]]
  %c1 = stablehlo.constant dense<0.000000e+00> : tensor<0x8xf32>
  %c2 = stablehlo.constant dense<0.000000e+00> : tensor<0x16xf32>
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<0x16xf32>) -> tensor<0x8xf32>
  %1 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%c1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}, {}]>]>} : (tensor<0x8xf32>) -> tensor<0x16xf32>
  return %c2 : tensor<0x16xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body(
func.func private @xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<2x8xf32>
  %1 = stablehlo.dot %0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = "stablehlo.all_reduce"(%1) <{replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %3 : tensor<f32>
  }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  return %2 : tensor<2x32xf32>
}

// CHECK-NOT: func private @my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234(
func.func private @my_model.___call__.fwd.xla.sdy.manual_computation_body_14.1234(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_0(
func.func private @xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_1(
func.func private @xla.sdy.manual_computation_body_1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  return %arg0 : tensor<8x4xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_2(
func.func private @xla.sdy.manual_computation_body_2(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_3(
func.func private @xla.sdy.manual_computation_body_3(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  %1 = call @xla.sdy.manual_computation_body_2(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>} : (tensor<2x4xf32>) -> tensor<2x8xf32>
  return %2 : tensor<2x8xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_4(
func.func private @xla.sdy.manual_computation_body_4(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_5(
func.func private @xla.sdy.manual_computation_body_5(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  %1 = call @xla.sdy.manual_computation_body_4(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>} : (tensor<2x4xf32>) -> tensor<2x8xf32>
  %3 = stablehlo.add %2, %2 : tensor<2x8xf32>
  return %3 : tensor<2x8xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_6(
func.func private @xla.sdy.manual_computation_body_6() -> tensor<2xi64> {
  %0 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
  return %0 : tensor<2xi64>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_7(
func.func private @xla.sdy.manual_computation_body_7(%arg0: tensor<2xi64>) {
  stablehlo.custom_call @sdy_testonly(%arg0) : (tensor<2xi64>) -> ()
  return
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_8(
func.func private @xla.sdy.manual_computation_body_8() {
  return
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_9(
func.func private @xla.sdy.manual_computation_body_9(%arg0: tensor<0x8xf32>, %arg1: tensor<8x32xf32>) -> (tensor<0x32xf32>, tensor<8x32xf32>) {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<0x8xf32>, tensor<8x32xf32>) -> tensor<0x32xf32>
  return %0, %arg1 : tensor<0x32xf32>, tensor<8x32xf32>
}

// CHECK-NOT: func private @xla.sdy.manual_computation_body_11(
func.func private @xla.sdy.manual_computation_body_11(%arg0: !stablehlo.token, %arg1: tensor<1xi64>) -> (!stablehlo.token, tensor<1xi64>) {
  return %arg0, %arg1 : !stablehlo.token, tensor<1xi64>
}
