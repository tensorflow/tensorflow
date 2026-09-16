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
// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_disassemble(
    %arg0: !ifrt.array<tensor<2x4xi32>,
                       #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>)
    attributes {ifrt.function} {
  %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
          !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
  return
}

// -----

func.func @disassemble_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x4xi32>,
                       #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>) {
  // expected-error@+1 {{'ifrt.Disassemble' op must be in a FuncOp with attr `ifrt.function`}}
  %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>,
          !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>)
  return
}

// -----

func.func @disassemble_requires_outputs_on_single_devices(
    %arg0: !ifrt.array<tensor<2x4xi32>,
                       #ifrt.sharding_param<1x4 to [0, 1] on 2x2>, [0,1,2,3]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Disassemble' op requires every output to be a single device array. Actual: '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>'}}
  %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x4 to [0, 1] on 2x2>, [0,1,2,3]>)
      -> (!ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>,
          !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>)
  return
}

// -----

func.func @disassemble_requires_same_device_list(
    %arg0: !ifrt.array<tensor<2x4xi32>,
                       #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Disassemble' op requires the same input/output device list. Input 0, 1 vs Output 1, 2}}
  %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0)
      {operand_segment_sizes=array<i32: 2, 0>}
      : (!ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>)
      -> (!ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>,
          !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>)
  return
}
