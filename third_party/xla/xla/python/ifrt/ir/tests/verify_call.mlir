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

func.func @good_call(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @good_call_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>,
    %arg1: !ifrt.control)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) after %arg1 on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @good_call_with_io_aliases(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

#devices = #ifrt<devices[0,1]>
func.func @good_call_with_aliased_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       #devices>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices #devices
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   #devices>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   #devices>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!token = !ifrt.array<tensor<!ifrt.token>,
                     #ifrt.sharding_param< to [0] on 2>, [0, 1]>
func.func @good_call_with_token_type(%arg0: !token) attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!token) -> !token
  return
}


func.func @callee(%arg0: !stablehlo.token) -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}

// -----

!token = !ifrt.array<tensor<!ifrt.token>,
                     #ifrt.sharding_param< to [0] on 2>, [0, 1]>
func.func @good_call_with_token_type(%arg0: !token) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<!ifrt.token>' vs Callee 'tensor<i32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!token) -> !token
  return
}


func.func @callee(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}


// -----

func.func @call_requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>) {
  // expected-error@+1 {{'ifrt.Call' op must be in a FuncOp with attr `ifrt.function` or attr `ifrt.reshard_function`}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
module @call_directly_in_module {
  func.func private @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
  %arg = "builtin.unrealized_conversion_cast"() : () -> !array
  // expected-error@+1 {{'ifrt.Call' op must be in a FuncOp with attr `ifrt.function` or attr `ifrt.reshard_function`}}
  %0, %ctrl = ifrt.Call @callee(%arg) on devices [0,1] : (!array) -> !array
}

// -----

!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
module @call_in_nested_region {
  func.func private @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    %cond = stablehlo.constant dense<true> : tensor<i1>
    "stablehlo.if"(%cond) ({
      // expected-error@+1 {{'ifrt.Call' op must be in a FuncOp with attr `ifrt.function` or attr `ifrt.reshard_function`}}
      %0, %ctrl = ifrt.Call @callee(%arg0) on devices [0,1] : (!array) -> !array
      stablehlo.return
    }, {
      stablehlo.return
    }) : (tensor<i1>) -> ()
    return
  }
}

// -----

func.func @call_requires_valid_reference(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires '@missing_reference' to reference a valid `func.func`}}
  %0, %ctrl_0 = ifrt.Call @missing_reference(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

// -----

func.func @call_requires_same_input_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same input size. Input 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee() -> (tensor<4x4xi32>) {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_input_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<2x2xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x4xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_same_output_size(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same output size. Output 1 vs Callee 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) {
  return
}

// -----

func.func @call_requires_same_output_shape(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Output #0 'tensor<4x4xi32>' vs Callee 'tensor<2x4xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @call_requires_non_negative_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' Device list has negative logical id -1}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,-1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_unique_devices_attr(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' Device list has duplicate logical id 0}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,0]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_input_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,2]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires all inputs placed on `devices` attr. The following input is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,2]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @call_requires_output_place_on_devices(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires all outputs placed on `devices` attr. The following output is placed on device 2 not found in `devices` attr. '!ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 2]>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,2]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<4x4xi32> {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

func.func @io_aliases_should_be_pairs(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op attribute 'io_aliases' failed to satisfy constraint: Array of pairs of aliased input/output indices}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_not_have_negative_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #-1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: -1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_input_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias or donate input #0 more than once}}
  %0, %1, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>, array<i32: 0, 1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>,
        !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>)
  return
}

func.func @callee(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  return %arg0, %arg0 : tensor<2x2xi32>, tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_have_valid_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_not_have_negative_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #-1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, -1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @io_aliases_should_only_alias_output_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias output #0 more than once}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>, array<i32: 1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>,
       !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array0 = !ifrt.array<tensor<2x1xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<1x1xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @io_aliases_of_different_type_but_same_per_shard_shape(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array1
  return
}

func.func @callee(%arg0: tensor<2x1xi32>) -> tensor<1x1xi32> {
  %0 = mhlo.constant dense<-2147483648> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.maximum across dimensions = [0, 1]
    : (tensor<2x1xi32>, tensor<i32>) -> tensor<i32>
  %2 = mhlo.reshape %1 : (tensor<i32>) -> tensor<1x1xi32>
  return %2 : tensor<1x1xi32>
}

// -----

func.func @good_call_local_view(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,2,3] {ifrt.local_view}
    : (!ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    -> !ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

func.func @call_local_view_should_have_valid_shape(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires the same global shape. Input #0 'tensor<4x4xi32>' vs Callee 'tensor<8x8xi32>'}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1,2,3] {ifrt.local_view}
    : (!ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>)
    -> !ifrt.array<tensor<4x4xi32>,
                   #ifrt.sharding_param<2x2 to [0, 1] on 2x2>, [0,1,2,3]>
  return
}

func.func @callee(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  return %arg0 : tensor<4x4xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @donate_an_arg_and_alias_another(%arg0: !array, %arg1: !array)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg1) on devices [0,1]
    <donated_input_indices=[0], io_aliases=[array<i32: 1, 0>]>
    : (!array, !array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @should_only_donate_once(%arg0: !array, %arg1: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't donate input #0 more than once}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg1) on devices [0,1]
    <donated_input_indices=[0, 0]>
    : (!array, !array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @should_not_both_donate_and_alias_the_same_arg(
    %arg0: !array, %arg1: !array) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias or donate input #0 more than once}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg1) on devices [0,1]
    <donated_input_indices=[0], io_aliases=[array<i32: 0, 0>]>
    : (!array, !array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @good_call_with_arg_and_res_attrs(%arg0: !array, %arg1: !array)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg1) on devices [0,1]
    <arg_attrs = [{test.arg = "a"}, {}], res_attrs = [{test.res = "r"}]>
    : (!array, !array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @call_requires_matching_arg_attrs_arity(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires `arg_attrs` to have the same size as `inputs` (2 vs. 1)}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <arg_attrs = [{}, {}]>
    : (!array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @call_requires_matching_res_attrs_arity(%arg0: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op requires `res_attrs` to have the same size as `outputs` (2 vs. 1)}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <res_attrs = [{}, {}]>
    : (!array) -> !array
  return
}

func.func @callee(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  return %arg0 : tensor<2x2xi32>
}

// -----

!arr_d0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!arr_d1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!arr_1x2_d0 = !ifrt.array<tensor<1x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!arr_2x2_d01 = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!arr_d01 = !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_d10 = !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 2>, [1,0]>
!arr_auto_d0 = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 1>, [0],
                           layout = "auto">
!arr_auto_d1 = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 1>, [1],
                           layout = "auto">
!arr_unspec_d0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_unspecified, [0]>

func.func @io_aliases_rejects_different_devices(%arg0: !arr_d0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_d0) -> !arr_d1
  return
}

func.func @io_aliases_rejects_different_device_count(%arg0: !arr_1x2_d0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.Call @callee_1x2(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_1x2_d0) -> !arr_2x2_d01
  return
}

func.func @io_aliases_rejects_different_device_order(%arg0: !arr_d01)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_d01) -> !arr_d10
  return
}

func.func @io_aliases_auto_layout_rejects_different_devices(%arg0: !arr_auto_d0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_auto_d0) -> !arr_auto_d1
  return
}

func.func @io_aliases_unspec_sharding_rejects_different_devices(
    %arg0: !arr_unspec_d0) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_unspec_d0) -> !arr_d1
  return
}

func.func private @callee(tensor<2x2xi32>) -> tensor<2x2xi32>
func.func private @callee_1x2(tensor<1x2xi32>) -> tensor<2x2xi32>

// -----

!arr_default = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_pinned = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                          memory_kind = "pinned_host">
!arr_unpinned = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                            memory_kind = "unpinned_host">
!arr_auto = !ifrt.array<tensor<2x2xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                        layout = "auto">
!arr_pinned_auto = !ifrt.array<tensor<2x2xi32>,
                               #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                               memory_kind = "pinned_host", layout = "auto">
!arr_unspec = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_unspecified, [0,1]>

func.func @io_aliases_rejects_default_vs_pinned_host(%arg0: !arr_default)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different memory kinds:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_default) -> !arr_pinned
  return
}

func.func @io_aliases_rejects_pinned_vs_unpinned_host(%arg0: !arr_pinned)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different memory kinds:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_pinned) -> !arr_unpinned
  return
}

func.func @io_aliases_auto_layout_rejects_different_memory_kinds(
    %arg0: !arr_auto) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different memory kinds:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_auto) -> !arr_pinned_auto
  return
}

func.func @io_aliases_unspec_sharding_rejects_different_memory_kinds(
    %arg0: !arr_unspec) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0 with different memory kinds:}}
  %0, %ctrl_0 = ifrt.Call @callee(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_unspec) -> !arr_pinned
  return
}

func.func private @callee(tensor<2x2xi32>) -> tensor<2x2xi32>

// -----

!token = !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 2>,
                     [0,1]>
!zero = !ifrt.array<tensor<0xf32>, #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!arr_auto = !ifrt.array<tensor<4x4xf32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                        layout = "auto">
!string_arr = !ifrt.array<tensor<2x!ifrt.string>,
                          #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!arr_user = !ifrt.array<tensor<4x4xf32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                        layout = "{1,0}">

func.func @io_aliases_rejects_token_to_token(%arg0: !token)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0: unsupported element type '!ifrt.token'}}
  %0, %ctrl_0 = ifrt.Call @callee_tok(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!token) -> !token
  return
}

func.func @io_aliases_rejects_token_to_zero_element_array(%arg0: !zero)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0: unsupported element type '!ifrt.token'}}
  %0, %ctrl_0 = ifrt.Call @callee_zero_tok(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!zero) -> !token
  return
}

func.func @io_aliases_rejects_token_with_auto_layout(%arg0: !token)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0: unsupported element type '!ifrt.token'}}
  %0, %ctrl_0 = ifrt.Call @callee_tok_f32(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!token) -> !arr_auto
  return
}

func.func @io_aliases_rejects_string_to_string(%arg0: !string_arr)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0: unsupported element type '!ifrt.string'}}
  %0, %ctrl_0 = ifrt.Call @callee_str(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!string_arr) -> !string_arr
  return
}

func.func @io_aliases_rejects_string_with_user_layout(%arg0: !string_arr)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.Call' op can't alias input #0 to output #0: unsupported element type '!ifrt.string'}}
  %0, %ctrl_0 = ifrt.Call @callee_str_f32(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!string_arr) -> !arr_user
  return
}

func.func private @callee_tok(!stablehlo.token) -> !stablehlo.token
func.func private @callee_zero_tok(tensor<0xf32>) -> !stablehlo.token
func.func private @callee_tok_f32(!stablehlo.token) -> tensor<4x4xf32>
func.func private @callee_str(tensor<2x!ifrt.string>) -> tensor<2x!ifrt.string>
func.func private @callee_str_f32(tensor<2x!ifrt.string>) -> tensor<4x4xf32>

// -----

!arr_i32 = !ifrt.array<tensor<2x2xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_device = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                          memory_kind = "device">
!arr_i16 = !ifrt.array<tensor<2x2xi16>,
                       #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                       layout = "default">
!arr_4_f32 = !ifrt.array<tensor<4xf32>,
                         #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!arr_zero_f32 = !ifrt.array<tensor<0x4xf32>,
                            #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_zero_i16 = !ifrt.array<tensor<0x8xi16>,
                            #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_pinned_0 = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                            memory_kind = "pinned_host">
!arr_pinned_1 = !ifrt.array<tensor<4xf32>,
                            #ifrt.sharding_param<1 to [0] on 2>, [0,1],
                            memory_kind = "pinned_host">
!arr_untiled_0 = !ifrt.array<tensor<12x16xf32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                             layout = "{1,0}">
!arr_tiled_1 = !ifrt.array<tensor<12x6xi32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                           layout = "{0,1:T(1,128)}">
!arr_auto_large = !ifrt.array<tensor<4x4xf32>,
                              #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                              layout = "auto">
!arr_unspec = !ifrt.array<tensor<12x16xf32>, #ifrt.sharding_unspecified, [0,1]>
!arr_i4 = !ifrt.array<tensor<8xi4>, #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!arr_i8 = !ifrt.array<tensor<4xi8>, #ifrt.sharding_param<1 to [0] on 2>, [0,1]>
!arr_non_div = !ifrt.array<tensor<3x2xi32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
func.func @io_aliases_allows_compatible_pairs(
    %arg0: !arr_i32, %arg1: !arr_zero_f32, %arg2: !arr_pinned_0,
    %arg3: !arr_untiled_0, %arg4: !arr_unspec, %arg5: !arr_i4,
    %arg6: !arr_non_div)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.Call @callee_same(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_i32) -> !arr_device
  %1, %ctrl_1 = ifrt.Call @callee_i16(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_i32) -> !arr_i16
  %2, %ctrl_2 = ifrt.Call @callee_rank(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_i32) -> !arr_4_f32
  %3, %ctrl_3 = ifrt.Call @callee_zero(%arg1) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_zero_f32) -> !arr_zero_i16
  %4, %ctrl_4 = ifrt.Call @callee_rank(%arg2) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_pinned_0) -> !arr_pinned_1
  %5, %ctrl_5 = ifrt.Call @callee_tiled(%arg3) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_untiled_0) -> !arr_tiled_1
  %6, %ctrl_6 = ifrt.Call @callee_auto(%arg0) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_i32) -> !arr_auto_large
  %7, %ctrl_7 = ifrt.Call @callee_tiled(%arg4) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_unspec) -> !arr_tiled_1
  %8, %ctrl_8 = ifrt.Call @callee_i4(%arg5) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_i4) -> !arr_i8
  %9, %ctrl_9 = ifrt.Call @callee_non_div(%arg6) on devices [0,1]
    <io_aliases=[array<i32: 0, 0>]> : (!arr_non_div) -> !arr_non_div
  return
}

func.func private @callee_same(tensor<2x2xi32>) -> tensor<2x2xi32>
func.func private @callee_i16(tensor<2x2xi32>) -> tensor<2x2xi16>
func.func private @callee_rank(tensor<2x2xi32>) -> tensor<4xf32>
func.func private @callee_zero(tensor<0x4xf32>) -> tensor<0x8xi16>
func.func private @callee_tiled(tensor<12x16xf32>) -> tensor<12x6xi32>
func.func private @callee_auto(tensor<2x2xi32>) -> tensor<4x4xf32>
func.func private @callee_i4(tensor<8xi4>) -> tensor<4xi8>
func.func private @callee_non_div(tensor<3x2xi32>) -> tensor<3x2xi32>
