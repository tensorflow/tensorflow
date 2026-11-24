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

func.func @good(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

func.func @good_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>,
    %arg1: !ifrt.control)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0) after %arg1
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

// -----

func.func @requires_in_ifrt_function(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op must be in a FuncOp with attr `ifrt.function`}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

// -----

func.func @requires_valid_reference() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@missing_reference' to reference a valid `ifrt.LoadedExecutable`}}
  %ctrl_0 = ifrt.CallLoadedExecutable @missing_reference() : () -> ()
  return
}

// -----

func.func @requires_loaded_executable_callee(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@wrong_reference' to reference a valid `ifrt.LoadedExecutable`}}
  %ctrl_0 = ifrt.CallLoadedExecutable @wrong_reference() : () -> ()
  return
}

func.func @wrong_reference() {
  return
}

// -----

func.func @requires_direct_module_child_callee() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@nested::@callee' to reference an `ifrt.LoadedExecutable` directly in the enclosing `builtin.module`}}
  %ctrl_0 = ifrt.CallLoadedExecutable @nested::@callee() : () -> ()
  return
}

module @nested {
  ifrt.LoadedExecutable @callee on devices [0,1] : () -> ()
}

// -----

func.func @requires_direct_module_child_even_for_private_nested_callee()
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@nested::@callee' to reference an `ifrt.LoadedExecutable` directly in the enclosing `builtin.module`}}
  %ctrl_0 = ifrt.CallLoadedExecutable @nested::@callee() : () -> ()
  return
}

module @nested {
  ifrt.LoadedExecutable "private" @callee on devices [0,1] : () -> ()
}

// -----

module @outer_module_callee_not_visible {
  module @nested {
    func.func @main() attributes {ifrt.function} {
      // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@callee' to reference a valid `ifrt.LoadedExecutable`}}
      %ctrl_0 = ifrt.CallLoadedExecutable @callee() : () -> ()
      return
    }
  }
  ifrt.LoadedExecutable @callee on devices [0,1] : () -> ()
}

// -----

func.func @requires_matching_signature(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires callee signature matching '(!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>) -> !ifrt.array<tensor<4x3xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>'. Actual '(!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>) -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0, 1]>'}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x3xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_be_pairs(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op attribute 'io_aliases' failed to satisfy constraint: Array of pairs of aliased input/output indices}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_have_valid_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: 1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_not_have_negative_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #-1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: -1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_only_alias_input_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias or donate input #0 more than once}}
  %0, %1, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: 0, 0>, array<i32: 0, 1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>,
        !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                    [0,1]>)
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_have_valid_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: 0, 1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_not_have_negative_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #-1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <io_aliases=[array<i32: 0, -1>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

func.func @io_aliases_should_only_alias_output_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                       [0,1]>)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias output #0 more than once}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg0)
    <io_aliases=[array<i32: 0, 0>, array<i32: 1, 0>]>
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>,
       !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>
  return
}

ifrt.LoadedExecutable @callee on devices [0,1]
    : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                   [0,1]>

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @donate_one_arg_and_alias_another_arg(%arg0: !array, %arg1: !array)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg1)
    <donated_input_indices=[0], io_aliases=[array<i32: 1, 0>]>
    : (!array, !array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array, !array) -> !array

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @should_only_donate_once(%arg0: !array, %arg1: !array)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't donate input #0 more than once}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg1)
    <donated_input_indices=[0, 0]> : (!array, !array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array, !array) -> !array

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @should_not_both_donate_and_alias_the_same_arg(
    %arg0: !array, %arg1: !array) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias or donate input #0 more than once}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg1)
    <donated_input_indices=[0], io_aliases=[array<i32: 0, 0>]>
    : (!array, !array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array, !array) -> !array

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @good_call_loaded_executable_with_arg_and_res_attrs(
    %arg0: !array, %arg1: !array) attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg1)
    <arg_attrs = [{test.arg = "a"}, {}], res_attrs = [{test.res = "r"}]>
    : (!array, !array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array, !array) -> !array

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @call_loaded_executable_requires_matching_arg_attrs_arity(
    %arg0: !array) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires `arg_attrs` to have the same size as `inputs` (2 vs. 1)}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <arg_attrs = [{}, {}]>
    : (!array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array) -> !array

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @call_loaded_executable_requires_matching_res_attrs_arity(
    %arg0: !array) attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires `res_attrs` to have the same size as `outputs` (2 vs. 1)}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    <res_attrs = [{}, {}]>
    : (!array) -> !array
  return
}

ifrt.LoadedExecutable @callee on devices [0,1] : (!array) -> !array

// -----

!arr_d0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!arr_d1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
!arr_default = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_pinned = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                          memory_kind = "pinned_host">
!token = !ifrt.array<tensor<!ifrt.token>, #ifrt.sharding_param< to [0] on 2>,
                     [0,1]>
!string_arr = !ifrt.array<tensor<2x!ifrt.string>,
                          #ifrt.sharding_param<1 to [0] on 2>, [0,1]>

func.func @io_aliases_rejects_different_devices(%arg0: !arr_d0)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #0 with different devices:}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee_dev(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_d0) -> !arr_d1
  return
}

func.func @io_aliases_rejects_different_memory_kinds(%arg0: !arr_default)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #0 with different memory kinds:}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee_mem(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_default) -> !arr_pinned
  return
}

func.func @io_aliases_rejects_token_to_token(%arg0: !token)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #0: unsupported element type '!ifrt.token'}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee_tok(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!token) -> !token
  return
}

func.func @io_aliases_rejects_string_to_string(%arg0: !string_arr)
    attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #0: unsupported element type '!ifrt.string'}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee_str(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!string_arr) -> !string_arr
  return
}

ifrt.LoadedExecutable @callee_dev on devices [0,1] : (!arr_d0) -> !arr_d1
ifrt.LoadedExecutable @callee_mem on devices [0,1] : (!arr_default) -> !arr_pinned
ifrt.LoadedExecutable @callee_tok on devices [0,1] : (!token) -> !token
ifrt.LoadedExecutable @callee_str on devices [0,1] : (!string_arr) -> !string_arr

// -----

!arr_default = !ifrt.array<tensor<2x2xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!arr_device = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                          memory_kind = "device">
!arr_1x4_f32 = !ifrt.array<tensor<1x4xf32>,
                           #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                           layout = "default">
!arr_tiled_0 = !ifrt.array<tensor<12x16xf32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                           layout = "{1,0:T(1,128)}">
!arr_tiled_1 = !ifrt.array<tensor<12x6xi32>,
                           #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                           layout = "{0,1:T(1,128)}">
!arr_auto_large = !ifrt.array<tensor<4x4xf32>,
                              #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                              layout = "auto">
!arr_unspec = !ifrt.array<tensor<12x16xf32>, #ifrt.sharding_unspecified, [0,1]>
func.func @io_aliases_allows_compatible_pairs(
    %arg0: !arr_default, %arg1: !arr_tiled_0, %arg2: !arr_unspec)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee_mem(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_default) -> !arr_device
  %1, %ctrl_1 = ifrt.CallLoadedExecutable @callee_default(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_default) -> !arr_1x4_f32
  %2, %ctrl_2 = ifrt.CallLoadedExecutable @callee_tiled(%arg1)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_tiled_0) -> !arr_tiled_1
  %3, %ctrl_3 = ifrt.CallLoadedExecutable @callee_auto(%arg0)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_default) -> !arr_auto_large
  %4, %ctrl_4 = ifrt.CallLoadedExecutable @callee_unspec(%arg2)
    <io_aliases=[array<i32: 0, 0>]> : (!arr_unspec) -> !arr_tiled_1
  return
}

ifrt.LoadedExecutable @callee_mem on devices [0,1] : (!arr_default) -> !arr_device
ifrt.LoadedExecutable @callee_default on devices [0,1] : (!arr_default) -> !arr_1x4_f32
ifrt.LoadedExecutable @callee_tiled on devices [0,1] : (!arr_tiled_0) -> !arr_tiled_1
ifrt.LoadedExecutable @callee_auto on devices [0,1] : (!arr_default) -> !arr_auto_large
ifrt.LoadedExecutable @callee_unspec on devices [0,1] : (!arr_unspec) -> !arr_tiled_1
