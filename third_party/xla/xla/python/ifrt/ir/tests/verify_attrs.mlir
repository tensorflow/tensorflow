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

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_d01 = !ifrt.array<tensor<4x4xi32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_d01_copy = !ifrt.array<tensor<4x4xi32>,
                              #ifrt.sharding_param<2x1 to [0] on 2>, [2, 3]>
!array_d01_bc = !ifrt.array<tensor<4x4xf32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_s0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array_s1 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 1>, [1]>

func.func @good_function_attr() attributes {ifrt.function} {
  return
}

func.func @good_donated_attr(%arg0: !array {ifrt.donated})
    attributes {ifrt.function} {
  return
}

func.func @good_function_with_control_types(
    %arg0: !array, %ctrl: !ifrt.control) -> (!array, !ifrt.control)
    attributes {ifrt.function} {
  return %arg0, %ctrl : !array, !ifrt.control
}

func.func @good_reshard_function(
    %arg0: !array_d01, %arg1: !array_s0, %arg2: !array_s1)
    -> (!array_d01_copy, !array_d01_bc, !array_d01, !array_s0, !array_s1,
        !array_d01)
    attributes {ifrt.reshard_function} {
  %0, %ctrl0 = ifrt.Reshard(%arg0) : (!array_d01) -> !array_d01_copy
  %1, %ctrl1 = ifrt.CopyArrays(%arg0) : (!array_d01) -> !array_d01_copy
  %2, %ctrl2 = ifrt.BitcastArrays(%arg0) : (!array_d01) -> !array_d01_bc
  %3, %ctrl3 = ifrt.Assemble(%arg1, %arg2)
      : (!array_s0, !array_s1) -> !array_d01
  %4, %5, %ctrl4 = ifrt.Disassemble(%arg0)
      : (!array_d01) -> (!array_s0, !array_s1)
  %6, %ctrl5 = ifrt.RemapArrays(%arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:2:1] to [0:2:1]>]>]
      : (!array_d01) -> !array_d01
  return %1, %2, %3, %4, %5, %6
      : !array_d01_copy, !array_d01_bc, !array_d01, !array_s0, !array_s1,
        !array_d01
}

module @good_module_attrs attributes {
    ifrt.num_devices = 2 : i32,
    ifrt.local_view,
    ifrt.compile_options_key = "override_key"} {
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>,
        ifrt.devices = #ifrt<devices[0]>,
        ifrt.memory_kind = "device"
      },
      %arg1: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_unspecified
      })
      -> (tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>,
        ifrt.devices = #ifrt<devices[0]>,
        ifrt.memory_kind = "device"
      }, tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_unspecified
      }) {
    return %arg0, %arg1 : tensor<2x2xi32>, tensor<2x2xi32>
  }
}

module @good_call_attrs {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func private @atom_local(%arg0: tensor<2x4xi32>) -> tensor<2x4xi32> {
    return %arg0 : tensor<2x4xi32>
  }
  ifrt.LoadedExecutable @exec on devices [0, 1] : (!array_d01) -> !array_d01

  func.func @main(%arg0: !array_d01) -> !array_d01 attributes {ifrt.function} {
    %0, %ctrl0 = ifrt.Call @atom(%arg0) on devices [0, 1] {
      ifrt.module_type = "xla",
      ifrt.compile_options_key = "k1",
      ifrt.mesh_name = "mesh0"
    } : (!array_d01) -> !array_d01
    %1, %ctrl1 = ifrt.Call @atom_local(%0) on devices [0, 1] {
      ifrt.local_view,
      ifrt.module_type = "mpmd_reshard"
    } : (!array_d01) -> !array_d01
    %2, %ctrl2 = ifrt.CallLoadedExecutable @exec(%1) {
      ifrt.local_view,
      ifrt.module_type = "xla",
      ifrt.compile_options_key = "k2",
      ifrt.mesh_name = "mesh1"
    } : (!array_d01) -> !array_d01
    %3, %ctrl3 = ifrt.CallLoadedExecutable @exec(%2) {
      ifrt.module_type = "mpmd_reshard"
    } : (!array_d01) -> !array_d01
    return %3 : !array_d01
  }
}

module @good_unknown_ifrt_attrs attributes {ifrt.unknown_module_attr} {
  func.func @main(%arg0: tensor<2x2xi32> {ifrt.unknown_arg_attr = "v"})
      -> (tensor<2x2xi32> {ifrt.unknown_res_attr = 1 : i32})
      attributes {ifrt.unknown_func_attr} {
    return %arg0 : tensor<2x2xi32>
  }
}

func.func @good_unknown_ifrt_attrs_on_ifrt_function(
    %arg0: !array {ifrt.unknown_arg_attr})
    -> (!array {ifrt.unknown_res_attr})
    attributes {ifrt.function, ifrt.unknown_func_attr} {
  return %arg0 : !array
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.function` attr that is not a UnitAttr}}
func.func @func_attr_should_be_unit() attributes {ifrt.function = "1"} {
  return
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.function` attr but is not a function}}
module @func_attr_should_be_on_func_op attributes {ifrt.function} {}

// -----

// expected-error@+1 {{'func.func' op cannot have both `ifrt.function` and `ifrt.reshard_function` attrs}}
func.func @func_and_reshard_mutually_exclusive()
    attributes {ifrt.function, ifrt.reshard_function} {
  return
}

// -----

func.func @outer() {
  // expected-error@+1 {{'func.func' op with `ifrt.function` attr must be nested directly in a `builtin.module` op}}
  func.func @inner() attributes {ifrt.function} {
    return
  }
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.function` attr must have a single-block body}}
func.func private @external_ifrt_function() attributes {ifrt.function}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.function` attr must have a single-block body}}
func.func @multi_block_ifrt_function() attributes {ifrt.function} {
  cf.br ^bb1
^bb1:
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.function` attr must end with `func.return`}}
func.func @non_return_terminator_ifrt_function() attributes {ifrt.function} {
  stablehlo.return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.function` attr must have only `!ifrt.array` or `!ifrt.control` argument types, got 'tensor<4x4xi32>'}}
func.func @invalid_arg_type_ifrt_function(%arg0: tensor<4x4xi32>)
    attributes {ifrt.function} {
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.function` attr must have only `!ifrt.array` or `!ifrt.control` result types, got 'tensor<4x4xi32>'}}
func.func @invalid_result_type_ifrt_function() -> tensor<4x4xi32>
    attributes {ifrt.function} {
  %0 = stablehlo.constant dense<0> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.reshard_function` attr that is not a UnitAttr}}
func.func @reshard_attr_should_be_unit()
    attributes {ifrt.reshard_function = 1 : i32} {
  return
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.reshard_function` attr but is not a function}}
module @reshard_attr_should_be_on_func_op attributes {ifrt.reshard_function} {}

// -----

func.func @outer_reshard() {
  // expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must be nested directly in a `builtin.module` op}}
  func.func @inner_reshard() attributes {ifrt.reshard_function} {
    return
  }
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must have a single-block body}}
func.func private @external_reshard_function()
    attributes {ifrt.reshard_function}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must have a single-block body}}
func.func @multi_block_reshard_function() attributes {ifrt.reshard_function} {
  cf.br ^bb1
^bb1:
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must end with `func.return`}}
func.func @non_return_terminator_reshard_function()
    attributes {ifrt.reshard_function} {
  stablehlo.return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must have only `!ifrt.array` argument types, got '!ifrt.control'}}
func.func @control_arg_reshard_function(%ctrl: !ifrt.control)
    attributes {ifrt.reshard_function} {
  return
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must have only `!ifrt.array` result types, got '!ifrt.control'}}
func.func @control_result_reshard_function(%arg0: !array) -> !ifrt.control
    attributes {ifrt.reshard_function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> !array
  return %ctrl : !ifrt.control
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr must have only `!ifrt.array` argument types, got 'tensor<4x4xi32>'}}
func.func @tensor_arg_reshard_function(%arg0: tensor<4x4xi32>)
    attributes {ifrt.reshard_function} {
  return
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @nested_call_in_reshard_function {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  // expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr may only contain resharding ops and `func.return`, got `ifrt.Call`}}
  func.func @reshard(%arg0: !array) -> !array
      attributes {ifrt.reshard_function} {
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0] : (!array) -> !array
    return %0 : !array
  }
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr may only contain resharding ops and `func.return`, got `ifrt.After`}}
func.func @after_in_reshard_function(%arg0: !array)
    attributes {ifrt.reshard_function} {
  %ctrl = "ifrt.After"(%arg0) : (!array) -> !ifrt.control
  return
}

// -----

// expected-error@+1 {{'func.func' op with `ifrt.reshard_function` attr may only contain resharding ops and `func.return`, got `stablehlo.constant`}}
func.func @non_ifrt_op_in_reshard_function()
    attributes {ifrt.reshard_function} {
  %0 = stablehlo.constant dense<0> : tensor<4x4xi32>
  return
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @local_view_not_unit_on_call {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.local_view` attr that is not a UnitAttr}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0] {ifrt.local_view = true}
        : (!array) -> !array
    return %0 : !array
  }
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.local_view` attr that is not a UnitAttr}}
module @local_view_not_unit_on_module attributes {ifrt.local_view = "1"} {}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.local_view` attr but is not a `ifrt.Call`, `ifrt.CallLoadedExecutable`, or `builtin.module` op}}
func.func @local_view_wrong_op() attributes {ifrt.function, ifrt.local_view} {
  return
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @module_type_invalid_value {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.module_type` attr that must be a StringAttr with value `xla` or `mpmd_reshard`}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0]
        {ifrt.module_type = "unknown"} : (!array) -> !array
    return %0 : !array
  }
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @module_type_not_string {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.module_type` attr that must be a StringAttr with value `xla` or `mpmd_reshard`}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0]
        {ifrt.module_type = 1 : i32} : (!array) -> !array
    return %0 : !array
  }
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.module_type` attr but is not a `ifrt.Call` or `ifrt.CallLoadedExecutable` op}}
module @module_type_wrong_op attributes {ifrt.module_type = "xla"} {}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @compile_options_key_empty_on_call {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.compile_options_key` attr that is not a non-empty StringAttr}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0]
        {ifrt.compile_options_key = ""} : (!array) -> !array
    return %0 : !array
  }
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.compile_options_key` attr that is not a non-empty StringAttr}}
module @compile_options_key_empty_on_module
    attributes {ifrt.compile_options_key = ""} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.compile_options_key` attr that is not a non-empty StringAttr}}
module @compile_options_key_not_string
    attributes {ifrt.compile_options_key = 1 : i32} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.num_devices` attr that is not a positive i32 IntegerAttr}}
module @num_devices_zero attributes {ifrt.num_devices = 0 : i32} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.num_devices` attr that is not a positive i32 IntegerAttr}}
module @num_devices_negative attributes {ifrt.num_devices = -1 : i32} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.num_devices` attr that is not a positive i32 IntegerAttr}}
module @num_devices_i64 attributes {ifrt.num_devices = 2} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.num_devices` attr that is not a positive i32 IntegerAttr}}
module @num_devices_bool attributes {ifrt.num_devices = true} {}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.num_devices` attr that is not a positive i32 IntegerAttr}}
module @num_devices_string attributes {ifrt.num_devices = "2"} {}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @mesh_name_empty_on_call {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.mesh_name` attr that is not a non-empty StringAttr}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0]
        {ifrt.mesh_name = ""} : (!array) -> !array
    return %0 : !array
  }
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @mesh_name_not_string_on_call {
  func.func private @atom(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    return %arg0 : tensor<4x4xi32>
  }
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Call' op has `ifrt.mesh_name` attr that is not a non-empty StringAttr}}
    %0, %ctrl = ifrt.Call @atom(%arg0) on devices [0]
        {ifrt.mesh_name = 1 : i32} : (!array) -> !array
    return %0 : !array
  }
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr that is not a UnitAttr}}
func.func @donated_attr_should_be_unit(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
        {ifrt.donated = "1"})
    attributes {ifrt.function} {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr but not has `ifrt.function` attr}}
func.func @donated_attr_should_be_with_func_attr(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>,
                       [0]> {ifrt.donated}) {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr but not has `ifrt.function` attr}}
func.func @donated_attr_not_allowed_on_reshard_function(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>,
                       [0]> {ifrt.donated})
    attributes {ifrt.reshard_function} {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr on argument #0 that is not an `!ifrt.array`}}
func.func @donated_attr_not_allowed_on_control_arg(
    %ctrl: !ifrt.control {ifrt.donated}) attributes {ifrt.function} {
  return
}

// -----

!array = !ifrt.array<tensor<4x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
// expected-error@+1 {{'func.func' op has `ifrt.donated` result attr, which is only valid on arguments}}
func.func @donated_attr_not_allowed_on_result(%arg0: !array)
    -> (!array {ifrt.donated}) attributes {ifrt.function} {
  return %arg0 : !array
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.sharding` arg attr that is not an `IfrtShardingAttrInterface`}}
func.func @sharding_arg_invalid(%arg0: tensor<2x2xi32> {ifrt.sharding = "bad"}) {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.sharding` result attr that is not an `IfrtShardingAttrInterface`}}
func.func @sharding_result_invalid(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32> {ifrt.sharding = 1 : i32}) {
  return %arg0 : tensor<2x2xi32>
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.devices` arg attr that is not an `IfrtDevicesAttr`}}
func.func @devices_arg_invalid(%arg0: tensor<2x2xi32> {ifrt.devices = "bad"}) {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.devices` result attr that is not an `IfrtDevicesAttr`}}
func.func @devices_result_invalid(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32> {ifrt.devices = [0]}) {
  return %arg0 : tensor<2x2xi32>
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.memory_kind` arg attr that is not a non-empty StringAttr}}
func.func @memory_kind_arg_empty(%arg0: tensor<2x2xi32> {ifrt.memory_kind = ""}) {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.memory_kind` result attr that is not a non-empty StringAttr}}
func.func @memory_kind_result_empty(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32> {ifrt.memory_kind = ""}) {
  return %arg0 : tensor<2x2xi32>
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.memory_kind` arg attr that is not a non-empty StringAttr}}
func.func @memory_kind_arg_not_string(
    %arg0: tensor<2x2xi32> {ifrt.memory_kind = 1 : i32}) {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.memory_kind` result attr that is not a non-empty StringAttr}}
func.func @memory_kind_result_not_string(%arg0: tensor<2x2xi32>)
    -> (tensor<2x2xi32> {ifrt.memory_kind = 1 : i32}) {
  return %arg0 : tensor<2x2xi32>
}
