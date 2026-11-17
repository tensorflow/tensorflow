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
// RUN: ifrt-opt %s -ifrt-verify-donation -split-input-file -verify-diagnostics | FileCheck %s

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
// CHECK-LABEL: @donate_call_output_to_call_and_reshard
module @donate_call_output_to_call_and_reshard {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> !array1
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array0
    %2, %ctrl_2 = ifrt.Reshard(%1) <donated=true> : (!array0) -> !array1
    return %2 : !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [2]>
!array2 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [3]>
module @donate_to_reshard_duplicated_arg {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array2)
        attributes {ifrt.function} {
    %0, %1, %ctrl_1 = ifrt.Reshard(%arg0, %arg0) <donated=true>
        : (!array0, !array0) -> (!array1, !array2)
    return %0, %1 : !array1, !array2
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @alias_to_two_calls_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @donate_to_two_calls_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        <donated_input_indices=[0]> : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_donated_to_call_not_donated_to_program {
  func.func @main(%arg0: !array) -> (!array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <donated_input_indices=[0]> : (!array) -> !array
    return %0 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @program_arg_not_donated_error {
  func.func @main(%arg0: !array0) -> (!array1) attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Reshard' op input #0 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    return %0 : !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_both_donated_and_not_donated_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> !array0
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #0 of @add_two_args was already donated}}
    %0, %ctrl_0 = ifrt.Call @add_two_args(%arg0, %arg0) on devices [0,1]
        <io_aliases=[array<i32: 1, 0>]> : (!array0, !array0) -> !array0
    return %0 : !array0
  }

  func.func private @add_two_args(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
        -> tensor<2xi32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_reshards_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_reshards_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_reshard_and_call_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) <donated=true> : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_copy_arrays_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <donated=true> : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.CopyArrays' op input #0 of op}}
    %1, %ctrl_1 = ifrt.CopyArrays(%arg0) <donated=true> : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @program_arg_not_donated_to_remap_error {
  func.func @main(%arg0: !array {ifrt.donated}, %arg1: !array) -> (!array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.RemapArrays' op input #1 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      <donated=true> : (!array, !array) -> !array
    return %0 : !array
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @donate_to_remap_and_call_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    // expected-error @+1 {{'ifrt.RemapArrays' op input #1 of op}}
    %1, %ctrl_1 = ifrt.RemapArrays(%0, %arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      <donated=true> : (!array, !array) -> !array
    return %1 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
module @program_arg_not_donated_to_bitcast_error {
  func.func @main(%arg0: !array0 {ifrt.donated}, %arg1: !array1) -> (!array1, !array0)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.BitcastArrays' op input #1 has not been donated to the program.}}
    %0, %1, %ctrl_0 = ifrt.BitcastArrays(%arg0, %arg1) <donated=true> : (!array0, !array1) -> (!array1, !array0)
    return %0, %1 : !array1, !array0
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x1xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
module @donate_to_bitcast_and_call_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.BitcastArrays' op input #0 of op}}
    %1, %ctrl_1 = ifrt.BitcastArrays(%arg0) <donated=true> : (!array0) -> (!array1)
    return %1 : !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_after_donation_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @reshard_with_already_donated_array_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @copy_arrays_with_already_donated_array_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.CopyArrays' op input #0 of op}}
    %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_with_already_donated_array_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %0, %arg0 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x1xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
module @donate_to_bitcast_duplicated_arg_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.BitcastArrays' op input #1 of op}}
    %0, %1, %ctrl_0 = ifrt.BitcastArrays(%arg0, %arg0) <donated=true>
        : (!array0, !array0) -> (!array1, !array1)
    return %0, %1 : !array1, !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
// CHECK-LABEL: @donate_to_copy_arrays_duplicated_arg
module @donate_to_copy_arrays_duplicated_arg {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg0) <donated=true>
        : (!array0, !array0) -> (!array1, !array1)
    return %0, %1 : !array1, !array1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
// CHECK-LABEL: @donate_to_remap_arrays_duplicated_arg
module @donate_to_remap_arrays_duplicated_arg {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.RemapArrays(%arg0, %arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                  #ifrt.array_mapping<1, 1, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
        <donated=true> : (!array, !array) -> (!shard0, !shard1)
    return %0, %1 : !shard0, !shard1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_loaded_executable_duplicate_donated_arg_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.CallLoadedExecutable' op input #1 of @callee was already donated or aliased}}
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg0)
        <donated_input_indices=[0, 1]> : (!array, !array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @callee on devices [0, 1]
      : (!array, !array) -> !array
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_loaded_executable_both_donated_and_not_donated_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.CallLoadedExecutable' op input #0 of @callee was already donated}}
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg0)
        <io_aliases=[array<i32: 1, 0>]> : (!array, !array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @callee on devices [0, 1]
      : (!array, !array) -> !array
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @assemble_with_already_donated_array_error {
  func.func @main(%arg0: !shard0 {ifrt.donated}, %arg1: !shard1) -> (!shard0, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    // expected-error @+1 {{'ifrt.Assemble' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Assemble(%arg0, %arg1) : (!shard0, !shard1) -> !array
    return %0, %1 : !shard0, !array
  }

  func.func private @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @disassemble_with_already_donated_array_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !shard0, !shard1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Disassemble' op input #0 of op}}
    %1, %2, %ctrl_1 = ifrt.Disassemble(%arg0) : (!array) -> (!shard0, !shard1)
    return %0, %1, %2 : !array, !shard0, !shard1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_reuse_program_arg_not_donated_error {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    return %1 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_reuse_donate_alias_then_return_original_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %1, %arg0 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_reuse_donate_original_then_return_alias_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %1, %0 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
// CHECK-LABEL: @copy_arrays_reuse_different_devices_no_alias
module @copy_arrays_reuse_different_devices_no_alias {
  func.func @main(%arg0: !array0) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array0) -> !array1
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [2, 3]
        <donated_input_indices=[0]> : (!array1) -> !array1
    return %arg0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
// CHECK-LABEL: @remap_disjoint_shards_donated_separately
module @remap_disjoint_shards_donated_separately {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                  #ifrt.array_mapping<0, 1, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
        : (!array) -> (!shard0, !shard1)
    %2, %ctrl_1 = ifrt.Call @identity(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    %3, %ctrl_2 = ifrt.Call @identity(%1) on devices [1]
        <io_aliases=[array<i32: 0, 0>]> : (!shard1) -> !shard1
    return %2, %3 : !shard0, !shard1
  }

  func.func private @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @remap_partial_shard_donation_then_use_original_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard0
    %1, %ctrl_1 = ifrt.Call @id_shard(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    // expected-error @+1 {{'ifrt.Call' op input #0 of @id_full was already donated}}
    %2, %ctrl_2 = ifrt.Call @id_full(%arg0) on devices [0, 1]
        : (!array) -> !array
    return %1, %2 : !shard0, !array
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }

  func.func private @id_full(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard0_2d = !ifrt.array<tensor<1x1xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
module @remap_overlapping_shard_aliases_to_bitcast_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0_2d, !shard0_2d)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard0
    %1, %ctrl_1 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard0
    // expected-error @+1 {{'ifrt.BitcastArrays' op input #1 of op}}
    %2, %3, %ctrl_2 = ifrt.BitcastArrays(%0, %1) <donated=true>
        : (!shard0, !shard0) -> (!shard0_2d, !shard0_2d)
    return %2, %3 : !shard0_2d, !shard0_2d
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @remap_alias_program_arg_not_donated_error {
  func.func @main(%arg0: !array) -> !shard0 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard0
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %1, %ctrl_1 = ifrt.Call @id_shard(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    return %1 : !shard0
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
// CHECK-LABEL: @disassemble_donate_one_shard_return_other
module @disassemble_donate_one_shard_return_other {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) : (!array) -> (!shard0, !shard1)
    %2, %ctrl_1 = ifrt.Call @id_shard(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    return %2, %1 : !shard0, !shard1
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @disassemble_donate_shard_then_use_original_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !array)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) : (!array) -> (!shard0, !shard1)
    %2, %ctrl_1 = ifrt.Call @id_shard(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    // expected-error @+1 {{'ifrt.Call' op input #0 of @id_full was already donated}}
    %3, %ctrl_2 = ifrt.Call @id_full(%arg0) on devices [0, 1]
        : (!array) -> !array
    return %2, %3 : !shard0, !array
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }

  func.func private @id_full(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @assemble_program_arg_not_donated_error {
  func.func @main(%arg0: !shard0 {ifrt.donated}, %arg1: !shard1) -> !array
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Assemble(%arg0, %arg1) : (!shard0, !shard1) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    return %1 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array01 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array02 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 2]>
module @copy_arrays_reuse_partially_overlapping_devices_undonated_error {
  func.func @main(%arg0: !array01) -> (!array01, !array02)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array01) -> !array02
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 2]
        <donated_input_indices=[0]> : (!array02) -> !array02
    return %arg0, %1 : !array01, !array02
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array01 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array02 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 2]>
module @copy_arrays_reuse_partially_overlapping_devices_use_after_donate_error {
  func.func @main(%arg0: !array01 {ifrt.donated}) -> (!array01, !array02)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array01) -> !array02
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 2]
        <donated_input_indices=[0]> : (!array02) -> !array02
    // expected-error @+1 {{'func.return' op result #0 of op at}}
    return %arg0, %1 : !array01, !array02
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array01 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array02 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 2]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
// CHECK-LABEL: @copy_arrays_reuse_partially_overlapping_devices_disjoint_shards
module @copy_arrays_reuse_partially_overlapping_devices_disjoint_shards {
  func.func @main(%arg0: !array01 {ifrt.donated}) -> (!array02, !shard1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array01) -> !array02
    %1, %ctrl_1 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[1:2:1] to [0:1:1]>]>]
        : (!array01) -> !shard1
    %2, %ctrl_2 = ifrt.Call @identity(%0) on devices [0, 2]
        <donated_input_indices=[0]> : (!array02) -> !array02
    %3, %ctrl_3 = ifrt.Call @id_shard(%1) on devices [1]
        <donated_input_indices=[0]> : (!shard1) -> !shard1
    return %2, %3 : !array02, !shard1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_donated_to_assemble_not_donated_to_program_error {
  func.func @main(%arg0: !shard0 {ifrt.donated}, %arg1: !shard1) -> !array
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Assemble' op input #1 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Assemble(%arg0, %arg1) <donated=true>
        : (!shard0, !shard1) -> !array
    return %0 : !array
  }
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @donate_to_assemble_then_return_input_error {
  func.func @main(%arg0: !shard0 {ifrt.donated}, %arg1: !shard1 {ifrt.donated})
      -> (!array, !shard1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Assemble(%arg0, %arg1) <donated=true>
        : (!shard0, !shard1) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %0, %arg1 : !array, !shard1
  }
}

// -----

!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
// CHECK-LABEL: @donate_to_assemble_and_donate_output
module @donate_to_assemble_and_donate_output {
  func.func @main(%arg0: !shard0 {ifrt.donated}, %arg1: !shard1 {ifrt.donated})
      -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Assemble(%arg0, %arg1) <donated=true>
        : (!shard0, !shard1) -> !array
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    return %1 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @arg_donated_to_disassemble_not_donated_to_program_error {
  func.func @main(%arg0: !array) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Disassemble' op input #0 has not been donated to the program.}}
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) <donated=true>
        : (!array) -> (!shard0, !shard1)
    return %0, %1 : !shard0, !shard1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @donate_to_disassemble_then_return_input_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !array)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) <donated=true>
        : (!array) -> (!shard0, !shard1)
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %0, %arg0 : !shard0, !array
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
// CHECK-LABEL: @donate_to_disassemble_and_donate_output_shard
module @donate_to_disassemble_and_donate_output_shard {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) <donated=true>
        : (!array) -> (!shard0, !shard1)
    %2, %ctrl_1 = ifrt.Call @id_shard(%0) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    return %2, %1 : !shard0, !shard1
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @donate_aliased_input_to_disassemble_not_donated_to_program_error {
  func.func @main(%arg0: !array) -> (!shard0, !shard1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    // expected-error @+1 {{'ifrt.Disassemble' op input #0 has not been donated to the program.}}
    %1, %2, %ctrl_1 = ifrt.Disassemble(%0) <donated=true>
        : (!array) -> (!shard0, !shard1)
    return %1, %2 : !shard0, !shard1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_aliased_to_call_not_donated_to_program_error {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0, 1]
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    return %0 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_duplicate_aliased_arg_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #1 of @callee was already donated or aliased}}
    %0, %1, %ctrl_0 = ifrt.Call @callee(%arg0, %arg0) on devices [0, 1]
        <io_aliases=[array<i32: 0, 0>, array<i32: 1, 1>]>
        : (!array, !array) -> (!array, !array)
    return %0, %1 : !array, !array
  }

  func.func private @callee(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
      -> (tensor<2xi32>, tensor<2xi32>) {
    return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_same_arg_aliased_and_donated_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #1 of @callee was already donated or aliased}}
    %0, %ctrl_0 = ifrt.Call @callee(%arg0, %arg0) on devices [0, 1]
        <donated_input_indices=[1], io_aliases=[array<i32: 0, 0>]>
        : (!array, !array) -> !array
    return %0 : !array
  }

  func.func private @callee(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
      -> tensor<2xi32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_loaded_executable_duplicate_aliased_arg_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.CallLoadedExecutable' op input #1 of @callee was already donated or aliased}}
    %0, %1, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg0)
        <io_aliases=[array<i32: 0, 0>, array<i32: 1, 1>]>
        : (!array, !array) -> (!array, !array)
    return %0, %1 : !array, !array
  }

  ifrt.LoadedExecutable @callee on devices [0, 1]
      : (!array, !array) -> (!array, !array)
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_loaded_executable_use_after_alias_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
        <io_aliases=[array<i32: 0, 0>]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %0, %arg0 : !array, !array
  }

  ifrt.LoadedExecutable @callee on devices [0, 1] : (!array) -> !array
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @disassemble_then_donate_whole_original_invalidates_shard_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !shard1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Disassemble(%arg0) : (!array) -> (!shard0, !shard1)
    %2, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %2, %1 : !array, !shard1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @whole_alias_before_disassemble_then_donate_shard_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    %1, %2, %ctrl_1 = ifrt.Disassemble(%arg0) : (!array) -> (!shard0, !shard1)
    %3, %ctrl_2 = ifrt.Call @id_shard(%1) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %3, %0 : !shard0, !array
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array_default = !ifrt.array<tensor<2xi32>,
                             #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array_pinned_host = !ifrt.array<tensor<2xi32>,
                                 #ifrt.sharding_param<2 to [0] on 2>, [0, 1],
                                 memory_kind = "pinned_host">
// CHECK-LABEL: @copy_arrays_reuse_different_memory_kind_no_alias
module @copy_arrays_reuse_different_memory_kind_no_alias {
  func.func @main(%arg0: !array_default) -> (!array_default, !array_pinned_host)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true>
        : (!array_default) -> !array_pinned_host
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array_pinned_host) -> !array_pinned_host
    return %arg0, %1 : !array_default, !array_pinned_host
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array_implicit = !ifrt.array<tensor<2xi32>,
                              #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array_device = !ifrt.array<tensor<2xi32>,
                            #ifrt.sharding_param<2 to [0] on 2>, [0, 1],
                            memory_kind = "device">
module @copy_arrays_reuse_implicit_and_explicit_device_memory_kind_alias_error {
  func.func @main(%arg0: !array_implicit {ifrt.donated})
      -> (!array_device, !array_implicit) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true>
        : (!array_implicit) -> !array_device
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array_device) -> !array_device
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %1, %arg0 : !array_device, !array_implicit
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
module @remap_donated_whole_alias_duplicate_root_output_alias_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!shard0, !shard0)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    %1, %2, %ctrl_1 = ifrt.RemapArrays(%arg0, %0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                  #ifrt.array_mapping<1, 1, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        <donated=true> : (!array, !array) -> (!shard0, !shard0)
    %3, %ctrl_2 = ifrt.Call @id_shard(%1) on devices [0]
        <donated_input_indices=[0]> : (!shard0) -> !shard0
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %3, %2 : !shard0, !shard0
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard1 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [1]>
module @remap_overlapping_shard_aliases_to_assemble_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> !array
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard0
    %1, %ctrl_1 = ifrt.RemapArrays(%arg0)
        mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>]
        : (!array) -> !shard1
    // expected-error @+1 {{'ifrt.Assemble' op input #1 of op}}
    %2, %ctrl_2 = ifrt.Assemble(%0, %1) <donated=true>
        : (!shard0, !shard1) -> !array
    return %2 : !array
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_donated_duplicate_input_output_alias_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg0) <donated=true>
        : (!array, !array) -> (!array, !array)
    %2, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %2, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_donated_whole_alias_duplicate_root_output_alias_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) <reuse=true> : (!array) -> !array
    %1, %2, %ctrl_1 = ifrt.CopyArrays(%arg0, %0) <donated=true>
        : (!array, !array) -> (!array, !array)
    %3, %ctrl_2 = ifrt.Call @identity(%1) on devices [0, 1]
        <donated_input_indices=[0]> : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %3, %2 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array01 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array02 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 2]>
module @copy_arrays_donated_partially_overlapping_devices_duplicate_input_error {
  func.func @main(%arg0: !array01 {ifrt.donated}) -> (!array02, !array02)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg0) <donated=true>
        : (!array01, !array01) -> (!array02, !array02)
    %2, %ctrl_1 = ifrt.Call @identity(%0) on devices [0, 2]
        <donated_input_indices=[0]> : (!array02) -> !array02
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %2, %1 : !array02, !array02
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array01 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array02 = !ifrt.array<tensor<2xi32>,
                       #ifrt.sharding_param<2 to [0] on 2>, [0, 2]>
!shard0 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [0]>
!shard2 = !ifrt.array<tensor<1xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [2]>
// CHECK-LABEL: @copy_arrays_donated_partially_overlapping_devices_disjoint_copied_shards
module @copy_arrays_donated_partially_overlapping_devices_disjoint_copied_shards {
  func.func @main(%arg0: !array01 {ifrt.donated}) -> (!shard0, !shard2, !shard2)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg0) <donated=true>
        : (!array01, !array01) -> (!array02, !array02)
    %s0_a, %s2_a, %ctrl_1 = ifrt.Disassemble(%0)
        : (!array02) -> (!shard0, !shard2)
    %s0_b, %s2_b, %ctrl_2 = ifrt.Disassemble(%1)
        : (!array02) -> (!shard0, !shard2)
    %2, %ctrl_3 = ifrt.Call @id_shard(%s2_a) on devices [2]
        <donated_input_indices=[0]> : (!shard2) -> !shard2
    %3, %ctrl_4 = ifrt.Call @id_shard(%s2_b) on devices [2]
        <donated_input_indices=[0]> : (!shard2) -> !shard2
    return %s0_a, %2, %3 : !shard0, !shard2, !shard2
  }

  func.func private @id_shard(%arg0: tensor<1xi32>) -> tensor<1xi32> {
    return %arg0 : tensor<1xi32>
  }
}
