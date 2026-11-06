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
// RUN: ifrt-opt %s -ifrt-reshard-to-copy-arrays='on-remaining-reshards=error' -split-input-file -verify-diagnostics 2>&1

#sharding_src = #ifrt.sharding_param<2x1 to [0] on 2>
#sharding_dst = #ifrt.sharding_param<1x1 to [0] on 2>
!array0 = !ifrt.array<tensor<2x2xi32>, #sharding_src, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #sharding_dst, [2, 3]>

module @single_reshard_remaining {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // expected-error @+1 {{Detected `ifrt.Reshard` op remaining after converting reshards to `ifrt.CopyArrays`; inputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>], outputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>]}}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}

// -----

#sharding_src = #ifrt.sharding_param<2x1 to [0] on 2>
#sharding_dst = #ifrt.sharding_param<1x1 to [0] on 2>
!array0 = !ifrt.array<tensor<2x2xi32>, #sharding_src, [0, 1]>
!array1_copy = !ifrt.array<tensor<2x2xi32>, #sharding_src, [2, 3]>
!array1_reshard = !ifrt.array<tensor<2x2xi32>, #sharding_dst, [2, 3]>

module @mixed_copy_and_reshard_remaining {
  func.func @main(%arg0: !array0, %arg1: !array0)
      -> (!array1_copy, !array1_reshard) attributes {ifrt.function} {
    // expected-error @+1 {{Detected `ifrt.Reshard` op remaining after converting reshards to `ifrt.CopyArrays`; inputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>], outputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>]}}
    %0, %1, %ctrl_0 = ifrt.Reshard(%arg0, %arg1)
        : (!array0, !array0) -> (!array1_copy, !array1_reshard)
    return %0, %1 : !array1_copy, !array1_reshard
  }
}

// -----

#sharding_a = #ifrt.sharding_param<2x1 to [0] on 2>
#sharding_b = #ifrt.sharding_param<1x1 to [0] on 2>
!array0 = !ifrt.array<tensor<2x2xi32>, #sharding_a, [0, 1]>
!array1_copy = !ifrt.array<tensor<2x2xi32>, #sharding_a, [2, 3]>
!array1_reshard = !ifrt.array<tensor<2x2xi32>, #sharding_b, [2, 3]>
!array2_reshard = !ifrt.array<tensor<2x2xi32>, #sharding_a, [4, 5]>

module @multiple_reshards_remaining {
  func.func @main(%arg0: !array0, %arg1: !array0)
      -> (!array1_copy, !array1_reshard, !array2_reshard)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1_copy
    // expected-error @+1 {{Detected `ifrt.Reshard` op remaining after converting reshards to `ifrt.CopyArrays`; inputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>], outputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>]}}
    %1, %ctrl_1 = ifrt.Reshard(%arg1) : (!array0) -> !array1_reshard
    // expected-error @+1 {{Detected `ifrt.Reshard` op remaining after converting reshards to `ifrt.CopyArrays`; inputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>], outputs: [!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [4, 5]>]}}
    %2, %ctrl_2 = ifrt.Reshard(%1) : (!array1_reshard) -> !array2_reshard
    return %0, %1, %2 : !array1_copy, !array1_reshard, !array2_reshard
  }
}

// -----

#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
!array0 = !ifrt.array<tensor<2x2xi32>, #sharding, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #sharding, [2, 3]>

module @all_reshards_converted_to_copy_arrays {
  func.func @main(%arg0: !array0, %arg1: !array0) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.Reshard(%arg0, %arg1)
        : (!array0, !array0) -> (!array1, !array1)
    return %0, %1 : !array1, !array1
  }
}
