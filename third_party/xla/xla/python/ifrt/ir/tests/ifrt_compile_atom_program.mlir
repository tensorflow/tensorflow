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
// RUN: ifrt-opt %s -ifrt-compile-atom-program -split-input-file | FileCheck %s

// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @call_hlo
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @call_hlo {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        {ifrt.module_type = "xla"} : (!array) -> !array
    return %0 : !array
  }

  // CHECK: ifrt.LoadedExecutable @
  // CHECK-SAME: on devices [0, 1]
  // CHECK: (!ifrt.array<tensor<2x2xi32>, #sp, [0, 1]>)
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #sp, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(
        %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
        -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @call_hlo_sdy_lowered
module @call_hlo_sdy_lowered attributes {
    ifrt.sdy.meshes ="{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"} {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @{{.*}}(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        {ifrt.module_type = "xla"} : (!array) -> !array
    return %0 : !array
  }

  // module @add_one attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"}, sym_visibility = "private"}
  // CHECK: ifrt.LoadedExecutable @
  // CHECK-SAME: on devices [0, 1]
  // CHECK: (!ifrt.array<tensor<2x2xi32>, #sp, [0, 1]>)
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #sp, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(
        %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
        -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
// CHECK: #sp = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @call_hlo_multiple_calls_same_target_module
module @call_hlo_multiple_calls_same_target_module {
  func.func @main(%arg0: !array0, %arg1: !array1) -> (!array0, !array1) attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @{{.*}}(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        {ifrt.module_type = "xla"} : (!array0) -> !array0
    // CHECK: ifrt.CallLoadedExecutable @{{.*}}(%arg1)
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%arg1) on devices [2,3]
        {ifrt.module_type = "xla"} : (!array1) -> !array1
    return %0, %1 : !array0, !array1
  }

  // CHECK: ifrt.LoadedExecutable @
  // CHECK-SAME: on devices [2, 3]
  // CHECK: ifrt.LoadedExecutable @
  // CHECK-SAME: on devices [0, 1]
  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(
        %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
        -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

