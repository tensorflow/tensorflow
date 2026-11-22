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

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
func.func @good_after(%arg0: !array, %arg1: !array) attributes {ifrt.function} {
  %ctrl_0 = "ifrt.After"(%arg0) : (!array) -> !ifrt.control
  %ctrl_1 = "ifrt.After"(%arg0, %arg1) : (!array, !array) -> !ifrt.control
  return
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
func.func @after_requires_in_ifrt_function(%arg0: !array) {
  // expected-error@+1 {{'ifrt.After' op must be in a FuncOp with attr `ifrt.function`}}
  %ctrl = "ifrt.After"(%arg0) : (!array) -> !ifrt.control
  return
}

// -----

func.func @after_requires_at_least_one_input() attributes {ifrt.function} {
  // expected-error@+1 {{'ifrt.After' op requires at least one input array}}
  %ctrl = "ifrt.After"() : () -> !ifrt.control
  return
}
