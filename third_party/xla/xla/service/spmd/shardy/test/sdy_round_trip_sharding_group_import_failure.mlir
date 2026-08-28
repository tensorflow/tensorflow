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
// RUN: sdy_opt %s -xla-sdy-import-sdy-custom-calls -split-input-file -verify-diagnostics

func.func @sharding_group_import_failure_if_no_group_id(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  // expected-error @+1 {{expected CustomCallOp with a sharding group id.}}
  stablehlo.custom_call @xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {}} : (tensor<16x16xf32>) -> ()
  return %arg0 : tensor<16x16xf32>
}

// -----

func.func @sharding_group_import_with_used_result(%arg0: tensor<8x8xf32>) -> tuple<tuple<>> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  // expected-error @+1 {{xla.sdy.ShardingGroup CustomCallOp should have no uses.}}
  %0 = stablehlo.custom_call @xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "21 : i64"}} : (tensor<8x8xf32>) -> tuple<>
  %1 = "stablehlo.tuple"(%0) : (tuple<>) -> tuple<tuple<>>
  return %1 : tuple<tuple<>>
}
