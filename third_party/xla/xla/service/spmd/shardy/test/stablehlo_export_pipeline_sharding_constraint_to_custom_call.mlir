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
// RUN: sdy_opt %s -xla-sdy-export-ops='keep-hlo-sharding-constraints=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @sharding_constraint_to_sharding_custom_call
func.func @sharding_constraint_to_sharding_custom_call(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %0 = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {?}]>]>}
  // CHECK-SAME: (tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
