// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: stablehlo-quant-opt %s -tf-stablehlo-remove-sharding-custom-call \
// RUN:   -split-input-file | FileCheck %s

// CHECK-LABEL: sharding_custom_call_removed
func.func @sharding_custom_call_removed(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %1 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = ""} : (tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}
// CHECK-NOT: custom_call

// -----

// Tests that a custom_call that is not @Sharding is not removed.

// CHECK-LABEL: custom_call_not_removed
func.func @custom_call_not_removed(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %1 = stablehlo.custom_call @NotSharding(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}
// CHECK: custom_call @NotSharding
