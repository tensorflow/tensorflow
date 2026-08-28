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
// RUN: mlir-hlo-opt -mhlo-test-materialize-broadcasts -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @clampBroadcast
// CHECK-SAME: (%[[MIN:.+]]: tensor<f32>, %[[VAL:.+]]: tensor<4xf32>, %[[MAX:.+]]: tensor<f32>)
func.func @clampBroadcast(%min: tensor<f32>, %value: tensor<4xf32>, %max: tensor<f32>) -> tensor<4xf32> {
  // CHECK-DAG: %[[MIN_BC:.+]] = "mhlo.broadcast"(%[[MIN]]) <{broadcast_sizes = dense<4> : tensor<1xi64>}> : (tensor<f32>) -> tensor<4xf32>
  // CHECK-DAG: %[[MAX_BC:.+]] = "mhlo.broadcast"(%[[MAX]]) <{broadcast_sizes = dense<4> : tensor<1xi64>}> : (tensor<f32>) -> tensor<4xf32>
  // CHECK: mhlo.clamp %[[MIN_BC]], %[[VAL]], %[[MAX_BC]] : tensor<4xf32>
  %0 = "mhlo.clamp"(%min, %value, %max) : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
