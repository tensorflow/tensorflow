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

// RUN: litert-opt %s -tfl-fold-fp16-casts-into-fully-connected | FileCheck %s

// CHECK-LABEL: @FoldCasts
// CHECK-SAME: (%[[X:.*]]: tensor<4x64xf16>, %[[W:.*]]: tensor<32x64xf32>, %[[B:.*]]: tensor<32xf32>)
func.func @FoldCasts(%arg0: tensor<4x64xf16>, %w: tensor<32x64xf32>, %b: tensor<32xf32>) -> tensor<4x32xf16> {
  %0 = "tfl.cast"(%arg0) : (tensor<4x64xf16>) -> tensor<4x64xf32>
  %1 = "tfl.fully_connected"(%0, %w, %b) {fused_activation_function = "RELU", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<4x32xf32>
  %2 = "tfl.cast"(%1) : (tensor<4x32xf32>) -> tensor<4x32xf16>
  func.return %2 : tensor<4x32xf16>
  // CHECK-NOT: tfl.cast
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%[[X]], %[[W]], %[[B]]) <{fused_activation_function = "RELU", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x64xf16>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<4x32xf16>
  // CHECK-NOT: tfl.cast
  // CHECK: return %[[FC]]
}

// The input cast feeding several fully_connected ops (e.g. q/k/v projections)
// is folded into each of them.

// CHECK-LABEL: @FoldSharedInputCast
func.func @FoldSharedInputCast(%arg0: tensor<4x64xf16>, %w0: tensor<32x64xf32>, %w1: tensor<16x64xf32>) -> (tensor<4x32xf16>, tensor<4x16xf16>) {
  %none = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.cast"(%arg0) : (tensor<4x64xf16>) -> tensor<4x64xf32>
  %1 = "tfl.fully_connected"(%0, %w0, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  %2 = "tfl.cast"(%1) : (tensor<4x32xf32>) -> tensor<4x32xf16>
  %3 = "tfl.fully_connected"(%0, %w1, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<16x64xf32>, none) -> tensor<4x16xf32>
  %4 = "tfl.cast"(%3) : (tensor<4x16xf32>) -> tensor<4x16xf16>
  func.return %2, %4 : tensor<4x32xf16>, tensor<4x16xf16>
  // CHECK-NOT: tfl.cast
  // CHECK: "tfl.fully_connected"(%arg0, %arg1, %{{.*}}) {{.*}} -> tensor<4x32xf16>
  // CHECK: "tfl.fully_connected"(%arg0, %arg2, %{{.*}}) {{.*}} -> tensor<4x16xf16>
  // CHECK-NOT: tfl.cast
}

// The f32 result has another user, so the fully_connected must stay f32.

// CHECK-LABEL: @NoFoldOutputHasOtherUsers
func.func @NoFoldOutputHasOtherUsers(%arg0: tensor<4x64xf16>, %w: tensor<32x64xf32>) -> (tensor<4x32xf16>, tensor<4x32xf32>) {
  %none = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.cast"(%arg0) : (tensor<4x64xf16>) -> tensor<4x64xf32>
  %1 = "tfl.fully_connected"(%0, %w, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  %2 = "tfl.cast"(%1) : (tensor<4x32xf32>) -> tensor<4x32xf16>
  func.return %2, %1 : tensor<4x32xf16>, tensor<4x32xf32>
  // CHECK: "tfl.fully_connected"({{.*}}) {{.*}} -> tensor<4x32xf32>
  // CHECK: "tfl.cast"
}

// The activation is not cast from f16, so there is nothing to fold.

// CHECK-LABEL: @NoFoldInputNotFromF16
func.func @NoFoldInputNotFromF16(%arg0: tensor<4x64xf32>, %w: tensor<32x64xf32>) -> tensor<4x32xf16> {
  %none = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.fully_connected"(%arg0, %w, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  %1 = "tfl.cast"(%0) : (tensor<4x32xf32>) -> tensor<4x32xf16>
  func.return %1 : tensor<4x32xf16>
  // CHECK: "tfl.fully_connected"({{.*}}) {{.*}} -> tensor<4x32xf32>
  // CHECK: "tfl.cast"
}

// A bf16 input cast is a different precision contract and must not be folded.

// CHECK-LABEL: @NoFoldBf16
func.func @NoFoldBf16(%arg0: tensor<4x64xbf16>, %w: tensor<32x64xf32>) -> tensor<4x32xbf16> {
  %none = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.cast"(%arg0) : (tensor<4x64xbf16>) -> tensor<4x64xf32>
  %1 = "tfl.fully_connected"(%0, %w, %none) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x64xf32>, tensor<32x64xf32>, none) -> tensor<4x32xf32>
  %2 = "tfl.cast"(%1) : (tensor<4x32xf32>) -> tensor<4x32xbf16>
  func.return %2 : tensor<4x32xbf16>
  // CHECK: "tfl.cast"
  // CHECK: "tfl.fully_connected"({{.*}}) {{.*}} -> tensor<4x32xf32>
  // CHECK: "tfl.cast"
}
