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

module @axpy {
  func.func public @main(%alpha: tensor<f32>, %x: tensor<4 x f32>, %y: tensor<4 x f32>) -> tensor<4 x f32> {
    %a = "stablehlo.broadcast_in_dim" (%alpha) {
      broadcast_dimensions = array<i64>
    } : (tensor<f32>) -> tensor<4 x f32>
    %ax = stablehlo.multiply %a, %x : tensor<4 x f32>
    %result = stablehlo.add %ax, %y : tensor<4 x f32>
    return %result: tensor<4 x f32>
  }
}
