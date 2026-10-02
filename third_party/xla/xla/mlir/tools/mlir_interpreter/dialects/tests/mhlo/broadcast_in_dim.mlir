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
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @broadcast() -> tensor<2x3xui16> {
  %0 = mhlo.constant dense<[1, 2]> : tensor<2xui16>
  %1 = "mhlo.broadcast_in_dim"(%0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<2xui16>) -> tensor<2x3xui16>
  return %1 : tensor<2x3xui16>
}

// CHECK{LITERAL}: [[1, 1, 1], [2, 2, 2]]

func.func @zero_rank() -> tensor<1x2x3xi32> {
  %in = mhlo.constant dense<1> : tensor<i32>
  %0 = "mhlo.broadcast_in_dim"(%in) {
    broadcast_dimensions = dense<[]> : tensor<0xi64>
  } : (tensor<i32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// CHECK{LITERAL}: [[[1, 1, 1], [1, 1, 1]]]
