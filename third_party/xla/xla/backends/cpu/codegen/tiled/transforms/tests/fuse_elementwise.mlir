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
// RUN: fusion_compiler_opt %s -split-input-file  \
// RUN:   -linalg-generalize-named-ops -xtile-cpu-fuse-elementwise | FileCheck %s

func.func @elementwise_add_to_vector(
    %lhs : tensor<8x1024xf32>,
    %rhs : tensor<8x1024xf32>) -> tensor<8x1024xf32> {
  %out = tensor.empty() : tensor<8x1024xf32>

  %intermediate = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%lhs, %rhs : tensor<8x1024xf32>, tensor<8x1024xf32>)
    outs(%out : tensor<8x1024xf32>) -> tensor<8x1024xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%intermediate, %rhs : tensor<8x1024xf32>, tensor<8x1024xf32>)
    outs(%out : tensor<8x1024xf32>) -> tensor<8x1024xf32>
  return %result : tensor<8x1024xf32>
}

// CHECK: linalg.generic
// CHECK:    (%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK:       %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f32
// CHECK:       %[[RES:.*]] = arith.addf %[[MUL]], %[[RHS]] : f32
// CHECK:       linalg.yield %[[RES]] : f32
// CHECK:     }
