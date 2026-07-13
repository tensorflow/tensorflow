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
// RUN: emitters_opt %s -split-input-file -xla-gpu-promote-shuffle-to-dpp | FileCheck %s

module {
  func.func @shuffle_down_1(%arg0: f32) -> f32 {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c1, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_1
// CHECK-SAME: (%[[ARG:.*]]: f32)
// CHECK: %[[DPP:.*]] = amdgpu.dpp %[[ARG]] %[[ARG]] row_shl(1 : i32) {bound_ctrl = true} : f32
// CHECK: return %[[DPP]] : f32

// -----

module {
  func.func @shuffle_down_4(%arg0: f32) -> f32 {
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c4, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_4
// CHECK: amdgpu.dpp %{{.*}} %{{.*}} row_shl(4 : i32) {bound_ctrl = true} : f32

// -----

module {
  func.func @shuffle_down_8(%arg0: f32) -> f32 {
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c8, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_8
// CHECK: amdgpu.dpp %{{.*}} %{{.*}} row_shl(8 : i32) {bound_ctrl = true} : f32

// -----

module {
  func.func @shuffle_down_15(%arg0: f32) -> f32 {
    %c15 = arith.constant 15 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c15, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_15
// CHECK: amdgpu.dpp %{{.*}} %{{.*}} row_shl(15 : i32) {bound_ctrl = true} : f32

// -----

// Offset 16 is promoted to swizzle_bitmode with XOR semantics (ds_swizzle).
module {
  func.func @shuffle_down_16_swizzle(%arg0: f32) -> f32 {
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c16, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_16_swizzle
// CHECK-SAME: (%[[ARG:.*]]: f32)
// CHECK: %[[SW:.*]] = amdgpu.swizzle_bitmode %[[ARG]] 31 0 16 : f32
// CHECK: return %[[SW]] : f32
// CHECK-NOT: gpu.shuffle
// CHECK-NOT: amdgpu.dpp

// -----

// XOR mode should NOT be promoted by this pass.
module {
  func.func @shuffle_xor_not_promoted(%arg0: f32) -> f32 {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle xor %arg0, %c1, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_xor_not_promoted
// CHECK: gpu.shuffle xor
// CHECK-NOT: amdgpu.dpp

// -----

// When the validity predicate IS used, do NOT promote.
module {
  func.func @shuffle_down_valid_used(%arg0: f32) -> (f32, i1) {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c1, %c64 : f32
    return %shfl, %valid : f32, i1
  }
}

// CHECK-LABEL: @shuffle_down_valid_used
// CHECK: gpu.shuffle down
// CHECK-NOT: amdgpu.dpp

// -----

// i32 type should also work.
module {
  func.func @shuffle_down_i32(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c2, %c64 : i32
    return %shfl : i32
  }
}

// CHECK-LABEL: @shuffle_down_i32
// CHECK: amdgpu.dpp %{{.*}} %{{.*}} row_shl(2 : i32) {bound_ctrl = true} : i32

// -----

// Offset 32 crosses the 32-lane swizzle boundary, should NOT be promoted.
module {
  func.func @shuffle_down_32_not_promoted(%arg0: f32) -> f32 {
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c32, %c64 : f32
    return %shfl : f32
  }
}

// CHECK-LABEL: @shuffle_down_32_not_promoted
// CHECK: gpu.shuffle down
// CHECK-NOT: amdgpu.dpp
// CHECK-NOT: amdgpu.swizzle_bitmode

// -----

// Offset 16 with validity predicate used should NOT be promoted to swizzle.
module {
  func.func @shuffle_down_16_valid_used(%arg0: f32) -> (f32, i1) {
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c16, %c64 : f32
    return %shfl, %valid : f32, i1
  }
}

// CHECK-LABEL: @shuffle_down_16_valid_used
// CHECK: gpu.shuffle down
// CHECK-NOT: amdgpu.swizzle_bitmode

// -----

// Offset 16 with i32 type should also work with swizzle.
module {
  func.func @shuffle_down_16_i32(%arg0: i32) -> i32 {
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %shfl, %valid = gpu.shuffle down %arg0, %c16, %c64 : i32
    return %shfl : i32
  }
}

// CHECK-LABEL: @shuffle_down_16_i32
// CHECK: amdgpu.swizzle_bitmode %{{.*}} 31 0 16 : i32
