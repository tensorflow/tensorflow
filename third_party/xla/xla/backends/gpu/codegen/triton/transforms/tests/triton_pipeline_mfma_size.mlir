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
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942' \
// RUN:   | FileCheck %s --check-prefix=CHECK-AUTO
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942 mfma-size=16' \
// RUN:   | FileCheck %s --check-prefix=CHECK-16

// Auto picks 32x32 on a 64x64 tile, so seeing 16x16 proves mfma-size reached
// AccelerateMatmul.

func.func @dot(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
  %lhs = triton_xla.extract from %arg0
      as memref<64x64xf16, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf16>
  %rhs = triton_xla.extract from %arg1
      as memref<64x64xf16, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf16>
  %acc = arith.constant dense<0.0> : tensor<64x64xf32>
  %res = tt.dot %lhs, %rhs, %acc : tensor<64x64xf16> * tensor<64x64xf16>
      -> tensor<64x64xf32>
  triton_xla.insert %res into %arg2
      as memref<64x64xf32, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf32>
  func.return
}

// CHECK-AUTO-LABEL: llvm.func @dot
// CHECK-AUTO:       rocdl.mfma.f32.32x32x{{[0-9]+}}f16

// CHECK-16-LABEL:   llvm.func @dot
// CHECK-16:         rocdl.mfma.f32.16x16x{{[0-9]+}}f16
