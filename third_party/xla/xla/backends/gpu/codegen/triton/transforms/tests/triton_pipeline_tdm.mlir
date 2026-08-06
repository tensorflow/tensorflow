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
// RUN: xla-opt %s -split-input-file --triton-xla-pipeline='target=gfx1250' \
// RUN:   | FileCheck %s --check-prefix=CHECK-TDM
//
// RUN: xla-opt %s -split-input-file --triton-xla-pipeline='target=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOTDM

// Verifies that the full Triton XLA + AMD lowering pipeline emits TDM
// intrinsics on gfx1250 and pointer-arithmetic buffer ops on non-TDM arches.

func.func @lower_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @lower_extract_insert
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @lower_extract_insert
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load

// -----

func.func @batched_dot(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>,
                       %arg2: !tt.ptr<bf16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>

  %lhs_tile = triton_xla.extract from %arg0
      as memref<4x32x256xbf16, #xtile.layout<[2, 1, 0]>>
      [2, 0, 0] [1, 32, 256] [1, 1, 1] : tensor<32x256xbf16>

  %rhs_tile = triton_xla.extract from %arg1
      as memref<4x256x32xbf16, #xtile.layout<[2, 1, 0]>>
      [2, 0, 0] [1, 256, 32] [1, 1, 1] : tensor<256x32xbf16>

  %dot = tt.dot %lhs_tile, %rhs_tile, %cst, inputPrecision = tf32
      : tensor<32x256xbf16> * tensor<256x32xbf16> -> tensor<32x32xf32>
  %dot_bf16 = arith.truncf %dot : tensor<32x32xf32> to tensor<32x32xbf16>

  triton_xla.insert %dot_bf16 into %arg2
      as memref<32x32xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [32, 32] [1, 1] : tensor<32x32xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @batched_dot
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @batched_dot
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load

// -----

func.func @lower_extract_insert_3d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [32, 256, 4] [1, 1, 1] : tensor<32x256x4xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [32, 256, 4] [1, 1, 1] : tensor<32x256x4xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @lower_extract_insert_3d
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @lower_extract_insert_3d
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load
