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
// RUN: xla-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: xla_triton_extract_insert
tt.func @xla_triton_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: index) {
  %c0 = arith.constant 0 : index
  // CHECK:       triton_xla.extract
  // CHECK-SAME:    [%arg1, 0] [16, 64] [128, 1]
  // CHECK-SAME:    {noinline = false}
  %tile = triton_xla.extract from %arg0
      as memref<512x128xbf16, #xtile.layout<[1, 0]>>
      [%arg1, %c0] [16, 64] [128, 1] {noinline = false} : tensor<16x64xbf16>
  // CHECK:       triton_xla.insert
  // CHECK-SAME:    [0, %arg1] [16, 64] [1, 1]
  // CHECK-SAME:    {noinline = false}
  triton_xla.insert %tile into %arg0
      as memref<512x128xbf16, #xtile.layout<[1, 0]>>
      [%c0, %arg1][16, 64][1, 1] {noinline = false} : tensor<16x64xbf16>
  tt.return
}

// CHECK-LABEL: @fold_ptr_memref_ptr(
// CHECK-SAME: %[[SRC:.*]]: !tt.ptr<f32>
func.func @fold_ptr_memref_ptr(%src: !tt.ptr<f32>) -> !tt.ptr<f32> {
  // CHECK: return %[[SRC]] : !tt.ptr<f32>
  %src_ptr = triton_xla.ptr_to_memref %src from !tt.ptr<f32> to memref<256xf32>
  %dst = triton_xla.memref_to_ptr %src_ptr from memref<256xf32> to !tt.ptr<f32>
  func.return %dst : !tt.ptr<f32>
}
