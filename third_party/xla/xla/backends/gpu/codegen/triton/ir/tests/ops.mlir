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
// RUN: xla-opt %s | FileCheck %s

// Verify the printed output can be parsed.
// RUN: xla-opt %s | xla-opt --split-input-file | FileCheck %s

// Verify the generic form can be parsed.
// RUN: xla-opt %s --mlir-print-op-generic | xla-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @xla_triton_extract
tt.func @xla_triton_extract(%src: !tt.ptr<bf16>, %i : index) -> tensor<16x64xbf16> {
  // CHECK: triton_xla.extract
  %extracted_tensor = triton_xla.extract from %src
    as memref<512x1x128xbf16, #xtile.layout<[2, 1, 0]>>
    [0, 0, %i] [16, 1, 64] [128, 1, 1] : tensor<16x64xbf16>
  tt.return %extracted_tensor : tensor<16x64xbf16>
}

// CHECK-LABEL: @xla_triton_insert
tt.func @xla_triton_insert(%src: tensor<16x64xbf16>, %dst: !tt.ptr<bf16>, %j: index) {
  // CHECK: triton_xla.insert
  triton_xla.insert %src into %dst
    as memref<512x128xbf16, #xtile.layout<[0, 1]>>
    [%j, 0][16, 64][1, 1] : tensor<16x64xbf16>
  tt.return
}
