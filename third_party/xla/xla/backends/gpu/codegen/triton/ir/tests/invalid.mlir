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
// RUN: xla-opt --split-input-file --verify-diagnostics %s

func.func @extract_0d(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{unsupported 0-d tensor}}
  %0 = triton_xla.extract from %arg0 as memref<bf16, #xtile.layout<[]>> [][][] : tensor<bf16>
  return
}

// -----

func.func @insert_0d(%arg0: tensor<bf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{unsupported 0-d tensor}}
  triton_xla.insert %arg0 into %arg1 as memref<bf16, #xtile.layout<[]>> [][][] : tensor<bf16>
  return
}

// -----

func.func @extract_wrong_layout(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout has 0 dimensions, but shape has 1}}
  %0 = triton_xla.extract from %arg0 as memref<8xbf16, #xtile.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_layout(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{layout has 0 dimensions, but shape has 1}}
  triton_xla.insert %arg0 into %arg1 as memref<8xbf16, #xtile.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @extract_wrong_rank(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{expected 0 offset values, got 1}}
  %0 = triton_xla.extract from %arg0 as memref<bf16, #xtile.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_rank(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{expected 0 offset values, got 1}}
  triton_xla.insert %arg0 into %arg1 as memref<bf16, #xtile.layout<[]>> [0][8][1] : tensor<8xbf16>
  return
}

// -----

func.func @extract_wrong_shape(%arg0: !tt.ptr<bf16>) {
  // expected-error @+1 {{expected type to be 'tensor<16xbf16>'}}
  %0 = triton_xla.extract from %arg0 as memref<16xbf16, #xtile.layout<[0]>> [0][16][1] : tensor<8xbf16>
  return
}

// -----

func.func @insert_wrong_shape(%arg0: tensor<8xbf16>, %arg1: !tt.ptr<bf16>) {
  // expected-error @+1 {{expected type to be 'tensor<16xbf16>'}}
  triton_xla.insert %arg0 into %arg1 as memref<16xbf16, #xtile.layout<[0]>> [0][16][1] : tensor<8xbf16>
  return
}
