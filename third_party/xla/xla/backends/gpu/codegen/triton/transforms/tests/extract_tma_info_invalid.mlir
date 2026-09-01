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
// RUN: xla-opt %s -split-input-file \
// RUN: -extract-tma-info \
// RUN: --verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
tt.func @extract_tma_info_no_tma_descriptor(
// expected-error @+1 {{Argument of type tt.tensordesc must have attribute tt.tma_descriptor}}
  %arg0: !tt.tensordesc<16x32xf32, #shared>
  {tt.nv_tma_desc = 1 : i32}) {
  tt.return
}
