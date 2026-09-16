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
// RUN: xla-opt %s --triton-xla-pipeline='target=9.0' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CUDA
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=ROCM

// CHECK: module attributes
// CUDA: ttg.target = "cuda:90"
// ROCM: ttg.target = "hip:gfx950"

// CHECK: llvm.func @func
tt.func @func() {
  // CHECK: llvm.return
  tt.return
}

