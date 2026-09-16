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
// RUN: not mlir-bisect %s \
// RUN: --pass-pipeline="builtin.module(test-break-linalg-transpose)" \
// RUN: | FileCheck %s

func.func @main() -> memref<2x2xindex> {
  %a = memref.alloc() : memref<2x2xindex>
  return %a : memref<2x2xindex>
}

// CHECK: Did not find bug in initial module
