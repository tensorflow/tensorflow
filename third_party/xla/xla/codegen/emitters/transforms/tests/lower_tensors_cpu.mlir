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
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

func.func @load_non_gep_from_args(%arg0: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.getelementptr inbounds %arg0[1]
    : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
  func.return %2 : !llvm.ptr
}

// CHECK-LABEL: @load_non_gep_from_args
// CHECK-NEXT:    %0 = llvm.getelementptr inbounds %arg0[1]
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    return %2 : !llvm.ptr
