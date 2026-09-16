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
// RUN: mlir-hlo-opt %s -gpu-kernel-to-nvvm | FileCheck %s

gpu.module @test_module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  gpu.func @test_kernel(%out: memref<32xf32>) kernel {
    %0 = gpu.block_id x
    %cst = arith.constant 0.0 : f32
    memref.store %cst, %out[%0] : memref<32xf32>
    gpu.return
  }
}

// CHECK-LABEL:  gpu.module @test_module
// CHECK-SAME:     attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
// CHECK-NEXT:    llvm.func @test_kernel
// CHECK-SAME         attributes {gpu.kernel, nvvm.kernel}
// CHECK:           %[[VAR:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
