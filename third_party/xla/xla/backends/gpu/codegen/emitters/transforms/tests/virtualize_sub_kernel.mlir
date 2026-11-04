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
// RUN: emitters_opt %s -split-input-file -verify-diagnostics \
// RUN:   -xla-gpu-virtualize-sub-kernel="entry_function_name=sub_kernel expected_num_tensor_args=2 block_dim_x=4 block_dim_y=2 block_dim_z=1 thread_dim_x=32 thread_dim_y=4 thread_dim_z=1 max_threads=256 warp_size=32" \
// RUN: | FileCheck %s

module {
  llvm.mlir.global private @smem_a() {addr_space = 3 : i32} : !llvm.array<64 x f32>
  llvm.mlir.global private @smem_b() {addr_space = 3 : i32} : !llvm.array<16 x f32>

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %scratch0: !llvm.ptr, %scratch1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    %bx = nvvm.read.ptx.sreg.ctaid.x : i32
    %by = nvvm.read.ptx.sreg.ctaid.y : i32
    %gx = nvvm.read.ptx.sreg.nctaid.x : i32
    %tx = nvvm.read.ptx.sreg.tid.x : i32
    %ty = nvvm.read.ptx.sreg.tid.y : i32
    %ntx = nvvm.read.ptx.sreg.ntid.x : i32
    %ptr_a = llvm.mlir.addressof @smem_a : !llvm.ptr<3>
    %ptr_b = llvm.mlir.addressof @smem_b : !llvm.ptr<3>
    nvvm.barrier
    %sum0 = llvm.add %bx, %by : i32
    %sum1 = llvm.add %sum0, %gx : i32
    %sum2 = llvm.add %tx, %ty : i32
    %sum3 = llvm.add %sum1, %sum2 : i32
    %sum4 = llvm.add %sum3, %ntx : i32
    llvm.store %sum4, %arg1 : i32, !llvm.ptr
    llvm.return
  }
}

// CHECK: module attributes {xla.shared_memory_bytes = 320 : i64}
// CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-NOT: @smem_a
// CHECK-NOT: @smem_b
// CHECK-LABEL: llvm.func @sub_kernel(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[REL_BID:.*]]: i32)
// CHECK-NOT: nvvm.kernel
// CHECK-NOT: xla.entry
// CHECK-DAG: %[[RAW_TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK-DAG: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK-DAG: %[[TX:.*]] = llvm.urem %[[RAW_TID]], %[[C32]] : i32
// CHECK-DAG: %[[TY:.*]] = llvm.udiv %[[RAW_TID]], %[[C32]] : i32
// CHECK-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-DAG: %[[BX:.*]] = llvm.urem %[[REL_BID]], %[[C4]] : i32
// CHECK-DAG: %[[BY:.*]] = llvm.udiv %[[REL_BID]], %[[C4]] : i32
// CHECK: %[[BASE_A:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
// CHECK: %[[BASE_B:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
// CHECK: %[[GEP_B:.*]] = llvm.getelementptr %[[BASE_B]][256] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: %[[BAR_ID:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[BAR_THREADS:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK: nvvm.barrier id = %[[BAR_ID]] number_of_threads = %[[BAR_THREADS]]
// CHECK: llvm.return

// -----

module {
  llvm.mlir.global private @smem_a() {addr_space = 3 : i32} : !llvm.array<16 x i8>
  llvm.mlir.global private @smem_b() {addr_space = 3 : i32, alignment = 128 : i64} : !llvm.array<32 x f32>

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    %ptr_a = llvm.mlir.addressof @smem_a : !llvm.ptr<3>
    %ptr_b = llvm.mlir.addressof @smem_b : !llvm.ptr<3>
    llvm.store %ptr_a, %arg0 : !llvm.ptr<3>, !llvm.ptr
    llvm.store %ptr_b, %arg1 : !llvm.ptr<3>, !llvm.ptr
    llvm.return
  }
}

// CHECK: module attributes {xla.shared_memory_bytes = 256 : i64}
// CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 128 : i64} : !llvm.array<0 x i8>
// CHECK-NOT: @smem_a
// CHECK-NOT: @smem_b
// CHECK-LABEL: llvm.func @sub_kernel(
// CHECK: %[[BASE_A:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
// CHECK: %[[BASE_B:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
// CHECK: %[[GEP_B:.*]] = llvm.getelementptr %[[BASE_B]][128] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: llvm.store %[[BASE_A]], %{{.*}} : !llvm.ptr<3>, !llvm.ptr
// CHECK: llvm.store %[[GEP_B]], %{{.*}} : !llvm.ptr<3>, !llvm.ptr
// CHECK: llvm.return

// -----

module {
  llvm.func @helper() {
    nvvm.barrier
    llvm.return
  }

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    llvm.call @helper() : () -> ()
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @helper()
// CHECK: %[[HELPER_BAR_ID:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[HELPER_BAR_THREADS:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK: nvvm.barrier id = %[[HELPER_BAR_ID]] number_of_threads = %[[HELPER_BAR_THREADS]]
// CHECK-LABEL: llvm.func @sub_kernel(

// -----

// expected-error @+1 {{Sub-kernel entry function 'sub_kernel' not found.}}
module {
  llvm.func @other_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    llvm.return
  }
}

// -----

module {
  llvm.func @helper_with_block_id() -> i32 {
    // expected-error @+1 {{Hardware ID and dimension ops in non-entry functions are not supported in horizontal fusion.}}
    %bx = nvvm.read.ptx.sreg.ctaid.x : i32
    llvm.return %bx : i32
  }

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    %0 = llvm.call @helper_with_block_id() : () -> i32
    llvm.return
  }
}

// -----

module {
  llvm.func @helper_with_thread_id() -> i32 {
    // expected-error @+1 {{Hardware ID and dimension ops in non-entry functions are not supported in horizontal fusion.}}
    %tx = nvvm.read.ptx.sreg.tid.x : i32
    llvm.return %tx : i32
  }

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {nvvm.kernel = 1 : ui1, xla.entry} {
    %0 = llvm.call @helper_with_thread_id() : () -> i32
    llvm.return
  }
}
