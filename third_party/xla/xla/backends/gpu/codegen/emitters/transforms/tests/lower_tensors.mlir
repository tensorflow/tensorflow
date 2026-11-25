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
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-PASCAL

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 7}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-VOLTA

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 8}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-AMPERE

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-HOPPER

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx908:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX908-MI100

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx90a:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX90A-MI200

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX942-MI300

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='oneapi_compute_capability {architecture: \"xe2hpg\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-BMG

// This example is a bit silly, in real life there wouldn't be a loop (the
// loop body would be executed by different threads). We're just doing it this
// way so control flow with shared memory is tested as well.
func.func @transpose_shared(%in: tensor<1024xf32>,
                            %out: tensor<1024xf32>) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  %shared = xla_gpu.allocate_shared : tensor<1024xf32>
  %loaded_tile = scf.for %i = %c0 to %c1024 step %c1
      iter_args(%tile = %shared) -> tensor<1024xf32> {
    %v = tensor.extract %in[%i] : tensor<1024xf32>
    %inserted = tensor.insert %v into %tile[%i] : tensor<1024xf32>
    scf.yield %inserted : tensor<1024xf32>
  }

  %synced = xla_gpu.sync_threads %shared : tensor<1024xf32>
  %written_tile = scf.for %i = %c0 to %c1024 step %c1
      iter_args(%written = %out) -> tensor<1024xf32> {
    %v = tensor.extract %shared[%i] : tensor<1024xf32>
    %inserted = tensor.insert %v into %written[%i] : tensor<1024xf32>
    scf.yield %inserted : tensor<1024xf32>
  }

  return %written_tile : tensor<1024xf32>
}
// CHECK:      llvm.mlir.global private @[[SHARED:shared_.*]]()
// CHECK-SAME:     {addr_space = 3 : i32} : !llvm.array<1024 x f32>
// CHECK:      @transpose_shared
// CHECK:        %[[ADDR:.*]] = llvm.mlir.addressof @[[SHARED]] : !llvm.ptr<3>
// CHECK:        %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]]
// CHECK-SAME:       : !llvm.ptr<3> to !llvm.ptr
// CHECK:        scf.for
// CHECK:          %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:          llvm.store {{.*}} %[[ELEM_ADDR]]
// CHECK:        gpu.barrier
// CHECK:        scf.for
// CHECK:          %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:          llvm.load %[[ELEM_ADDR]]

// -----

func.func @shared_complex() -> tensor<10xcomplex<f32>> {
  %shared = xla_gpu.allocate_shared : tensor<10xcomplex<f32>>
  return %shared : tensor<10xcomplex<f32>>
}
// CHECK: llvm.mlir.global private @{{.*}}() {addr_space = 3 : i32} : !llvm.array<10 x struct<(f32, f32)>>
// CHECK-LABEL: @shared_complex

// -----

func.func @shared_i4() -> tensor<10xi4> {
  %shared = xla_gpu.allocate_shared : tensor<10xi4>
  return %shared : tensor<10xi4>
}
// Five packed bytes are padded to two whole words.
// CHECK: llvm.mlir.global private @{{.*}}() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<8 x i8>
// CHECK-LABEL: @shared_i4

// -----

func.func @shared_i4_store(%v: i4, %i: index) -> tensor<10xi4> {
  %shared = xla_gpu.allocate_shared : tensor<10xi4>
  %r = tensor.insert %v into %shared[%i] : tensor<10xi4>
  return %r : tensor<10xi4>
}
// A 4 bit store is a 32 bit CAS on the containing word, so the shared
// allocation must be word aligned and a whole number of words.
// CHECK: llvm.mlir.global private @{{.*}}() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<8 x i8>
// CHECK-LABEL: @shared_i4_store
// CHECK: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK: %[[INIT:.*]] = llvm.load %[[BASE]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-HOPPER-LABEL: @shared_i4_store
// CHECK-HOPPER: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK-HOPPER: %[[INIT:.*]] = llvm.load %[[BASE]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-HOPPER-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK-HOPPER: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-GFX908-MI100-LABEL: @shared_i4_store
// CHECK-GFX908-MI100: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK-GFX908-MI100: %[[INIT:.*]] = llvm.load %[[BASE]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> i32
// CHECK-GFX908-MI100-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK-GFX908-MI100: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32
// CHECK-GFX942-MI300-LABEL: @shared_i4_store
// CHECK-GFX942-MI300: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK-GFX942-MI300: %[[INIT:.*]] = llvm.load %[[BASE]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> i32
// CHECK-GFX942-MI300-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK-GFX942-MI300: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32
// CHECK-BMG-LABEL: @shared_i4_store
// CHECK-BMG: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK-BMG: %[[INIT:.*]] = llvm.load %[[BASE]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-BMG-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK-BMG: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} monotonic monotonic : !llvm.ptr, i32

// -----

func.func @shared_atomic_store_and_fmax(%v16: f16, %v32: f32, %i: index)
    -> (tensor<4xf16>, tensor<4xf32>) {
  %sh16 = xla_gpu.allocate_shared : tensor<4xf16>
  %s16 = xla.atomic_store %v16 into %sh16[%i] : tensor<4xf16>
  %sh32 = xla_gpu.allocate_shared : tensor<4xf32>
  %s32 = xla.atomic_rmw %sh32[%i] : tensor<4xf32> {
    ^bb0(%cur: f32):
      %m = arith.maximumf %cur, %v32 : f32
      xla.yield %m : f32
  }
  return %s16, %s32 : tensor<4xf16>, tensor<4xf32>
}
// CHECK-LABEL: @shared_atomic_store_and_fmax
// CHECK: %[[BASE16:.*]] = llvm.intr.ptrmask
// CHECK: %[[INIT16:.*]] = llvm.load %[[BASE16]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-NEXT: scf.while (%[[VAR16:.*]] = %[[INIT16]]) : (i32) -> i32
// CHECK: llvm.cmpxchg %[[BASE16]], %[[VAR16]], %{{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK: %[[ADDR32:.*]] = llvm.getelementptr inbounds
// CHECK-NEXT: %[[CUR32:.*]] = llvm.load %[[ADDR32]] atomic unordered <alignment = 4> : !llvm.ptr -> f32
// CHECK: llvm.atomicrmw max %[[ADDR32]], %{{.*}} monotonic
// CHECK: llvm.atomicrmw umin %[[ADDR32]], %{{.*}} monotonic
// CHECK-HOPPER-LABEL: @shared_atomic_store_and_fmax
// CHECK-HOPPER: %[[BASE16:.*]] = llvm.intr.ptrmask
// CHECK-HOPPER: %[[INIT16:.*]] = llvm.load %[[BASE16]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-HOPPER: llvm.cmpxchg %[[BASE16]], %{{.*}}, %{{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-HOPPER: %[[ADDR32:.*]] = llvm.getelementptr inbounds
// CHECK-HOPPER-NEXT: llvm.load %[[ADDR32]] atomic unordered <alignment = 4> : !llvm.ptr -> f32
// CHECK-GFX908-MI100-LABEL: @shared_atomic_store_and_fmax
// CHECK-GFX908-MI100: %[[BASE16:.*]] = llvm.intr.ptrmask
// CHECK-GFX908-MI100: llvm.load %[[BASE16]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> i32
// CHECK-GFX908-MI100: llvm.cmpxchg %[[BASE16]], %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32
// CHECK-GFX908-MI100: %[[ADDR32:.*]] = llvm.getelementptr inbounds
// CHECK-GFX908-MI100-NEXT: llvm.load %[[ADDR32]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> f32
// CHECK-GFX942-MI300-LABEL: @shared_atomic_store_and_fmax
// CHECK-GFX942-MI300: %[[BASE16:.*]] = llvm.intr.ptrmask
// CHECK-GFX942-MI300: llvm.load %[[BASE16]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> i32
// CHECK-GFX942-MI300: llvm.cmpxchg %[[BASE16]], %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32
// CHECK-GFX942-MI300: %[[ADDR32:.*]] = llvm.getelementptr inbounds
// CHECK-GFX942-MI300-NEXT: llvm.load %[[ADDR32]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> f32
// CHECK-BMG-LABEL: @shared_atomic_store_and_fmax
// CHECK-BMG: %[[BASE16:.*]] = llvm.intr.ptrmask
// CHECK-BMG: llvm.load %[[BASE16]] atomic unordered <alignment = 4> : !llvm.ptr -> i32
// CHECK-BMG: llvm.cmpxchg %[[BASE16]], %{{.*}}, %{{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-BMG: %[[ADDR32:.*]] = llvm.getelementptr inbounds
// CHECK-BMG-NEXT: llvm.load %[[ADDR32]] atomic unordered <alignment = 4> : !llvm.ptr -> f32

// -----

func.func @shared_f64() -> tensor<16xf64> {
  %shared = xla_gpu.allocate_shared : tensor<16xf64>
  return %shared : tensor<16xf64>
}
// Wide element types keep their natural alignment.
// CHECK: llvm.mlir.global private @{{.*}}() {addr_space = 3 : i32} : !llvm.array<16 x f64>
// CHECK-LABEL: @shared_f64

// -----

func.func @shared_f16() -> tensor<3xf16> {
  %shared = xla_gpu.allocate_shared : tensor<3xf16>
  return %shared : tensor<3xf16>
}
// Six bytes of f16 are padded to eight.
// CHECK: llvm.mlir.global private @{{.*}}() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<4 x f16>
// CHECK-LABEL: @shared_f16

// -----

func.func @shared_reinterpreted_transfer_alignment(%v: vector<2xf64>)
    -> (tensor<2xf64>, vector<2xf64>, vector<8xi8>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f64
  %i0 = arith.constant 0 : i8
  %narrow = xla_gpu.allocate_shared : tensor<16xi8>
  %as_f64 = builtin.unrealized_conversion_cast %narrow : tensor<16xi8> to tensor<2xf64>
  %r0 = vector.transfer_read %as_f64[%c0], %f0 {in_bounds = [true]} : tensor<2xf64>, vector<2xf64>
  %w = vector.transfer_write %v, %as_f64[%c0] {in_bounds = [true]} : vector<2xf64>, tensor<2xf64>
  %wide = xla_gpu.allocate_shared : tensor<2xf64>
  %as_i8 = builtin.unrealized_conversion_cast %wide : tensor<2xf64> to tensor<16xi8>
  %r1 = vector.transfer_read %as_i8[%c0], %i0 {in_bounds = [true]} : tensor<16xi8>, vector<8xi8>
  return %w, %r0, %r1 : tensor<2xf64>, vector<2xf64>, vector<8xi8>
}
// Shared buffer alignment follows the allocation element type, not a cast view.
// CHECK-LABEL: @shared_reinterpreted_transfer_alignment
// CHECK:     llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> vector<2xf64>
// CHECK:     llvm.store %{{.*}}, %{{.*}} <alignment = 4> : vector<2xf64>, !llvm.ptr
// CHECK:     llvm.load %{{.*}} <alignment = 8> : !llvm.ptr -> vector<8xi8>

// -----

func.func @direct_atomic_rmw_overwrite(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      xla.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_overwrite
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.store %[[C2]], %[[ADDR]] atomic unordered <alignment = 4>

// -----

func.func @direct_atomic_rmw_addi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.addi %current, %c2 : i32
      xla.yield %min : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_addi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw add %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_maxsi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.maxsi %current, %c2 : i32
      xla.yield %min : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_maxsi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw max %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_maxui(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.maxui %current, %c2 : i32
      xla.yield %min : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_maxui
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw umax %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_minsi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.minsi %current, %c2 : i32
      xla.yield %min : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_minsi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw min %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_minui(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.minui %current, %c2 : i32
      xla.yield %min : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_minui
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw umin %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_fadd_f32(%in: tensor<8xf32>,
  %i: index) -> (tensor<8xf32>) {
  %c2 = arith.constant 2.0 : f32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %min = arith.addf %current, %c2 : f32
      xla.yield %min : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-VOLTA-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-VOLTA: %[[C2:.*]] = arith.constant 2
// CHECK-VOLTA: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-VOLTA: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-AMPERE: %[[C2:.*]] = arith.constant 2
// CHECK-AMPERE: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-AMPERE: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-GFX908-MI100-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-GFX908-MI100: %[[C2:.*]] = arith.constant 2
// CHECK-GFX908-MI100: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// CHECK-GFX90A-MI200-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-GFX90A-MI200: %[[C2:.*]] = arith.constant 2
// CHECK-GFX90A-MI200: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX90A-MI200: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// CHECK-BMG-LABEL: @direct_atomic_rmw_fadd_f32
// CHECK-BMG: %[[C2:.*]] = arith.constant 2
// CHECK-BMG: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-BMG: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// -----

func.func @direct_atomic_rmw_fsub_f32(%in: tensor<8xf32>,
  %i: index, %x: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sub = arith.subf %current, %x : f32
      xla.yield %sub : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-PASCAL: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[NEG]] monotonic

// CHECK-VOLTA-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-VOLTA: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-VOLTA: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-VOLTA: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[NEG]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-AMPERE: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-AMPERE: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-AMPERE: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[NEG]] monotonic

// CHECK-GFX908-MI100-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-GFX908-MI100: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX908-MI100: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[NEG]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// CHECK-GFX90A-MI200-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-GFX90A-MI200: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX90A-MI200: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-GFX90A-MI200: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[NEG]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// CHECK-BMG-LABEL: @direct_atomic_rmw_fsub_f32
// CHECK-BMG: %[[NEG:.*]] = arith.negf %{{.*}} : f32
// CHECK-BMG: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// -----

func.func @direct_atomic_rmw_fsub_swapped_f32(%in: tensor<8xf32>,
  %i: index, %x: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sub = arith.subf %x, %current : f32
      xla.yield %sub : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fsub_swapped_f32
// CHECK-PASCAL-NOT: llvm.atomicrmw fadd
// CHECK-PASCAL: llvm.cmpxchg

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_swapped_f32
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fsub_swapped_f32
// CHECK-HOPPER-NOT: llvm.atomicrmw fadd
// CHECK-HOPPER: llvm.cmpxchg

// -----

// Yielding %current folds away even when dead operations remain in the body.
func.func @no_direct_atomic_rmw_fsub_not_yielded(%in: tensor<8xf32>,
  %i: index, %x: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sub = arith.subf %current, %x : f32
      xla.yield %current : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fsub_not_yielded
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE-NOT: llvm.cmpxchg
// CHECK-AMPERE: return

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_not_yielded
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER-NOT: llvm.cmpxchg
// CHECK-HOPPER: return

// -----

// A dead addf alongside a yielded mulf must fall back to cmpxchg.
func.func @no_direct_atomic_rmw_fadd_not_yielded(%in: tensor<8xf32>,
  %i: index, %x: f32, %y: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sum = arith.addf %current, %x : f32
      %prod = arith.mulf %current, %y : f32
      xla.yield %prod : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fadd_not_yielded
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE: llvm.cmpxchg
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE: return

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fadd_not_yielded
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: llvm.cmpxchg
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: return

// -----

// A dead subf alongside a yielded mulf must fall back to cmpxchg.
func.func @no_direct_atomic_rmw_dead_fsub_with_mulf(%in: tensor<8xf32>,
  %i: index, %x: f32, %y: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sub = arith.subf %current, %x : f32
      %prod = arith.mulf %current, %y : f32
      xla.yield %prod : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_dead_fsub_with_mulf
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE: llvm.cmpxchg
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE: return

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_dead_fsub_with_mulf
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: llvm.cmpxchg
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: return

// -----

// Neither addf operand is the atomic body argument, so the sum is hoisted and
// the update becomes an atomic store.
func.func @no_direct_atomic_rmw_fadd_no_current(%in: tensor<8xf32>,
  %i: index, %x: f32, %y: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sum = arith.addf %x, %y : f32
      xla.yield %sum : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fadd_no_current
// CHECK-AMPERE: %[[SUM:.*]] = arith.addf
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE-NOT: llvm.cmpxchg
// CHECK-AMPERE: llvm.store %[[SUM]], %{{.*}} atomic unordered <alignment = 4>
// CHECK-AMPERE-NOT: llvm.atomicrmw
// CHECK-AMPERE-NOT: llvm.cmpxchg
// CHECK-AMPERE: return

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fadd_no_current
// CHECK-HOPPER: %[[SUM:.*]] = arith.addf
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER-NOT: llvm.cmpxchg
// CHECK-HOPPER: llvm.store %[[SUM]], %{{.*}} atomic unordered <alignment = 4>
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER-NOT: llvm.cmpxchg
// CHECK-HOPPER: return

// -----

func.func @no_direct_atomic_rmw_fsub_constant_body(%in: tensor<8xf32>,
  %i: index) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %c = arith.constant 1.0 : f32
      xla.yield %c : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fsub_constant_body
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd
// CHECK-AMPERE: llvm.store

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_constant_body
// CHECK-HOPPER-NOT: llvm.atomicrmw fadd
// CHECK-HOPPER: llvm.store

// CHECK-BMG-LABEL: @no_direct_atomic_rmw_fsub_constant_body
// CHECK-BMG-NOT: llvm.atomicrmw fadd
// CHECK-BMG: llvm.store

// -----

func.func @direct_atomic_rmw_fsub_f16(%in: tensor<8xf16>,
  %i: index, %x: f16) -> (tensor<8xf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf16> {
    ^bb0(%current : f16):
      %sub = arith.subf %current, %x : f16
      xla.yield %sub : f16
  }
  return %ret : tensor<8xf16>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-PASCAL-NOT: arith.negf
// CHECK-PASCAL: llvm.cmpxchg

// CHECK-VOLTA-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-VOLTA: %[[NEG:.*]] = arith.negf %{{.*}} : f16
// CHECK-VOLTA: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-AMPERE: %[[NEG:.*]] = arith.negf %{{.*}} : f16
// CHECK-AMPERE: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-HOPPER: %[[NEG:.*]] = arith.negf %{{.*}} : f16
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-BMG-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-BMG-NOT: arith.negf
// CHECK-BMG: llvm.cmpxchg

// CHECK-GFX942-MI300-LABEL: @direct_atomic_rmw_fsub_f16
// CHECK-GFX942-MI300: %[[NEG:.*]] = arith.negf %{{.*}} : f16
// CHECK-GFX942-MI300: %[[NEG_INT:.*]] = llvm.bitcast %[[NEG]] : f16 to i16
// CHECK-GFX942-MI300: %[[VAL:.*]] = llvm.bitcast %{{.*}} : i32 to vector<2xf16>
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[VAL]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

func.func @direct_atomic_rmw_fadd_f16(%in: tensor<8xf16>,
  %i: index) -> (tensor<8xf16>) {
  %c2 = arith.constant 2.0 : f16
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf16> {
    ^bb0(%current : f16):
      %min = arith.addf %current, %c2 : f16
      xla.yield %min : f16
  }
  return %ret : tensor<8xf16>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fadd_f16
// CHECK-PASCAL-NOT: llvm.atomicrmw fadd

// CHECK-VOLTA-LABEL: @direct_atomic_rmw_fadd_f16
// CHECK-VOLTA: %[[C2:.*]] = arith.constant 2
// CHECK-VOLTA: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-VOLTA: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fadd_f16
// CHECK-AMPERE: %[[C2:.*]] = arith.constant 2
// CHECK-AMPERE: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-AMPERE: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-GFX908-MI100-LABEL: @direct_atomic_rmw_fadd_f16
// CHECK-GFX908-MI100: %[[CST:.*]] = arith.constant 2
// CHECK-GFX908-MI100: %[[C_NEG4:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK-GFX908-MI100: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-GFX908-MI100: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GFX908-MI100: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX908-MI100: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GFX908-MI100: %[[ADDR_MASKED:.*]] = llvm.and %[[ADDR_INT]], %[[C_NEG4]]
// CHECK-GFX908-MI100: %[[ADDR_TRUNC:.*]] = llvm.trunc %[[ADDR_INT]]
// CHECK-GFX908-MI100: %[[OFFSET:.*]] = llvm.and %[[ADDR_TRUNC]], %[[C2]]
// CHECK-GFX908-MI100: %[[SHIFT:.*]] = llvm.mul %[[OFFSET]], %[[C8]]
// CHECK-GFX908-MI100: %[[VAL_INT:.*]] = llvm.bitcast %[[CST]] : f16 to i16
// CHECK-GFX908-MI100: %[[VAL_WIDE:.*]] = llvm.zext %[[VAL_INT]] : i16 to i32
// CHECK-GFX908-MI100: %[[VAL_SHIFT:.*]] = llvm.shl %[[VAL_WIDE]], %[[SHIFT]]
// CHECK-GFX908-MI100: %[[ADDR:.*]] = llvm.inttoptr %[[ADDR_MASKED]]
// CHECK-GFX908-MI100: %[[VAL:.*]] = llvm.bitcast %[[VAL_SHIFT]] : i32 to vector<2xf16>
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[VAL]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}


// CHECK-GFX90A-MI200-LABEL: @direct_atomic_rmw_fadd_f16
// CHECK-GFX90A-MI200: %[[CST:.*]] = arith.constant 2
// CHECK-GFX90A-MI200: %[[C_NEG4:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK-GFX90A-MI200: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-GFX90A-MI200: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GFX90A-MI200: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX90A-MI200: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GFX90A-MI200: %[[ADDR_MASKED:.*]] = llvm.and %[[ADDR_INT]], %[[C_NEG4]]
// CHECK-GFX90A-MI200: %[[ADDR_TRUNC:.*]] = llvm.trunc %[[ADDR_INT]]
// CHECK-GFX90A-MI200: %[[OFFSET:.*]] = llvm.and %[[ADDR_TRUNC]], %[[C2]]
// CHECK-GFX90A-MI200: %[[SHIFT:.*]] = llvm.mul %[[OFFSET]], %[[C8]]
// CHECK-GFX90A-MI200: %[[VAL_INT:.*]] = llvm.bitcast %[[CST]] : f16 to i16
// CHECK-GFX90A-MI200: %[[VAL_WIDE:.*]] = llvm.zext %[[VAL_INT]] : i16 to i32
// CHECK-GFX90A-MI200: %[[VAL_SHIFT:.*]] = llvm.shl %[[VAL_WIDE]], %[[SHIFT]]
// CHECK-GFX90A-MI200: %[[ADDR:.*]] = llvm.inttoptr %[[ADDR_MASKED]]
// CHECK-GFX90A-MI200: %[[VAL:.*]] = llvm.bitcast %[[VAL_SHIFT]] : i32 to vector<2xf16>
// CHECK-GFX90A-MI200: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[VAL]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

func.func @no_direct_atomic_rmw_fadd_f8E4M3(%in: tensor<8xf8E4M3>,
    %i: index) -> (tensor<8xf8E4M3>) {
  %c2 = arith.constant 2.0 : f8E4M3
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf8E4M3> {
    ^bb0(%current : f8E4M3):
      %min = arith.addf %current, %c2 : f8E4M3
      xla.yield %min : f8E4M3
  }
  return %ret : tensor<8xf8E4M3>
}
// CHECK-PASCAL-LABEL: @no_direct_atomic_rmw_fadd_f8E4M3
// CHECK-PASCAL-NOT: llvm.atomicrmw fadd

// CHECK-VOLTA-LABEL: @no_direct_atomic_rmw_fadd_f8E4M3
// CHECK-VOLTA-NOT: llvm.atomicrmw fadd

// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fadd_f8E4M3
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd

// CHECK-GFX908-MI100-LABEL: @no_direct_atomic_rmw_fadd_f8E4M3
// CHECK-GFX908-MI100-NOT: llvm.atomicrmw fadd
// CHECK-GFX908-MI100: return

// CHECK-GFX90A-MI200-LABEL: @no_direct_atomic_rmw_fadd_f8E4M3
// CHECK-GFX90A-MI200-NOT: llvm.atomicrmw fadd
// CHECK-GFX90A-MI200: return

// -----

func.func @no_direct_atomic_rmw_fsub_f8E4M3(%in: tensor<8xf8E4M3>,
    %i: index) -> (tensor<8xf8E4M3>) {
  %c2 = arith.constant 2.0 : f8E4M3
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf8E4M3> {
    ^bb0(%current : f8E4M3):
      %sub = arith.subf %current, %c2 : f8E4M3
      xla.yield %sub : f8E4M3
  }
  return %ret : tensor<8xf8E4M3>
}
// CHECK-PASCAL-LABEL: @no_direct_atomic_rmw_fsub_f8E4M3
// CHECK-PASCAL-NOT: arith.negf
// CHECK-PASCAL: return

// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fsub_f8E4M3
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: return

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_f8E4M3
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: return

// -----

func.func @direct_atomic_rmw_fadd_bf16(%in: tensor<8xbf16>,
    %i: index) -> (tensor<8xbf16>) {
  %c2 = arith.constant 2.0 : bf16
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %min = arith.addf %current, %c2 : bf16
      xla.yield %min : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fadd_bf16
// CHECK-PASCAL-NOT: llvm.atomicrmw fadd

// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fadd_bf16
// CHECK-HOPPER: %[[C2:.*]] = arith.constant 2
// CHECK-HOPPER: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// -----

// The scatter/reduce combiner for bf16 is computed in f32, so the atomic_rmw
// body is the widened pattern extf -> addf(f32) -> truncf. The modifier is a
// genuine bf16 value (a constant would have its extf folded to f32). On MI300
// this must lower to a packed `atomicrmw fadd <2 x bf16>` rather than a slow
// compare-and-swap loop.
func.func @direct_atomic_rmw_fadd_bf16_widened(%in: tensor<8xbf16>,
    %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %arg : bf16 to f32
      %add = arith.addf %ext0, %ext1 : f32
      %res = arith.truncf %add : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-GFX942-MI300-LABEL: @direct_atomic_rmw_fadd_bf16_widened
// CHECK-GFX942-MI300-DAG: %[[C_NEG4:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK-GFX942-MI300-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-GFX942-MI300-DAG: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GFX942-MI300: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX942-MI300: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GFX942-MI300: %[[ADDR_MASKED:.*]] = llvm.and %[[ADDR_INT]], %[[C_NEG4]]
// CHECK-GFX942-MI300: %[[ADDR_TRUNC:.*]] = llvm.trunc %[[ADDR_INT]]
// CHECK-GFX942-MI300: %[[OFFSET:.*]] = llvm.and %[[ADDR_TRUNC]], %[[C2]]
// CHECK-GFX942-MI300: %[[SHIFT:.*]] = llvm.mul %[[OFFSET]], %[[C8]]
// CHECK-GFX942-MI300: %[[VAL_INT:.*]] = llvm.bitcast %{{.*}} : bf16 to i16
// CHECK-GFX942-MI300: %[[VAL_WIDE:.*]] = llvm.zext %[[VAL_INT]] : i16 to i32
// CHECK-GFX942-MI300: %[[VAL_SHIFT:.*]] = llvm.shl %[[VAL_WIDE]], %[[SHIFT]]
// CHECK-GFX942-MI300: %[[ADDR_PTR:.*]] = llvm.inttoptr %[[ADDR_MASKED]]
// CHECK-GFX942-MI300: %[[VAL:.*]] = llvm.bitcast %[[VAL_SHIFT]] : i32 to vector<2xbf16>
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR_PTR]], %[[VAL]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

func.func @direct_atomic_rmw_fsub_bf16_widened(%in: tensor<8xbf16>,
    %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %arg : bf16 to f32
      %sub = arith.subf %ext0, %ext1 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fsub_bf16_widened
// CHECK-HOPPER: %[[NEG:.*]] = arith.negf %{{.*}} : bf16
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_bf16_widened
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-GFX942-MI300-LABEL: @direct_atomic_rmw_fsub_bf16_widened
// CHECK-GFX942-MI300: %[[NEG:.*]] = arith.negf %{{.*}} : bf16
// CHECK-GFX942-MI300: %[[NEG_INT:.*]] = llvm.bitcast %[[NEG]] : bf16 to i16
// CHECK-GFX942-MI300: %[[VAL:.*]] = llvm.bitcast %{{.*}} : i32 to vector<2xbf16>
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[VAL]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

// The subtracted value does not depend on %current, so it hoists out of the
// atomic body and lowers to a direct atomic fadd on targets with bf16 atomics.
func.func @direct_atomic_rmw_fsub_bf16_widened_in_body(%in: tensor<8xbf16>,
    %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %arg : bf16 to f32
      %square = arith.mulf %ext1, %ext1 : f32
      %square_bf16 = arith.truncf %square : f32 to bf16
      %square_f32 = arith.extf %square_bf16 : bf16 to f32
      %sub = arith.subf %ext0, %square_f32 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-HOPPER: %[[SQ_BF16:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK-HOPPER: %[[NEG:.*]] = arith.negf %[[SQ_BF16]] : bf16
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: %[[SQ_BF16:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: %[[SQ_F32:.*]] = arith.extf %[[SQ_BF16]] : bf16 to f32
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: scf.while
// CHECK-AMPERE: arith.subf %{{.*}}, %[[SQ_F32]] : f32
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-GFX942-MI300-LABEL: @direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-GFX942-MI300: %[[SQ_BF16:.*]] = arith.truncf %{{.*}} : f32 to bf16
// CHECK-GFX942-MI300: %[[NEG:.*]] = arith.negf %[[SQ_BF16]] : bf16
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

func.func @direct_atomic_rmw_fsub_bf16_widened_nested_in_body(
    %in: tensor<8xbf16>, %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %cond = arith.constant true
      %modifier = scf.if %cond -> (bf16) {
        scf.yield %arg : bf16
      } else {
        scf.yield %arg : bf16
      }
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %modifier : bf16 to f32
      %sub = arith.subf %ext0, %ext1 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fsub_bf16_widened_nested_in_body
// CHECK-HOPPER: %[[NEG:.*]] = arith.negf %{{.*}} : bf16
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[NEG]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fsub_bf16_widened_nested_in_body
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: llvm.cmpxchg

// -----

// When the subtracted value genuinely depends on %current, it cannot be hoisted
// and must fall back to cmpxchg.
func.func @no_direct_atomic_rmw_fsub_bf16_widened_dependent_in_body(
    %in: tensor<8xbf16>, %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %arg : bf16 to f32
      %prod = arith.mulf %ext0, %ext1 : f32
      %prod_bf16 = arith.truncf %prod : f32 to bf16
      %prod_f32 = arith.extf %prod_bf16 : bf16 to f32
      %sub = arith.subf %ext0, %prod_f32 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_dependent_in_body
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: llvm.cmpxchg

// CHECK-GFX942-MI300-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_dependent_in_body
// CHECK-GFX942-MI300-NOT: arith.negf
// CHECK-GFX942-MI300: llvm.cmpxchg

// -----

// A widened bf16 addition with an f32 constant that is exact in bf16 recovers
// the narrow bf16 constant and emits a direct atomic fadd.
func.func @direct_atomic_rmw_fadd_bf16_widened_exact_cst(
    %in: tensor<8xbf16>, %i: index) -> (tensor<8xbf16>) {
  %c2 = arith.constant 2.0 : f32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %add = arith.addf %ext0, %c2 : f32
      %res = arith.truncf %add : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @direct_atomic_rmw_fadd_bf16_widened_exact_cst
// CHECK-HOPPER: %[[C2_BF16:.*]] = arith.constant 2.000000e+00 : bf16
// CHECK-HOPPER: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %[[C2_BF16]] monotonic

// CHECK-GFX942-MI300-LABEL: @direct_atomic_rmw_fadd_bf16_widened_exact_cst
// CHECK-GFX942-MI300: %[[C2_BF16:.*]] = arith.constant 2.000000e+00 : bf16
// CHECK-GFX942-MI300: llvm.bitcast %[[C2_BF16]] : bf16 to i16
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// -----

// An f32 constant that is not exact in bf16 must not be narrowed to bf16.
func.func @no_direct_atomic_rmw_fadd_bf16_widened_inexact_cst(
    %in: tensor<8xbf16>, %i: index) -> (tensor<8xbf16>) {
  %c_inexact = arith.constant 0.1 : f32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %add = arith.addf %ext0, %c_inexact : f32
      %res = arith.truncf %add : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fadd_bf16_widened_inexact_cst
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: llvm.cmpxchg
// CHECK-HOPPER-NOT: llvm.atomicrmw
// CHECK-HOPPER: return

// -----

func.func @aligned_vector_atomic_rmw_amd(
    %in: tensor<8xbf16> {llvm.align = 16 : index}, %vec: vector<2xbf16>)
    -> tensor<8xbf16> {
  %c2 = arith.constant 2 : index
  %r = xla.atomic_rmw %in[%c2] : tensor<8xbf16> {
  ^bb0(%cur: vector<2xbf16>):
    %add = arith.addf %cur, %vec : vector<2xbf16>
    xla.yield %add : vector<2xbf16>
  }
  return %r : tensor<8xbf16>
}
// CHECK-GFX942-MI300-LABEL: @aligned_vector_atomic_rmw_amd
// CHECK-GFX942-MI300: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory} : !llvm.ptr, vector<2xbf16>
// CHECK-GFX942-MI300-NOT: llvm.atomicrmw
// CHECK-GFX942-MI300: return

// CHECK-GFX908-MI100-LABEL: @aligned_vector_atomic_rmw_amd
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory} : !llvm.ptr, bf16
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %{{.*}}, %{{.*}} syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory} : !llvm.ptr, bf16

// -----

func.func @shared_bf16_atomic_rmw_amd(%out: tensor<8xbf16>, %i: index, %m: bf16)
    -> tensor<8xbf16> {
  %sh = xla_gpu.allocate_shared : tensor<8xbf16>
  %a = xla.atomic_rmw %sh[%i] : tensor<8xbf16> {
  ^bb0(%cur: bf16):
    %ec = arith.extf %cur : bf16 to f32
    %em = arith.extf %m : bf16 to f32
    %r = arith.addf %ec, %em : f32
    %tr = arith.truncf %r : f32 to bf16
    xla.yield %tr : bf16
  }
  %s = xla_gpu.sync_threads %a : tensor<8xbf16>
  %v = tensor.extract %s[%i] : tensor<8xbf16>
  %o = tensor.insert %v into %out[%i] : tensor<8xbf16>
  return %o : tensor<8xbf16>
}
// CHECK-GFX942-MI300-LABEL: @shared_bf16_atomic_rmw_amd
// CHECK-GFX942-MI300-NOT: !llvm.ptr<1>
// CHECK-GFX942-MI300-NOT: llvm.atomicrmw
// CHECK-GFX942-MI300: %[[BASE:.*]] = llvm.intr.ptrmask
// CHECK-GFX942-MI300: %[[INIT:.*]] = llvm.load %[[BASE]] atomic syncscope("agent-one-as") monotonic <alignment = 4> : !llvm.ptr -> i32
// CHECK-GFX942-MI300-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i32) -> i32
// CHECK-GFX942-MI300: llvm.cmpxchg %[[BASE]], %[[VAR]], %{{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32
// CHECK-GFX942-MI300-NOT: !llvm.ptr<1>
// CHECK-GFX942-MI300-NOT: llvm.atomicrmw
// CHECK-GFX942-MI300: return

// -----

func.func @no_direct_atomic_rmw_fsub_bf16_widened_swapped(%in: tensor<8xbf16>,
    %i: index, %arg: bf16) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %ext1 = arith.extf %arg : bf16 to f32
      %sub = arith.subf %ext1, %ext0 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_swapped
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: llvm.cmpxchg

// -----

func.func @no_direct_atomic_rmw_fsub_bf16_widened_same(%in: tensor<8xbf16>,
    %i: index) -> (tensor<8xbf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xbf16> {
    ^bb0(%current : bf16):
      %ext0 = arith.extf %current : bf16 to f32
      %sub = arith.subf %ext0, %ext0 : f32
      %res = arith.truncf %sub : f32 to bf16
      xla.yield %res : bf16
  }
  return %ret : tensor<8xbf16>
}
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_same
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: llvm.cmpxchg

// -----

func.func @direct_atomic_rmw_fadd_f64(%in: tensor<8xf64>,
  %i: index) -> (tensor<8xf64>) {
  %c2 = arith.constant 2.0 : f64
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf64> {
    ^bb0(%current : f64):
      %min = arith.addf %current, %c2 : f64
      xla.yield %min : f64
  }
  return %ret : tensor<8xf64>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-VOLTA-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-VOLTA: %[[C2:.*]] = arith.constant 2
// CHECK-VOLTA: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-VOLTA: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-AMPERE-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-AMPERE: %[[C2:.*]] = arith.constant 2
// CHECK-AMPERE: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-AMPERE: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// CHECK-GFX908-MI100-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-GFX908-MI100: %[[C2:.*]] = arith.constant 2
// CHECK-GFX908-MI100: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX908-MI100: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}


// CHECK-GFX90A-MI200-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-GFX90A-MI200: %[[C2:.*]] = arith.constant 2
// CHECK-GFX90A-MI200: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GFX90A-MI200: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] syncscope("agent-one-as") monotonic {rocdl.no_fine_grained_memory, rocdl.no_remote_memory}

// CHECK-BMG-LABEL: @direct_atomic_rmw_fadd_f64
// CHECK-BMG: %[[C2:.*]] = arith.constant 2
// CHECK-BMG: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-BMG: llvm.atomicrmw ignore_denormal_mode fadd %[[ADDR]], %[[C2]] monotonic

// -----

#i4_map = #xla.indexing_map<"(d0) -> (d0 * 8), domain: d0 in [0, 7]">
func.func @transfer_alignment_subbyte_shared_and_global(
    %arg0: tensor<64xi4> {llvm.align = 16 : index}, %tid: index) -> tensor<64xi4> {
  %c0_i4 = arith.constant 0 : i4
  %shmem = xla_gpu.allocate_shared : tensor<64xi4>
  %idx = xla.apply_indexing #i4_map(%tid)
  %v = vector.transfer_read %arg0[%idx], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>
  %shmem_w = vector.transfer_write %v, %shmem[%idx] {in_bounds = [true]} : vector<8xi4>, tensor<64xi4>
  %synced = xla_gpu.sync_threads %shmem_w : tensor<64xi4>
  %v2 = vector.transfer_read %synced[%idx], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>
  %out = vector.transfer_write %v2, %arg0[%idx] {in_bounds = [true]} : vector<8xi4>, tensor<64xi4>
  return %out : tensor<64xi4>
}
// CHECK-LABEL: @transfer_alignment_subbyte_shared_and_global(
// CHECK:           %[[LD0:.*]] = llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> i32
// CHECK:       llvm.store %[[LD0]], %{{.*}} <alignment = 4> : i32, !llvm.ptr
// CHECK:       gpu.barrier
// CHECK:       %[[LD1:.*]] = llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> i32
// CHECK:           llvm.store %[[LD1]], %{{.*}} <alignment = 4> : i32, !llvm.ptr

// -----

func.func @vector_atomic_rmw(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 4.200000e+01 : f32
  %cst_42 = arith.constant dense<42.000000e+00> : vector<4xf32>
  %atomic_rmw = xla.atomic_rmw %arg0[%c0] : tensor<4xf32> {
  ^bb0(%arg1: vector<4xf32>):
    %1 = arith.addf %arg1, %cst_42 : vector<4xf32>
    xla.yield %1 : vector<4xf32>
  }
  return %atomic_rmw : tensor<4xf32>
}

// CHECK-HOPPER-LABEL:  @vector_atomic_rmw
// CHECK-HOPPER-SAME:     %[[ADDR:.*]]: !llvm.ptr
// CHECK-HOPPER:          llvm.inline_asm
// CHECK-HOPPER-SAME:     atom.global.v4.f32.add {$0, $1, $2, $3}, [$4], {$5, $6, $7, $8}

// -----

func.func @vector_atomic_rmw_sub(%arg0: tensor<4xf32>,
    %x: vector<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %atomic_rmw = xla.atomic_rmw %arg0[%c0] : tensor<4xf32> {
  ^bb0(%arg1: vector<4xf32>):
    %sub = arith.subf %arg1, %x : vector<4xf32>
    xla.yield %sub : vector<4xf32>
  }
  return %atomic_rmw : tensor<4xf32>
}

// CHECK-HOPPER-LABEL:  @vector_atomic_rmw_sub
// CHECK-HOPPER:          %[[NEG:.*]] = arith.negf %{{.*}} : vector<4xf32>
// CHECK-HOPPER:          llvm.inline_asm
// CHECK-HOPPER-SAME:     atom.global.v4.f32.add {$0, $1, $2, $3}, [$4], {$5, $6, $7, $8}

// -----

func.func @non_entry_block_arg_no_align_spoof_or_oob(
    %aligned_dummy: tensor<8xf32> {llvm.align = 16 : index},
    %unaligned: tensor<8xbf16>, %v: vector<2xbf16>) -> bf16 {
  %c0 = arith.constant 0 : index
  "test.br"(%unaligned, %c0, %c0, %unaligned) [^bb1]
      : (tensor<8xbf16>, index, index, tensor<8xbf16>) -> ()
^bb1(%bb_first: tensor<8xbf16>, %i0: index, %i1: index,
     %bb_oob: tensor<8xbf16>):
  %r0 = xla.atomic_rmw %bb_first[%c0] : tensor<8xbf16> {
  ^bb0(%cur: vector<2xbf16>):
    %a = arith.addf %cur, %v : vector<2xbf16>
    xla.yield %a : vector<2xbf16>
  }
  %r1 = xla.atomic_rmw %bb_oob[%i0] : tensor<8xbf16> {
  ^bb0(%cur: vector<2xbf16>):
    %a = arith.addf %cur, %v : vector<2xbf16>
    xla.yield %a : vector<2xbf16>
  }
  %e0 = tensor.extract %r0[%c0] : tensor<8xbf16>
  %e1 = tensor.extract %r1[%i1] : tensor<8xbf16>
  %sum = arith.addf %e0, %e1 : bf16
  return %sum : bf16
}
// CHECK-GFX942-MI300-LABEL: @non_entry_block_arg_no_align_spoof_or_oob
// CHECK-GFX942-MI300-COUNT-4: llvm.atomicrmw ignore_denormal_mode fadd {{.*}} : !llvm.ptr<1>, vector<2xbf16>
// CHECK-GFX942-MI300-NOT: : !llvm.ptr, vector<2xbf16>
// CHECK-GFX942-MI300: return
