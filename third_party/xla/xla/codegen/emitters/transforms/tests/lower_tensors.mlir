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
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-GPU

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-CPU

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-PASCAL

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 7}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-VOLTA

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 8}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-AMPERE

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-HOPPER

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx908:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX908-MI100

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx90a:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX90A-MI200

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-GFX942-MI300

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="gpu_device_info='oneapi_compute_capability {architecture: \"xe2hpg\"}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-BMG

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func private @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    func.return %sum : f32
  }

  func.func private @tensorarg(%arg0: tensor<43xf32> {xla.invariant, xla.slice_index = 0}, %arg1: index) -> f32 {
    %v1 = arith.constant 2.0 : f32
    %v2 = tensor.extract %arg0[%arg1] : tensor<43xf32>
    %sum = func.call @add(%v1, %v2) : (f32, f32) -> f32
    func.return %sum : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32> {xla.slice_index = 0}, %arg1: index) -> f32 {
    %call = func.call @tensorarg(%arg0, %arg1) : (tensor<43xf32>, index) -> f32
    func.return %call : f32
  }

  func.func @stores(%arg0: tensor<17xf32> {xla.slice_index = 0}, %arg1: tensor<43xf32> {xla.slice_index = 1}) -> tensor<43xf32> {
    %c17 = arith.constant 17 : index
    %c23 = arith.constant 23 : index
    %cst = arith.constant 3.0 : f32
    %out = tensor.insert %cst into %arg1[%c17] : tensor<43xf32>
    %out2 = tensor.insert %cst into %out[%c23] : tensor<43xf32>
    func.return %out2 : tensor<43xf32>
  }
}

// CHECK:        func.func private @add(%{{.*}}: f32, %{{.*}}: f32) -> f32 {
// CHECK-NEXT:     arith.addf
// CHECK-NEXT:     return

// The kernel only reads the buffer, but other kernels may write it, so the
// argument is readonly and its loads are not invariant.
// CHECK:        func.func private @tensorarg(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {llvm.readonly, xla.invariant, xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index) -> f32 {
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2.000000e+00
// CHECK-DAG:       %[[IDX:.*]] = arith.index_castui %[[ARG1]] : index to i32
// CHECK-DAG:       %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX]]]
// CHECK-DAG:       %[[V2:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> f32
// CHECK:           %[[RET:.*]] = call @add(%[[C2]], %[[V2]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @tensorcall(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {xla.slice_index = 0 : i64}, %[[ARG1:.*]]: index)
// CHECK:           %[[RET:.*]] = call @tensorarg(%[[ARG0]], %[[ARG1]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @stores(
// CHECK-SAME:        %[[ARG0:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:        %[[ARG1:.*]]: !llvm.ptr {xla.slice_index = 1 : i64})
// CHECK-NEXT:      %[[CST:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-NEXT:      %[[PTR1:.*]] = llvm.getelementptr inbounds %[[ARG1]][0, 17]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[ARG1]][0, 23]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]
// CHECK-NEXT:      return

// -----

func.func @store_control_flow( %arg0: tensor<2xf32>, %arg1: index)
    -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32

  %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
    %new_out = tensor.insert %cst into %arg2[%i] : tensor<2xf32>
    scf.yield %new_out : tensor<2xf32>
  }

  %inbounds = arith.cmpi sle, %arg1, %c1 : index
  %result = scf.if %inbounds -> tensor<2xf32> {
    %if = tensor.insert %cst2 into %for[%arg1] : tensor<2xf32>
    scf.yield %if : tensor<2xf32>
  } else {
    scf.yield %for : tensor<2xf32>
  }
  func.return %result : tensor<2xf32>
}
// CHECK-LABEL:     @store_control_flow(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr, %[[X:.*]]: index) {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:         %[[CAST:.*]] = arith.index_castui %[[I]] : index to i64
// CHECK:         %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[CAST]]]
// CHECK:         llvm.store {{.*}}, %[[PTR]]
// CHECK:       %[[INBOUNDS:.*]] = arith.cmpi
// CHECK:       scf.if %[[INBOUNDS]] {
// CHECK:         llvm.store
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

func.func @large_tensor(%arg0: tensor<8000000000xf32>, %arg1: index) -> f32 {
  %v = tensor.extract %arg0[%arg1] : tensor<8000000000xf32>
  func.return %v : f32
}
// CHECK-LABEL: @large_tensor
// CHECK: arith.index_castui {{.*}} : index to i64

// -----

func.func @extract_from_constant(%arg0: tensor<2xf32>, %arg1: index) -> f32 {
  %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %extracted = tensor.extract %arg0[%arg1] : tensor<2xf32>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<2xf32>
  %0 = arith.addf %extracted, %extracted_0 : f32
  return %0 : f32
}
// CHECK: llvm.mlir.global private constant @global_cst_0(dense<
// CHECK-SAME: [1.000000e+00, 2.000000e+00]> : tensor<2xf32>) {addr_space = 0 : i32} : !llvm.array<2 x f32>
// CHECK-LABEL: @extract_from_constant
// CHECK: %[[ADDR_OF:.*]] = llvm.mlir.addressof @global_cst_0 : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ADDR_OF]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x f32>
// CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> f32
// CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[LOAD]] : f32
// CHECK: return %[[ADD]] : f32

// -----

func.func @vector_constant() -> vector<2xindex> {
  %c1 = arith.constant dense<[1, 2]> : vector<2xindex>
  func.return %c1 : vector<2xindex>
}
// vector constants should not be rewritten.
// CHECK-LABEL: @vector_constant
// CHECK-NEXT: arith.constant

// -----

func.func @complex_tensor_insert(
    %arg0: tensor<10xcomplex<f32>>) -> tensor<10xcomplex<f32>> {
  %c1 = arith.constant 1 : index
  %real = arith.constant 3.0 : f32
  %imag = arith.constant 2.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %out = tensor.insert %complex into %arg0[%c1] : tensor<10xcomplex<f32>>
  func.return %out : tensor<10xcomplex<f32>>
}
// CHECK-LABEL: @complex_tensor_insert(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK: %[[C:.*]] = complex.create
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x struct<(f32, f32)>>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[C]] : complex<f32> to !llvm.struct<(f32, f32)>
// CHECK: llvm.store %[[CAST]], %[[GEP]] : !llvm.struct<(f32, f32)>, !llvm.ptr

// -----

func.func @complex_tensor_extract(
    %arg0: tensor<10xcomplex<f32>>) -> complex<f32> {
  %c1 = arith.constant 1 : index
  %v2 = tensor.extract %arg0[%c1] : tensor<10xcomplex<f32>>
  func.return %v2 : complex<f32>
}
// CHECK-LABEL: @complex_tensor_extract(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<10 x struct<(f32, f32)>>
// CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> !llvm.struct<(f32, f32)>
// CHECK: builtin.unrealized_conversion_cast %[[LOAD]] : !llvm.struct<(f32, f32)> to complex<f32>

// -----

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

func.func @atomic_rmw_f32(%in: tensor<8xf32>, %i: index) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 1.0 : f32
      %add = arith.minimumf %current, %c42 : f32
      xla.yield %add : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-LABEL: @atomic_rmw_f32
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[INIT:.*]] = llvm.load %[[ADDR]]
// CHECK-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK: %[[RES:.*]] = arith.bitcast %{{.*}} : f32 to i32
// CHECK-NEXT: llvm.cmpxchg %[[ADDR]], %[[VAR]], %[[RES]]

// -----

func.func @atomic_rmw_f16(%in: tensor<8xf16>, %i: index)
    -> (tensor<8xf16>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf16> {
    ^bb0(%current : f16):
      %c1 = arith.constant 1.0 : f16
      %add = arith.addf %current, %c1 : f16
      xla.yield %add : f16
  }
  return %ret : tensor<8xf16>
}
// CHECK-LABEL: @atomic_rmw_f16
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GPU-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GPU-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-GPU-NEXT: %[[BASE:.*]] = llvm.intr.ptrmask %[[ADDR]], %{{.*}}
// CHECK-GPU: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-GPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-GPU-NEXT: %[[VAR_SHIFT:.*]] = llvm.lshr %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[VAR_TRUNC:.*]] = llvm.trunc %[[VAR_SHIFT]]
// CHECK-GPU-NEXT: arith.bitcast %[[VAR_TRUNC]] : i16 to f16
// CHECK-GPU: %[[RES:.*]] = arith.bitcast %{{.*}} : f16 to i16
// CHECK-GPU-NEXT: %[[RES_WIDE:.*]] = llvm.zext %[[RES]]
// CHECK-GPU-NEXT: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[RES_SHIFT:.*]] = llvm.shl %[[RES_WIDE]], %{{.*}}
// CHECK-GPU-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-GPU-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]
// CHECK-CPU-NEXT: %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i16
// CHECK-CPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i16) -> i16
// CHECK-CPU-NEXT: %[[CUR:.*]] = arith.bitcast %[[VAR]] : i16 to f16
// CHECK-CPU-NEXT: %[[ADD:.*]] = arith.addf %[[CUR]], %{{.*}} : f16
// CHECK-CPU-NEXT: %[[RES:.*]] = arith.bitcast %[[ADD]] : f16 to i16
// CHECK-CPU-NEXT: llvm.cmpxchg %[[ADDR]], %[[VAR]], %[[RES]] monotonic monotonic : !llvm.ptr, i16

// -----

func.func @atomic_rmw_f8E4M3(%in: tensor<8xf8E4M3>, %i: index)
    -> (tensor<8xf8E4M3>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf8E4M3> {
    ^bb0(%current : f8E4M3):
      %c1 = arith.constant 1.0 : f8E4M3
      %add = arith.addf %current, %c1 : f8E4M3
      xla.yield %add : f8E4M3
  }
  return %ret : tensor<8xf8E4M3>
}
// CHECK-LABEL: @atomic_rmw_f8E4M3
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GPU-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GPU-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-GPU-NEXT: %[[BASE:.*]] = llvm.intr.ptrmask %[[ADDR]], %{{.*}}
// CHECK-GPU: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-GPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-GPU-NEXT: %[[VAR_SHIFT:.*]] = llvm.lshr %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[VAR_TRUNC:.*]] = llvm.trunc %[[VAR_SHIFT]]
// CHECK-GPU-NEXT: arith.bitcast %[[VAR_TRUNC]] : i8 to f8E4M3
// CHECK-GPU: %[[RES:.*]] = arith.bitcast %{{.*}} : f8E4M3 to i8
// CHECK-GPU-NEXT: %[[RES_WIDE:.*]] = llvm.zext %[[RES]]
// CHECK-GPU-DAG: %[[RES_SHIFT:.*]] = llvm.shl %[[RES_WIDE]], %{{.*}}
// CHECK-GPU-DAG: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-GPU-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]
// CHECK-CPU-NEXT: %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i8
// CHECK-CPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i8) -> i8
// CHECK-CPU-NEXT: %[[CUR:.*]] = arith.bitcast %[[VAR]] : i8 to f8E4M3
// CHECK-CPU-NEXT: %[[ADD:.*]] = arith.addf %[[CUR]], %{{.*}} : f8E4M3
// CHECK-CPU-NEXT: %[[RES:.*]] = arith.bitcast %[[ADD]] : f8E4M3 to i8
// CHECK-CPU-NEXT: llvm.cmpxchg %[[ADDR]], %[[VAR]], %[[RES]] monotonic monotonic : !llvm.ptr, i8

// -----

func.func @atomic_rmw_i4(%in: tensor<8xi4>, %i: index) -> (tensor<8xi4>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xi4> {
    ^bb0(%current : i4):
      %c1 = arith.constant 1 : i4
      %add = arith.addi %current, %c1 : i4
      xla.yield %add : i4
  }
  return %ret : tensor<8xi4>
}

// CHECK-LABEL: @atomic_rmw_i4(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: index) {
// CHECK-GPU:       %[[C1_I4:.*]] = arith.constant 1 : i4
// CHECK-GPU:       %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-GPU:       %[[C4_I8:.*]] = arith.constant 4 : i8
// CHECK-GPU:       %[[C3_I64:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-GPU:       %[[C_NEG4_I64:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK-GPU:       %[[C8_I32:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GPU:       %[[C_NEG1_I32:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-GPU:       %[[C15_I32:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK-GPU:       %[[C_TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-GPU:       %[[ARG1_I64:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK-GPU:       %[[AND_1:.*]] = arith.andi %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK-GPU:       %[[TRUNC_I8:.*]] = arith.trunci %[[AND_1]] : i64 to i8
// CHECK-GPU:       %[[MUL_4:.*]] = arith.muli %[[TRUNC_I8]], %[[C4_I8]] : i8
// CHECK-GPU:       %[[SHR_1:.*]] = arith.shrui %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK-GPU:       %[[GEP_BASE:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[SHR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK-GPU:       %[[PTR_INT:.*]] = llvm.ptrtoint %[[GEP_BASE]] : !llvm.ptr to i64
// CHECK-GPU:       %[[AND_3:.*]] = llvm.and %[[PTR_INT]], %[[C3_I64]] : i64
// CHECK-GPU:       %[[GEP_OFFSET:.*]] = llvm.intr.ptrmask %[[GEP_BASE]], %[[C_NEG4_I64]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-GPU:       %[[TRUNC_I32:.*]] = llvm.trunc %[[AND_3]] : i64 to i32
// CHECK-GPU:       %[[MUL_8:.*]] = llvm.mul %[[TRUNC_I32]], %[[C8_I32]] : i32
// CHECK-GPU:       %[[ZEXT_I32:.*]] = llvm.zext %[[MUL_4]] : i8 to i32
// CHECK-GPU:       %[[ADD_19:.*]] = llvm.add %[[MUL_8]], %[[ZEXT_I32]] : i32
// CHECK-GPU:       %[[SHL_20:.*]] = llvm.shl %[[C15_I32]], %[[ADD_19]] : i32
// CHECK-GPU:       %[[XOR_21:.*]] = llvm.xor %[[C_NEG1_I32]], %[[SHL_20]] : i32
// CHECK-GPU:       %[[LOADED_I32:.*]] = llvm.load %[[GEP_OFFSET]] : !llvm.ptr -> i32
// CHECK-GPU:       %[[WHILE_RES:.*]] = scf.while (%[[ARG2:.*]] = %[[LOADED_I32]]) : (i32) -> i32 {
// CHECK-GPU:         %[[LSHR:.*]] = llvm.lshr %[[ARG2]], %[[ADD_19]] : i32
// CHECK-GPU:         %[[TRUNC_I4:.*]] = llvm.trunc %[[LSHR]] : i32 to i4
// CHECK-GPU:         %[[ADDI:.*]] = arith.addi %[[TRUNC_I4]], %[[C1_I4]] : i4
// CHECK-GPU:         %[[ZEXT_I32_2:.*]] = llvm.zext %[[ADDI]] : i4 to i32
// CHECK-GPU:         %[[AND_MASK_I32:.*]] = llvm.and %[[ARG2]], %[[XOR_21]] : i32
// CHECK-GPU:         %[[SHL_NEW:.*]] = llvm.shl %[[ZEXT_I32_2]], %[[ADD_19]] : i32
// CHECK-GPU:         %[[OR_NEW:.*]] = llvm.or %[[AND_MASK_I32]], %[[SHL_NEW]] : i32
// CHECK-GPU:         %[[CMPXCHG:.*]] = llvm.cmpxchg %[[GEP_OFFSET]], %[[ARG2]], %[[OR_NEW]] monotonic monotonic : !llvm.ptr, i32
// CHECK-GPU:         %[[EXTRACT0:.*]] = llvm.extractvalue %[[CMPXCHG]][0] : !llvm.struct<(i32, i1)>
// CHECK-GPU:         %[[EXTRACT1:.*]] = llvm.extractvalue %[[CMPXCHG]][1] : !llvm.struct<(i32, i1)>
// CHECK-GPU:         %[[XOR_34:.*]] = llvm.xor %[[EXTRACT1]], %[[C_TRUE]] : i1
// CHECK-GPU:         scf.condition
// CHECK-CPU:       %[[C1_I4:.*]] = arith.constant 1 : i4
// CHECK-CPU:       %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-CPU:       %[[C4_I8:.*]] = arith.constant 4 : i8
// CHECK-CPU:       %[[C_NEG1_I8:.*]] = llvm.mlir.constant(-1 : i8) : i8
// CHECK-CPU:       %[[C15_I8:.*]] = llvm.mlir.constant(15 : i8) : i8
// CHECK-CPU:       %[[C_TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-CPU:       %[[ARG1_I64:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK-CPU:       %[[AND_1:.*]] = arith.andi %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK-CPU:       %[[TRUNC_I8:.*]] = arith.trunci %[[AND_1]] : i64 to i8
// CHECK-CPU:       %[[MUL_4:.*]] = arith.muli %[[TRUNC_I8]], %[[C4_I8]] : i8
// CHECK-CPU:       %[[SHR_1:.*]] = arith.shrui %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK-CPU:       %[[GEP_BASE:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[SHR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK-CPU-NEXT:  %[[SHL_MASK:.*]] = llvm.shl %[[C15_I8]], %[[MUL_4]] : i8
// CHECK-CPU-NEXT:  %[[INV_MASK:.*]] = llvm.xor %[[C_NEG1_I8]], %[[SHL_MASK]] : i8
// CHECK-CPU-NEXT:  %[[LOADED_I8:.*]] = llvm.load %[[GEP_BASE]] : !llvm.ptr -> i8
// CHECK-CPU-NEXT:  %[[WHILE_RES:.*]] = scf.while (%[[ARG2:.*]] = %[[LOADED_I8]]) : (i8) -> i8 {
// CHECK-CPU-NEXT:    %[[LSHR:.*]] = llvm.lshr %[[ARG2]], %[[MUL_4]] : i8
// CHECK-CPU-NEXT:    %[[TRUNC_I4:.*]] = llvm.trunc %[[LSHR]] : i8 to i4
// CHECK-CPU-NEXT:    %[[ADDI:.*]] = arith.addi %[[TRUNC_I4]], %[[C1_I4]] : i4
// CHECK-CPU-NEXT:    %[[ZEXT_I8:.*]] = llvm.zext %[[ADDI]] : i4 to i8
// CHECK-CPU-NEXT:    %[[AND_MASK_I8:.*]] = llvm.and %[[ARG2]], %[[INV_MASK]] : i8
// CHECK-CPU-NEXT:    %[[SHL_NEW:.*]] = llvm.shl %[[ZEXT_I8]], %[[MUL_4]] : i8
// CHECK-CPU-NEXT:    %[[OR_NEW:.*]] = llvm.or %[[AND_MASK_I8]], %[[SHL_NEW]] : i8
// CHECK-CPU-NEXT:    %[[CMPXCHG:.*]] = llvm.cmpxchg %[[GEP_BASE]], %[[ARG2]], %[[OR_NEW]] monotonic monotonic : !llvm.ptr, i8
// CHECK-CPU-NEXT:    %[[EXTRACT0:.*]] = llvm.extractvalue %[[CMPXCHG]][0] : !llvm.struct<(i8, i1)>
// CHECK-CPU-NEXT:    %[[EXTRACT1:.*]] = llvm.extractvalue %[[CMPXCHG]][1] : !llvm.struct<(i8, i1)>
// CHECK-CPU-NEXT:    %[[XOR_OK:.*]] = llvm.xor %[[EXTRACT1]], %[[C_TRUE]] : i1
// CHECK-CPU-NEXT:    scf.condition

// -----

func.func @atomic_rmw_overwrite(%in: tensor<8xf16>, %i: index)
    -> (tensor<8xf16>) {
  %c1 = arith.constant 1.0 : f16
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf16> {
    ^bb0(%current : f16):
      xla.yield %c1 : f16
  }
  return %ret : tensor<8xf16>
}
// CHECK-LABEL: @atomic_rmw_overwrite
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GPU-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GPU-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-GPU-NEXT: %[[BASE:.*]] = llvm.intr.ptrmask %[[ADDR]], %{{.*}}
// CHECK-GPU: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-GPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-GPU-DAG: %[[RES_SHIFT:.*]] = llvm.shl %{{.*}}, %{{.*}}
// CHECK-GPU-DAG: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-GPU-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]
// CHECK-CPU-NEXT: %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i16
// CHECK-CPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i16) -> i16
// CHECK-CPU-NEXT: llvm.cmpxchg %[[ADDR]], %[[VAR]], %{{.*}} monotonic monotonic : !llvm.ptr, i16

// -----

func.func @atomic_rmw_overwrite_f8E4M3(%in: tensor<8xf8E4M3>, %i: index)
    -> (tensor<8xf8E4M3>) {
  %c1 = arith.constant 1.0 : f8E4M3
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf8E4M3> {
    ^bb0(%current : f8E4M3):
      xla.yield %c1 : f8E4M3
  }
  return %ret : tensor<8xf8E4M3>
}
// CHECK-LABEL: @atomic_rmw_overwrite_f8E4M3
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-GPU-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-GPU-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-GPU-NEXT: %[[BASE:.*]] = llvm.intr.ptrmask %[[ADDR]], %{{.*}}
// CHECK-GPU: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-GPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-GPU-DAG: %[[RES_SHIFT:.*]] = llvm.shl %{{.*}}, %{{.*}}
// CHECK-GPU-DAG: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-GPU-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-GPU-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]
// CHECK-CPU-NEXT: %[[INIT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> i8
// CHECK-CPU-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]]) : (i8) -> i8
// CHECK-CPU-NEXT: llvm.cmpxchg %[[ADDR]], %[[VAR]], %{{.*}} monotonic monotonic : !llvm.ptr, i8

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
// CHECK-GPU: llvm.cmpxchg {{.*}} : !llvm.ptr, i32

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

func.func @i4_load(%arg: tensor<10xi4>, %i: index)
    -> i4 {
  %v = tensor.extract %arg[%i] : tensor<10xi4>
  return %v : i4
}

// CHECK-LABEL: @i4_load(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: index) -> i4 {
// CHECK-DAG:     %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG:     %[[C4_I8:.*]] = arith.constant 4 : i8
// CHECK:         %[[ARG1_I64:.*]] = arith.index_castui %[[ARG1]] : index to i64
// CHECK:         %[[AND_1:.*]] = arith.andi %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK:         %[[TRUNC_I8:.*]] = arith.trunci %[[AND_1]] : i64 to i8
// CHECK:         %[[MUL_4:.*]] = arith.muli %[[TRUNC_I8]], %[[C4_I8]] : i8
// CHECK:         %[[SHR_1:.*]] = arith.shrui %[[ARG1_I64]], %[[C1_I64]] : i64
// CHECK:         %[[GEP_BASE:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[SHR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<5 x i8>
// CHECK:         %[[LOADED_I8:.*]] = llvm.load %[[GEP_BASE]] : !llvm.ptr -> i8
// CHECK:         %[[SHR_VAL:.*]] = arith.shrui %[[LOADED_I8]], %[[MUL_4]] : i8
// CHECK:         %[[TRUNC_I4:.*]] = arith.trunci %[[SHR_VAL]] : i8 to i4
// CHECK:         return %[[TRUNC_I4]] : i4

// -----

func.func @i4_store(%arg: tensor<10xi4>, %v: i4, %i: index)
    -> tensor<10xi4> {
  %r = tensor.insert %v into %arg[%i] : tensor<10xi4>
  return %r : tensor<10xi4>
}

// CHECK-LABEL: @i4_store(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: i4, %[[ARG2:.*]]: index) {
// CHECK-DAG:     %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG:     %[[C4_I8:.*]] = arith.constant 4 : i8
// CHECK-DAG:     %[[C15_I8:.*]] = arith.constant 15 : i8
// CHECK-DAG:     %[[C_NEG1_I8:.*]] = arith.constant -1 : i8
// CHECK-GPU-DAG: %[[C3_I64:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-GPU-DAG: %[[C_NEG4_I64:.*]] = llvm.mlir.constant(-4 : i64) : i64
// CHECK-GPU-DAG: %[[C8_I32:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GPU-DAG: %[[C_NEG1_I32:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-GPU-DAG: %[[C255_I32:.*]] = llvm.mlir.constant(255 : i32) : i32
// CHECK-GPU-DAG: %[[C_TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-GPU:     %[[CAST_ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !llvm.ptr to tensor<10xi4>
// CHECK:         %[[ARG2_I64:.*]] = arith.index_castui %[[ARG2]] : index to i64
// CHECK:         %[[AND_1:.*]] = arith.andi %[[ARG2_I64]], %[[C1_I64]] : i64
// CHECK:         %[[TRUNC_I8:.*]] = arith.trunci %[[AND_1]] : i64 to i8
// CHECK:         %[[MUL_4:.*]] = arith.muli %[[TRUNC_I8]], %[[C4_I8]] : i8
// CHECK:         %[[SHR_1:.*]] = arith.shrui %[[ARG2_I64]], %[[C1_I64]] : i64
// CHECK:         %[[EXT_ARG1:.*]] = arith.extui %[[ARG1]] : i4 to i8
// CHECK:         %[[SHL_15:.*]] = arith.shli %[[C15_I8]], %[[MUL_4]] : i8
// CHECK:         %[[XOR_NEG1_I8:.*]] = arith.xori %[[SHL_15]], %[[C_NEG1_I8]] : i8
// CHECK:         %[[SHL_EXT:.*]] = arith.shli %[[EXT_ARG1]], %[[MUL_4]] : i8
// CHECK-GPU:     %[[CAST_TENSOR:.*]] = builtin.unrealized_conversion_cast %[[CAST_ARG0]] : tensor<10xi4> to tensor<10xi8>
// CHECK-GPU:     %[[CAST_TENSOR_PTR:.*]] = builtin.unrealized_conversion_cast %[[CAST_TENSOR]] : tensor<10xi8> to !llvm.ptr
// CHECK-GPU:     %[[GEP_BASE:.*]] = llvm.getelementptr inbounds %[[CAST_TENSOR_PTR]][0, %[[SHR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<10 x i8>
// CHECK-GPU:     %[[PTR_INT:.*]] = llvm.ptrtoint %[[GEP_BASE]] : !llvm.ptr to i64
// CHECK-GPU:     %[[AND_3:.*]] = llvm.and %[[PTR_INT]], %[[C3_I64]] : i64
// CHECK-GPU:     %[[GEP_OFFSET:.*]] = llvm.intr.ptrmask %[[GEP_BASE]], %[[C_NEG4_I64]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-GPU:     %[[TRUNC_I32:.*]] = llvm.trunc %[[AND_3]] : i64 to i32
// CHECK-GPU:     %[[MUL_8:.*]] = llvm.mul %[[TRUNC_I32]], %[[C8_I32]] : i32
// CHECK-GPU:     %[[SHL_255:.*]] = llvm.shl %[[C255_I32]], %[[MUL_8]] : i32
// CHECK-GPU:     %[[XOR_NEG1_I32:.*]] = llvm.xor %[[C_NEG1_I32]], %[[SHL_255]] : i32
// CHECK-GPU:     %[[LOADED_I32:.*]] = llvm.load %[[GEP_OFFSET]] : !llvm.ptr -> i32
// CHECK-GPU:     %[[WHILE_RES:.*]] = scf.while (%[[INIT:.*]] = %[[LOADED_I32]]) : (i32) -> i32 {
// CHECK-GPU:       %[[LSHR:.*]] = llvm.lshr %[[INIT]], %[[MUL_8]] : i32
// CHECK-GPU:       %[[TRUNC_I8_2:.*]] = llvm.trunc %[[LSHR]] : i32 to i8
// CHECK-GPU:       %[[AND_MASK_I8:.*]] = arith.andi %[[TRUNC_I8_2]], %[[XOR_NEG1_I8]] : i8
// CHECK-GPU:       %[[ORI_NEW_I8:.*]] = arith.ori %[[AND_MASK_I8]], %[[SHL_EXT]] : i8
// CHECK-GPU:       %[[ZEXT_I32:.*]] = llvm.zext %[[ORI_NEW_I8]] : i8 to i32
// CHECK-GPU:       %[[AND_MASK_I32:.*]] = llvm.and %[[INIT]], %[[XOR_NEG1_I32]] : i32
// CHECK-GPU:       %[[SHL_NEW:.*]] = llvm.shl %[[ZEXT_I32]], %[[MUL_8]] : i32
// CHECK-GPU:       %[[OR_NEW:.*]] = llvm.or %[[AND_MASK_I32]], %[[SHL_NEW]] : i32
// CHECK-GPU:       llvm.cmpxchg %[[GEP_OFFSET]], %[[INIT]], %[[OR_NEW]] monotonic monotonic : !llvm.ptr, i32
// CHECK-GPU:       scf.condition
// CHECK-CPU-NOT: llvm.cmpxchg
// CHECK-CPU:     %[[GEP_BASE:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[SHR_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<5 x i8>
// CHECK-CPU-NEXT: %[[LOADED_I8:.*]] = llvm.load %[[GEP_BASE]] : !llvm.ptr -> i8
// CHECK-CPU-NEXT: %[[AND_MASK_I8:.*]] = arith.andi %[[LOADED_I8]], %[[XOR_NEG1_I8]] : i8
// CHECK-CPU-NEXT: %[[ORI_NEW_I8:.*]] = arith.ori %[[AND_MASK_I8]], %[[SHL_EXT]] : i8
// CHECK-CPU-NEXT: llvm.store %[[ORI_NEW_I8]], %[[GEP_BASE]] : i8, !llvm.ptr
// CHECK-CPU-NEXT: return


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

// The subf result is not the value yielded by the atomic body. This test
// relies on lower-tensors running before any DCE, so the dead subf stays.
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
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_not_yielded
// CHECK-HOPPER-NOT: llvm.atomicrmw fadd
// CHECK-HOPPER: llvm.cmpxchg

// -----

// The addf result is not the value yielded by the atomic body.
func.func @no_direct_atomic_rmw_fadd_not_yielded(%in: tensor<8xf32>,
  %i: index, %x: f32) -> (tensor<8xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sum = arith.addf %current, %x : f32
      xla.yield %current : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fadd_not_yielded
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fadd_not_yielded
// CHECK-HOPPER-NOT: llvm.atomicrmw fadd
// CHECK-HOPPER: llvm.cmpxchg

// -----

// Neither addf operand is the atomic body argument.
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
// CHECK-AMPERE-NOT: llvm.atomicrmw fadd
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fadd_no_current
// CHECK-HOPPER-NOT: llvm.atomicrmw fadd
// CHECK-HOPPER: llvm.cmpxchg

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

// The subtracted value is computed inside the atomic body. It cannot be used
// as the modifier of a direct atomic after the body is erased.
func.func @no_direct_atomic_rmw_fsub_bf16_widened_in_body(%in: tensor<8xbf16>,
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
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: llvm.cmpxchg

// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: llvm.cmpxchg

// CHECK-GFX942-MI300-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_in_body
// CHECK-GFX942-MI300-NOT: arith.negf
// CHECK-GFX942-MI300: llvm.cmpxchg

// -----

func.func @no_direct_atomic_rmw_fsub_bf16_widened_nested_in_body(
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
// CHECK-HOPPER-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_nested_in_body
// CHECK-HOPPER-NOT: arith.negf
// CHECK-HOPPER: llvm.cmpxchg

// CHECK-AMPERE-LABEL: @no_direct_atomic_rmw_fsub_bf16_widened_nested_in_body
// CHECK-AMPERE-NOT: arith.negf
// CHECK-AMPERE: llvm.cmpxchg

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

func.func @direct_atomic_rmw_maximumf(%in: tensor<8xf32>,
  %i: index) -> (tensor<8xf32>) {
  %c2 = arith.constant 2.0 : f32
  %ret = xla.atomic_rmw %in[%i] : tensor<8xf32> {
    ^bb0(%current : f32):
      %min = arith.maximumf %current, %c2 : f32
      xla.yield %min : f32
  }
  return %ret : tensor<8xf32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_maximumf

// CHECK-PASCAL: %[[MODIFIER:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-PASCAL: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
// CHECK-PASCAL: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: %[[CURRENT:.*]] = llvm.load %[[ADDR]] : !llvm.ptr -> f32
// CHECK-PASCAL: %[[CURRENT_IS_NAN:.*]] = llvm.fcmp "uno" %[[CURRENT]], %[[CURRENT]] : f32
// CHECK-PASCAL: scf.if %[[CURRENT_IS_NAN]] {
// CHECK-PASCAL: } else {
// CHECK-PASCAL:   %[[MODIFIER_IS_NAN:.*]] = llvm.fcmp "uno" %[[MODIFIER]], %[[MODIFIER]] : f32
// CHECK-PASCAL:   %[[MODIFIER_OR_NAN:.*]] = llvm.select %[[MODIFIER_IS_NAN]], %[[NAN]], %[[MODIFIER]] : i1, f32
// CHECK-PASCAL:   %[[VAL_13:.*]] = llvm.fcmp "ult" %[[CURRENT]], %[[MODIFIER_OR_NAN]] : f32
// CHECK-PASCAL:   scf.if %[[VAL_13]] {
// CHECK-PASCAL:     %[[INT_MODIFIER_OR_NAN:.*]] = arith.bitcast %[[MODIFIER_OR_NAN]] : f32 to i32
// CHECK-PASCAL:     %[[IS_POSITIVE:.*]] = llvm.icmp "sge" %[[INT_MODIFIER_OR_NAN]], %[[C0]] : i32
// CHECK-PASCAL:     scf.if %[[IS_POSITIVE]] {
// CHECK-PASCAL:       llvm.atomicrmw max %[[ADDR]], %[[INT_MODIFIER_OR_NAN]] monotonic
// CHECK-PASCAL:     } else {
// CHECK-PASCAL:       llvm.atomicrmw umin %[[ADDR]], %[[INT_MODIFIER_OR_NAN]] monotonic
// CHECK-PASCAL:     }
// CHECK-PASCAL:   }
// CHECK-PASCAL: }
// CHECK-PASCAL: return

// -----

func.func @atomic_rmw_c32(%in: tensor<8xcomplex<f32>>, %i: index)
    -> (tensor<8xcomplex<f32>>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<8xcomplex<f32>> {
    ^bb0(%current : complex<f32>):
      %a = complex.add %current, %current : complex<f32>
      xla.yield %a : complex<f32>
  }
  return %ret : tensor<8xcomplex<f32>>
}
// CHECK-LABEL: @atomic_rmw_c32

// CHECK: %[[TMP:.*]] = llvm.alloca %{{.*}} x i64 : (i32) -> !llvm.ptr
// CHECK: scf.while (%[[ITER_ARG:.*]] = %{{.*}}) : (i64) -> i64
// CHECK: llvm.store %[[ITER_ARG]], %[[TMP]]
// CHECK: %[[LD:.*]] = llvm.load %[[TMP]] : {{.*}} -> !llvm.struct<(f32, f32)>
// CHECK: builtin.unrealized_conversion_cast %[[LD]] : {{.*}} to complex<f32>

// -----

func.func @unused_index_switch_results(%i: index) -> index {
  %ret, %ret2 = scf.index_switch %i -> tensor<8xi32>, tensor<3xf32>
  case 0 {
    %x, %y = "dummy.op1"() : () -> (tensor<8xi32>, tensor<3xf32>)
    scf.yield %x, %y : tensor<8xi32>, tensor<3xf32>
  }
  default {
    %x, %y = "dummy.op2"() : () -> (tensor<8xi32>, tensor<3xf32>)
    scf.yield %x, %y : tensor<8xi32>, tensor<3xf32>
  }
  return %i : index
}
// CHECK-LABEL: func.func @unused_index_switch_results
// CHECK-SAME:      (%[[I:.*]]: index)
// CHECK-NEXT:    scf.index_switch %[[I]]
// CHECK-NEXT:    case 0 {
// CHECK-NEXT:      "dummy.op1"
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      "dummy.op2"
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[I]] : index

// -----

func.func @transfer_write(%arg0: tensor<43xf32> {xla.slice_index = 1}) -> tensor<43xf32> {
  %c16 = arith.constant 16 : index
  %c22 = arith.constant 22 : index
  %cst = arith.constant dense<[1.0, 2.0]> : vector<2xf32>
  %out = vector.transfer_write %cst, %arg0[%c16] : vector<2xf32>, tensor<43xf32>
  %out2 = vector.transfer_write %cst, %out[%c22] : vector<2xf32>, tensor<43xf32>
  func.return %out2 : tensor<43xf32>
}
// CHECK-LABEL: @transfer_write
// CHECK:           %[[PTR1:.*]] = llvm.getelementptr inbounds %[[BUF:.*]][0, 16]
// CHECK-NEXT:      llvm.store %[[CST:.*]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[BUF]][0, 22]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]

// -----

func.func @transfer_read(%arg0: tensor<43xf32> {xla.slice_index = 1}) -> vector<2xf32> {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0.0 : f32
  %out = vector.transfer_read %arg0[%c16], %c0 : tensor<43xf32>, vector<2xf32>
  func.return %out : vector<2xf32>
}
// CHECK-LABEL: @transfer_read
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[0, 16]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 4> : !llvm.ptr -> vector<2xf32>

// -----

func.func @transfer_read_alignment(%arg0: tensor<8xi64> {llvm.align = 32 : index}) -> vector<8xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %arg0[%c0], %c0_i64 {in_bounds = [true]} : tensor<8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}
// CHECK-LABEL: @transfer_read_alignment(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 0] :
// CHECK-SAME:        !llvm.array<8 x i64>
// CHECK:           %[[LOADED:.*]] = llvm.load %[[GEP]] <alignment = 32> : !llvm.ptr
// CHECK:           return %[[LOADED]] : vector<8xi64>

// -----

func.func @transfer_read_alignment_non_zero_index(%arg0: tensor<16xi64> {llvm.align = 32 : index}) -> vector<8xi64> {
  %c8 = arith.constant 8 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %arg0[%c8], %c0_i64 {in_bounds = [true]} : tensor<16xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}
// CHECK-LABEL: @transfer_read_alignment_non_zero_index(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 8]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 32> : !llvm.ptr -> vector<8xi64>

// -----

func.func @transfer_read_alignment_partial_offset(%arg0: tensor<16xi64> {llvm.align = 32 : index}) -> vector<2xi64> {
  %c2 = arith.constant 2 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %arg0[%c2], %c0_i64 {in_bounds = [true]} : tensor<16xi64>, vector<2xi64>
  return %0 : vector<2xi64>
}
// CHECK-LABEL: @transfer_read_alignment_partial_offset(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 2]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr -> vector<2xi64>

// -----

#align_map = #xla.indexing_map<"(d0) -> (d0 * 4 + 8), domain: d0 in [0, 3]">
func.func @transfer_read_alignment_apply_indexing(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = xla.apply_indexing #align_map(%idx)
  %1 = vector.transfer_read %arg0[%0], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_apply_indexing(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[IDX:.*]] = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]] : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr -> vector<4xf32>

// -----

#align_multi_result_map = #xla.indexing_map<"(d0) -> (d0 + 1, d0 * 4 + 8), domain: d0 in [0, 3]">
func.func @transfer_read_alignment_apply_indexing_multi_result(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index, %other: tensor<32xf32>) -> (f32, vector<4xf32>) {
  %cst = arith.constant 0.0 : f32
  %0:2 = xla.apply_indexing #align_multi_result_map(%idx)
  %1 = tensor.extract %other[%0#0] : tensor<32xf32>
  %2 = vector.transfer_read %arg0[%0#1], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %1, %2 : f32, vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_apply_indexing_multi_result(
// CHECK-SAME:  %[[ARG0:[^:]*]]: !llvm.ptr
// CHECK:           %[[IDX:.*]]:2 = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]]#1 : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr -> vector<4xf32>

// -----

#align_max_map = #xla.indexing_map<"(d0) -> (max(d0 * 4, 8)), domain: d0 in [0, 3]">
func.func @transfer_read_alignment_apply_indexing_max(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = xla.apply_indexing #align_max_map(%idx)
  %1 = vector.transfer_read %arg0[%0], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_apply_indexing_max(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[IDX:.*]] = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]] : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr -> vector<4xf32>

// -----

#align_mod_min_map = #xla.indexing_map<"(d0) -> (min((d0 * 8) mod 24, 12)), domain: d0 in [0, 7]">
func.func @transfer_read_alignment_apply_indexing_mod_min(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = xla.apply_indexing #align_mod_min_map(%idx)
  %1 = vector.transfer_read %arg0[%0], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_apply_indexing_mod_min(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[IDX:.*]] = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]] : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 16> : !llvm.ptr -> vector<4xf32>

// -----

#align_div_map = #xla.indexing_map<"(d0) -> (d0 floordiv 3), domain: d0 in [0, 11]">
func.func @transfer_read_alignment_apply_indexing_floordiv(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = xla.apply_indexing #align_div_map(%idx)
  %1 = vector.transfer_read %arg0[%0], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_apply_indexing_floordiv(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[IDX:.*]] = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]] : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 4> : !llvm.ptr -> vector<4xf32>

// -----

func.func @transfer_read_alignment_block_arg_index(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%idx], %cst {in_bounds = [true]} : tensor<32xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: @transfer_read_alignment_block_arg_index(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %{{.*}} : index to i64
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.load %[[PTR]] <alignment = 4> : !llvm.ptr -> vector<4xf32>

// -----

#write_align_map = #xla.indexing_map<"(d0) -> (d0 * 4 + 4), domain: d0 in [0, 3]">
func.func @transfer_write_alignment_chained_apply_indexing(%arg0: tensor<32xf32> {llvm.align = 16 : index}, %idx: index) -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %v = arith.constant dense<0.0> : vector<4xf32>
  %0 = vector.transfer_write %v, %arg0[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<32xf32>
  %1 = xla.apply_indexing #write_align_map(%idx)
  %2 = vector.transfer_write %v, %0[%1] {in_bounds = [true]} : vector<4xf32>, tensor<32xf32>
  return %2 : tensor<32xf32>
}
// CHECK-LABEL: @transfer_write_alignment_chained_apply_indexing(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK-DAG:       %[[V:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK:           llvm.store %[[V]], %{{.*}} <alignment = 16> : vector<4xf32>, !llvm.ptr
// CHECK:           %[[IDX:.*]] = xla.apply_indexing
// CHECK:           %[[IDX_I64:.*]] = arith.index_castui %[[IDX]] : index to i64
// CHECK:           %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %[[IDX_I64]]]
// CHECK-NEXT:      llvm.store %[[V]], %[[GEP]] <alignment = 16> : vector<4xf32>, !llvm.ptr

// -----

func.func @transfer_write_alignment(%arg0: tensor<8xi64> {llvm.align = 32 : index}) -> tensor<8xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant dense<0> : vector<8xi64>
  %0 = vector.transfer_write %c0_i64, %arg0[%c0] {in_bounds = [true]} : vector<8xi64>, tensor<8xi64>
  return %0 : tensor<8xi64>
}
// CHECK-LABEL: @transfer_write_alignment(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK-DAG:       %[[C0_I64:.*]] = arith.constant dense<0> : vector<8xi64>
// CHECK:           %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 0] :
// CHECK-SAME:        !llvm.array<8 x i64>
// CHECK:           llvm.store %[[C0_I64]], %[[GEP]] <alignment = 32> : vector<8xi64>, !llvm.ptr

// -----

func.func @transfer_write_alignment_non_zero_index(%arg0: tensor<8xi64> {llvm.align = 32 : index}) -> tensor<8xi64> {
  %c8 = arith.constant 8 : index
  %c0_i64 = arith.constant dense<0> : vector<8xi64>
  %0 = vector.transfer_write %c0_i64, %arg0[%c8] {in_bounds = [true]} : vector<8xi64>, tensor<8xi64>
  return %0 : tensor<8xi64>
}
// CHECK-LABEL: @transfer_write_alignment_non_zero_index(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr
// CHECK-DAG:       %[[C0_I64:.*]] = arith.constant dense<0> : vector<8xi64>
// CHECK:           %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, 8] :
// CHECK-SAME:        !llvm.array<8 x i64>
// CHECK:           llvm.store %[[C0_I64]], %[[GEP]] <alignment = 32> : vector<8xi64>, !llvm.ptr

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
// CHECK:           llvm.store %[[LD0]], %{{.*}} <alignment = 4> : i32, !llvm.ptr
// CHECK:           gpu.barrier
// CHECK:           %[[LD1:.*]] = llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> i32
// CHECK:           llvm.store %[[LD1]], %{{.*}} <alignment = 4> : i32, !llvm.ptr

// -----

func.func @transfer_read_non_entry_block_arg(
    %arg0: tensor<8xi64> {llvm.align = 32 : index},
    %cond: i1, %p0: !llvm.ptr, %p1: !llvm.ptr) -> vector<8xi64> {
  llvm.cond_br %cond,
      ^bb2(%p0, %p1, %p0, %p1, %p0 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr),
      ^bb1
^bb1:
  "dummy.side_effect"() : () -> ()
  llvm.br ^bb2(%p1, %p0, %p1, %p0, %p1 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
^bb2(%bb0: !llvm.ptr, %bb1: !llvm.ptr, %bb2: !llvm.ptr, %bb3: !llvm.ptr, %bb4: !llvm.ptr):
  "dummy.use"(%bb0, %bb1, %bb2, %bb3, %bb4)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %t0 = builtin.unrealized_conversion_cast %bb0 : !llvm.ptr to tensor<8xi64>
  %t4 = builtin.unrealized_conversion_cast %bb4 : !llvm.ptr to tensor<8xi64>
  %v0 = vector.transfer_read %t0[%c0], %c0_i64 {in_bounds = [true]} : tensor<8xi64>, vector<8xi64>
  %v4 = vector.transfer_read %t4[%c0], %c0_i64 {in_bounds = [true]} : tensor<8xi64>, vector<8xi64>
  %sum = arith.addi %v0, %v4 : vector<8xi64>
  return %sum : vector<8xi64>
}
// CHECK-LABEL: @transfer_read_non_entry_block_arg(
// CHECK:         ^bb2(%[[BB0:.*]]: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[BB4:.*]]: !llvm.ptr):
// CHECK:           %[[GEP0:.*]] = llvm.getelementptr inbounds %[[BB0]][0, 0]
// CHECK-NEXT:      llvm.load %[[GEP0]] <alignment = 8> : !llvm.ptr -> vector<8xi64>
// CHECK:           %[[GEP4:.*]] = llvm.getelementptr inbounds %[[BB4]][0, 0]
// CHECK-NEXT:      llvm.load %[[GEP4]] <alignment = 8> : !llvm.ptr -> vector<8xi64>

// -----

func.func @transfer_read_scf_if_mismatched_alignment(
    %cond: i1,
    %arg0: tensor<16xi64> {llvm.align = 32 : index},
    %arg1: tensor<16xi64> {llvm.align = 16 : index}) -> vector<2xi64> {
  %t = scf.if %cond -> tensor<16xi64> {
    scf.yield %arg0 : tensor<16xi64>
  } else {
    scf.yield %arg1 : tensor<16xi64>
  }
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = vector.transfer_read %t[%c0], %c0_i64 {in_bounds = [true]} : tensor<16xi64>, vector<2xi64>
  return %0 : vector<2xi64>
}
// CHECK-LABEL: @transfer_read_scf_if_mismatched_alignment(
// CHECK:           llvm.load %{{.*}} <alignment = 16> : !llvm.ptr -> vector<2xi64>

// -----

func.func @transfer_alignment_scf_for(
    %arg0: tensor<64xi4> {llvm.align = 16 : index},
    %arg1: tensor<64xi4> {llvm.align = 16 : index}) -> tensor<64xi4> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0_i4 = arith.constant 0 : i4
  %out = scf.for %iv = %c0 to %c64 step %c8 iter_args(%acc = %arg1) -> (tensor<64xi4>) {
    %v = vector.transfer_read %arg0[%iv], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>
    %w = vector.transfer_write %v, %acc[%iv] {in_bounds = [true]} : vector<8xi4>, tensor<64xi4>
    scf.yield %w : tensor<64xi4>
  }
  return %out : tensor<64xi4>
}
// CHECK-LABEL: @transfer_alignment_scf_for(
// CHECK:           %[[LD:.*]] = llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> i32
// CHECK:           llvm.store %[[LD]], %{{.*}} <alignment = 4> : i32, !llvm.ptr

// -----

#div_mod_map = #xla.indexing_map<"(d0) -> ((d0 floordiv 2) + (d0 mod 16)), domain: d0 in [0, 63]">
func.func @transfer_alignment_div_mod_shl(
    %arg0: tensor<64xi4> {llvm.align = 16 : index}, %tid: index) -> vector<8xi4> {
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c0_i4 = arith.constant 0 : i4
  %shifted = arith.shli %tid, %c3 : index
  %scaled = arith.muli %shifted, %c2 : index
  %idx = xla.apply_indexing #div_mod_map(%scaled)
  %v = vector.transfer_read %arg0[%idx], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>
  return %v : vector<8xi4>
}
// CHECK-LABEL: @transfer_alignment_div_mod_shl(
// CHECK:           llvm.load %{{.*}} <alignment = 4> : !llvm.ptr -> i32

// -----

module {
  llvm.mlir.global private @raw_shmem() {addr_space = 3 : i32} : !llvm.array<32 x i8>
  llvm.mlir.global private @aligned_shmem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<32 x i8>

  func.func @transfer_alignment_global_ops() -> (vector<8xi4>, vector<8xi4>, vector<2xi64>) {
    %c0 = arith.constant 0 : index
    %c0_i4 = arith.constant 0 : i4
    %c0_i64 = arith.constant 0 : i64
    %raw_ptr3 = llvm.mlir.addressof @raw_shmem : !llvm.ptr<3>
    %raw_ptr = llvm.addrspacecast %raw_ptr3 : !llvm.ptr<3> to !llvm.ptr
    %raw_t = builtin.unrealized_conversion_cast %raw_ptr : !llvm.ptr to tensor<64xi4>
    %v_raw = vector.transfer_read %raw_t[%c0], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>

    %al_ptr3 = llvm.mlir.addressof @aligned_shmem : !llvm.ptr<3>
    %al_ptr = llvm.addrspacecast %al_ptr3 : !llvm.ptr<3> to !llvm.ptr
    %al_t = builtin.unrealized_conversion_cast %al_ptr : !llvm.ptr to tensor<64xi4>
    %v_al = vector.transfer_read %al_t[%c0], %c0_i4 {in_bounds = [true]} : tensor<64xi4>, vector<8xi4>

    %cst = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi64>
    %v_cst = vector.transfer_read %cst[%c0], %c0_i64 {in_bounds = [true]} : tensor<8xi64>, vector<2xi64>
    return %v_raw, %v_al, %v_cst : vector<8xi4>, vector<8xi4>, vector<2xi64>
  }
}
// CHECK-LABEL: @transfer_alignment_global_ops(
// CHECK:           llvm.load %{{.*}} <alignment = 1> : !llvm.ptr -> i32
// CHECK:           llvm.load %{{.*}} <alignment = 16> : !llvm.ptr -> i32
// CHECK:           llvm.load %{{.*}} <alignment = 8> : !llvm.ptr -> vector<2xi64>

// -----

func.func @int4_constant(%arg0: tensor<3xi4>, %arg1: index) -> i4 {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi4>
  %extracted = tensor.extract %arg0[%arg1] : tensor<3xi4>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<3xi4>
  %0 = arith.addi %extracted, %extracted_0 : i4
  return %0 : i4
}
// CHECK: llvm.mlir.global private constant
// CHECK-SAME: dense<[33, 3]>
// CHECK-LABEL: @int4_constant

// -----

func.func @for_op(%arg0: tensor<500xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %for:2 = scf.for %i = %c0 to %c2 step %c1
      iter_args(%cst_ = %cst, %arg_ = %arg0)
        -> (vector<4xf32>, tensor<500xf32>) {
    %nested_for:2 = scf.for %j = %c0 to %c2 step %c1
        iter_args(%cst__ = %cst_, %arg__ = %arg_)
          -> (vector<4xf32>, tensor<500xf32>) {
      %index = arith.addi %i, %j : index
      %tensor_elem = tensor.extract %arg__[%index] : tensor<500xf32>
      %vector_elem = vector.extract %cst__[%index] : f32 from vector<4xf32>
      %sum = arith.addf %tensor_elem, %vector_elem : f32
      %v_update = vector.insert %sum, %cst__[%index] : f32 into vector<4xf32>
      %t_update = tensor.insert %sum into %arg__[%index] : tensor<500xf32>
      scf.yield %v_update, %t_update : vector<4xf32>, tensor<500xf32>
    }
    scf.yield %nested_for#0, %nested_for#1 : vector<4xf32>, tensor<500xf32>
  }
  %result = tensor.extract %for#1[%c0] : tensor<500xf32>
  func.return %result : f32
}

// CHECK-LABEL: @for_op
// CHECK:  scf.for {{.*}} -> (vector<4xf32>) {
// CHECK-NEXT:  scf.for {{.*}} -> (vector<4xf32>) {

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

func.func @f4_constant(%arg0: tensor<3xf4E2M1FN>, %arg1: index) -> f4E2M1FN {
  %cst = arith.constant dense<[0.5, -0.5, 2.5]> : tensor<3xf4E2M1FN>
  %extracted = tensor.extract %arg0[%arg1] : tensor<3xf4E2M1FN>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<3xf4E2M1FN>
  %0 = arith.addf %extracted, %extracted_0 : f4E2M1FN
  return %0 : f4E2M1FN
}
// CHECK: llvm.mlir.global private constant
// CHECK-SAME: dense<[-111, 4]>
// CHECK-LABEL: @f4_constant

// -----

func.func @transfer_read_f4(%arg0: tensor<43xf4E2M1FN> {xla.slice_index = 1}) -> vector<2xf4E2M1FN> {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0.0 : f4E2M1FN
  %out = vector.transfer_read %arg0[%c16], %c0 : tensor<43xf4E2M1FN>, vector<2xf4E2M1FN>
  func.return %out : vector<2xf4E2M1FN>
}
// CHECK-LABEL: @transfer_read_f4
// CHECK: %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[0, 8] :
// CHECK-SAME: (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
// CHECK: %[[LD:.*]] = llvm.load %[[PTR]] <alignment = 1> : !llvm.ptr -> i8
// CHECK: %[[VEC:.*]] = llvm.bitcast %[[LD]] : i8 to vector<2xi4>
// CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[VEC]] : vector<2xi4> to vector<2xf4E2M1FN>
// CHECK: return %[[OUT]] : vector<2xf4E2M1FN>

// -----

func.func @transfer_write_f4(%arg0: tensor<43xf4E2M1FN> {xla.slice_index = 1},
                             %arg1: vector<2xf4E2M1FN>) -> tensor<43xf4E2M1FN> {
  %c10 = arith.constant 10 : index
  %out = vector.transfer_write %arg1, %arg0[%c10] : vector<2xf4E2M1FN>, tensor<43xf4E2M1FN>
  func.return %out : tensor<43xf4E2M1FN>
}
// CHECK-LABEL: @transfer_write_f4
// CHECK: %[[PTR:.*]] = llvm.getelementptr inbounds %arg0[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<22 x i8>
// CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %{{.*}} : vector<2xf4E2M1FN> to vector<2xi4>
// CHECK: %[[INT:.*]] = llvm.bitcast %[[OUT]] : vector<2xi4> to i8
// CHECK: llvm.store %[[INT]], %[[PTR]] <alignment = 1> : i8, !llvm.ptr

// -----

func.func @get_dynamic_dim_size(%arg0: tensor<512xf32>) -> i32 {
  %0 = xla.get_dynamic_dim_size %arg0 1 : tensor<512xf32>
  func.return %0 : i32
}
// CHECK-LABEL: @get_dynamic_dim_size
// CHECK: llvm.mlir.constant(2052 : i64) : i64

// -----

func.func @get_dynamic_dim_size_sub_byte_width(%arg0: tensor<512xi4>) -> i32 {
  %0 = xla.get_dynamic_dim_size %arg0 1 : tensor<512xi4>
  func.return %0 : i32
}
// CHECK-LABEL: @get_dynamic_dim_size_sub_byte_width
// CHECK: llvm.mlir.constant(260 : i64) : i64

// -----

func.func @get_dynamic_dim_size_unaligned(%arg0: tensor<7xf16>) -> i32 {
  %0 = xla.get_dynamic_dim_size %arg0 1 : tensor<7xf16>
  func.return %0 : i32
}
// CHECK-LABEL: @get_dynamic_dim_size_unaligned
// CHECK: llvm.mlir.constant(18 : i64) : i64
// CHECK: llvm.load %{{.*}} <alignment = 1> : !llvm.ptr -> i32

// -----

func.func @get_dynamic_dim_size_unaligned_odd(%arg0: tensor<3xi8>) -> i32 {
  %0 = xla.get_dynamic_dim_size %arg0 0 : tensor<3xi8>
  func.return %0 : i32
}
// CHECK-LABEL: @get_dynamic_dim_size_unaligned_odd
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.load %{{.*}} <alignment = 1> : !llvm.ptr -> i32

// -----

func.func @set_dynamic_dim_size(%arg0: tensor<512xf32>, %size: i32) -> tensor<512xf32> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[1] : tensor<512xf32>
  func.return %0 : tensor<512xf32>
}
// CHECK-LABEL: @set_dynamic_dim_size
// CHECK: llvm.mlir.constant(2052 : i64) : i64
// CHECK: llvm.store %{{.*}}, %{{.*}} <alignment = 4> : i32, !llvm.ptr

// -----

func.func @set_dynamic_dim_size_unaligned_sub_byte(%arg0: tensor<5xi4>, %size: i32) -> tensor<5xi4> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[0] : tensor<5xi4>
  func.return %0 : tensor<5xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_unaligned_sub_byte
// CHECK-SAME:    (%[[ARG0:.*]]: !llvm.ptr, %[[SIZE:.*]]: i32)
// CHECK-CPU:     llvm.mlir.constant(3 : i64) : i64
// CHECK-CPU:     llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i32, !llvm.ptr
// CHECK-GPU-DAG: %[[MASK:.*]] = llvm.mlir.constant(16777215 : i32) : i32
// CHECK-GPU-DAG: %[[SHIFT24:.*]] = llvm.mlir.constant(24 : i32) : i32
// CHECK-GPU-DAG: %[[SHIFT8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GPU-DAG: %[[SHIFT16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-GPU-DAG: %[[TAIL_OFF:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-GPU-DAG: %[[TOP_OFF:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-GPU:     %[[SHL:.*]] = llvm.shl %[[SIZE]], %[[SHIFT24]] : i32
// CHECK-GPU:     %[[INIT:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> i32
// CHECK-GPU:     scf.while (%[[CUR:.*]] = %[[INIT]]) : (i32) -> i32 {
// CHECK-GPU:       %[[KEPT:.*]] = llvm.and %[[CUR]], %[[MASK]] : i32
// CHECK-GPU:       %[[NEW:.*]] = llvm.or %[[KEPT]], %[[SHL]] : i32
// CHECK-GPU:       llvm.cmpxchg %[[ARG0]], %[[CUR]], %[[NEW]] monotonic monotonic : !llvm.ptr, i32
// CHECK-GPU:       scf.condition
// CHECK-GPU:     }
// CHECK-GPU:     %[[TAIL_INT:.*]] = llvm.add %{{.*}}, %[[TAIL_OFF]] : i64
// CHECK-GPU:     %[[TAIL_PTR:.*]] = llvm.inttoptr %[[TAIL_INT]] : i64 to !llvm.ptr
// CHECK-GPU:     %[[HIGH_SHIFTED:.*]] = llvm.lshr %[[SIZE]], %[[SHIFT8]] : i32
// CHECK-GPU:     %[[LOW_I16:.*]] = llvm.trunc %[[HIGH_SHIFTED]] : i32 to i16
// CHECK-GPU:     llvm.store %[[LOW_I16]], %[[TAIL_PTR]] <alignment = 2> : i16, !llvm.ptr
// CHECK-GPU:     %[[TOP_SHIFTED:.*]] = llvm.lshr %[[HIGH_SHIFTED]], %[[SHIFT16]] : i32
// CHECK-GPU:     %[[TOP_I8:.*]] = llvm.trunc %[[TOP_SHIFTED]] : i32 to i8
// CHECK-GPU:     %[[TOP_INT:.*]] = llvm.add %{{.*}}, %[[TOP_OFF]] : i64
// CHECK-GPU:     %[[TOP_PTR:.*]] = llvm.inttoptr %[[TOP_INT]] : i64 to !llvm.ptr
// CHECK-GPU:     llvm.store %[[TOP_I8]], %[[TOP_PTR]] <alignment = 1> : i8, !llvm.ptr
// CHECK-GFX908-MI100-LABEL: @set_dynamic_dim_size_unaligned_sub_byte
// CHECK-GFX908-MI100: llvm.cmpxchg {{.*}} syncscope("agent-one-as") monotonic monotonic : !llvm.ptr, i32

// -----

func.func @set_dynamic_dim_size_unaligned_sub_byte_misalign1(%arg0: tensor<9xi4>, %size: i32) -> tensor<9xi4> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[0] : tensor<9xi4>
  func.return %0 : tensor<9xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_unaligned_sub_byte_misalign1
// CHECK-CPU:     llvm.mlir.constant(5 : i64) : i64
// CHECK-CPU:     llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i32, !llvm.ptr
// CHECK-GPU-DAG: %[[WORD_OFF:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-GPU-DAG: %[[MASK:.*]] = llvm.mlir.constant(255 : i32) : i32
// CHECK-GPU-DAG: %[[SHIFT8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK-GPU-DAG: %[[TAIL_OFF:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-GPU-DAG: %[[SHIFT24:.*]] = llvm.mlir.constant(24 : i32) : i32
// CHECK-GPU:     llvm.add %{{.*}}, %[[WORD_OFF]] : i64
// CHECK-GPU:     llvm.cmpxchg {{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-GPU:     llvm.add %{{.*}}, %[[TAIL_OFF]] : i64
// CHECK-GPU:     llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i8, !llvm.ptr

// -----

func.func @set_dynamic_dim_size_unaligned_sub_byte_misalign2(%arg0: tensor<3xi4>, %size: i32) -> tensor<3xi4> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[0] : tensor<3xi4>
  func.return %0 : tensor<3xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_unaligned_sub_byte_misalign2
// CHECK-CPU:     llvm.mlir.constant(2 : i64) : i64
// CHECK-CPU:     llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i32, !llvm.ptr
// CHECK-GPU-DAG: %[[MASK:.*]] = llvm.mlir.constant(65535 : i32) : i32
// CHECK-GPU-DAG: %[[SHIFT16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-GPU-DAG: %[[TAIL_OFF:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-GPU:     llvm.cmpxchg {{.*}} monotonic monotonic : !llvm.ptr, i32
// CHECK-GPU:     llvm.add %{{.*}}, %[[TAIL_OFF]] : i64
// CHECK-GPU:     llvm.store %{{.*}}, %{{.*}} <alignment = 2> : i16, !llvm.ptr

// -----

func.func @set_dynamic_dim_size_sub_byte_dim1(%arg0: tensor<5xi4>, %size: i32) -> tensor<5xi4> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[1] : tensor<5xi4>
  func.return %0 : tensor<5xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_sub_byte_dim1
// CHECK-NOT:     llvm.cmpxchg
// CHECK:         llvm.mlir.constant(7 : i64) : i64
// CHECK:         llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i32, !llvm.ptr

// -----

func.func @set_dynamic_dim_size_aligned_sub_byte(%arg0: tensor<8xi4>, %size: i32) -> tensor<8xi4> {
  %0 = xla.set_dynamic_dim_size %size into %arg0[0] : tensor<8xi4>
  func.return %0 : tensor<8xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_aligned_sub_byte
// CHECK-NOT:     llvm.cmpxchg
// CHECK:         llvm.mlir.constant(4 : i64) : i64
// CHECK:         llvm.store %{{.*}}, %{{.*}} <alignment = 4> : i32, !llvm.ptr

// -----

func.func @set_dynamic_dim_size_before_sub_byte_insert(
    %arg0: tensor<5xi4>, %size: i32, %val: i4, %idx: index, %cond0: i1,
    %cond1: i1) -> tensor<5xi4> {
  %with_dim = scf.if %cond0 -> (tensor<5xi4>) {
    %updated = xla.set_dynamic_dim_size %size into %arg0[0] : tensor<5xi4>
    scf.yield %updated : tensor<5xi4>
  } else {
    scf.yield %arg0 : tensor<5xi4>
  }
  %res = scf.if %cond1 -> (tensor<5xi4>) {
    %ins = tensor.insert %val into %with_dim[%idx] : tensor<5xi4>
    scf.yield %ins : tensor<5xi4>
  } else {
    scf.yield %with_dim : tensor<5xi4>
  }
  func.return %res : tensor<5xi4>
}
// CHECK-LABEL: @set_dynamic_dim_size_before_sub_byte_insert
// CHECK: scf.if %{{.*}} {
// CHECK-GPU:   llvm.cmpxchg
// CHECK-CPU:   llvm.store %{{.*}}, %{{.*}} <alignment = 1> : i32, !llvm.ptr
// CHECK: }
// CHECK: scf.if %{{.*}} {
// CHECK-GPU: llvm.cmpxchg
// CHECK-CPU: llvm.store
// CHECK: return

// -----

func.func @get_destination_buffer_through_cast(
    %arg0: tensor<2x4xf32>, %val: f32, %idx: index) -> tensor<2x4xf32> {
  %flat = builtin.unrealized_conversion_cast %arg0 : tensor<2x4xf32> to tensor<8xf32>
  %ins = tensor.insert %val into %flat[%idx] : tensor<8xf32>
  %res = builtin.unrealized_conversion_cast %ins : tensor<8xf32> to tensor<2x4xf32>
  func.return %res : tensor<2x4xf32>
}
// CHECK-LABEL: func.func @get_destination_buffer_through_cast(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr,
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK:         llvm.store %{{.*}}, %[[GEP]] : f32, !llvm.ptr
// CHECK:         return

// -----

func.func @get_destination_buffer_through_atomic_rmw(
    %arg0: tensor<8xf32>, %val: f32, %idx: index) -> tensor<8xf32> {
  %rmw = xla.atomic_rmw %arg0[%idx] : tensor<8xf32> {
    ^bb0(%cur: f32):
      %add = arith.addf %cur, %val : f32
      xla.yield %add : f32
  }
  %ins = tensor.insert %val into %rmw[%idx] : tensor<8xf32>
  func.return %ins : tensor<8xf32>
}
// CHECK-LABEL: func.func @get_destination_buffer_through_atomic_rmw(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr,
// CHECK-GPU:     llvm.atomicrmw ignore_denormal_mode fadd
// CHECK-CPU:     llvm.cmpxchg
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[ARG0]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x f32>
// CHECK:         llvm.store %{{.*}}, %[[GEP]] : f32, !llvm.ptr
// CHECK:         return

