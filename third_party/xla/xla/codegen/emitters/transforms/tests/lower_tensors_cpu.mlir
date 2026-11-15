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

// -----

func.func @copy_slice(%src: tensor<12xf32>, %dst: tensor<20xf32>,
                      %src_idx: index, %dst_idx: index) -> tensor<20xf32> {
  %res = xla.copy_slice %src[%src_idx] into %dst[%dst_idx] count(6)
      : tensor<12xf32> into tensor<20xf32>
  func.return %res : tensor<20xf32>
}

// CHECK-LABEL: func.func @copy_slice(
// CHECK-SAME:      %[[SRC:.*]]: !llvm.ptr, %[[DST:.*]]: !llvm.ptr,
// CHECK-SAME:      %[[SRC_IDX:.*]]: index, %[[DST_IDX:.*]]: index) {
// CHECK-DAG:     %[[BYTES:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-DAG:     %[[SRC_PTR:.*]] = llvm.getelementptr inbounds %[[SRC]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<12 x f32>
// CHECK-DAG:     %[[DST_PTR:.*]] = llvm.getelementptr inbounds %[[DST]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<20 x f32>
// CHECK:         "llvm.intr.memcpy"(%[[DST_PTR]], %[[SRC_PTR]], %[[BYTES]]) <{arg_attrs = [{llvm.align = 4 : i64}, {llvm.align = 4 : i64}, {}], isVolatile = false}>
// CHECK:         return

// -----

func.func @copy_slice_aligned_args(
    %src: tensor<12xf32> {llvm.align = 32 : index},
    %dst: tensor<20xf32> {llvm.align = 32 : index},
    %src_i8: tensor<12xi8> {llvm.align = 32 : index},
    %dst_i8: tensor<20xi8> {llvm.align = 32 : index},
    %dyn_idx: index) -> (tensor<20xf32>, tensor<20xi8>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %r0 = xla.copy_slice %src[%c0] into %dst[%c8] count(4)
      : tensor<12xf32> into tensor<20xf32>
  %r1 = xla.copy_slice %src[%c0] into %r0[%c3] count(4)
      : tensor<12xf32> into tensor<20xf32>
  %r2 = xla.copy_slice %src[%dyn_idx] into %r1[%dyn_idx] count(4)
      : tensor<12xf32> into tensor<20xf32>
  %b0 = xla.copy_slice %src_i8[%c0] into %dst_i8[%c0] count(4)
      : tensor<12xi8> into tensor<20xi8>
  %b1 = xla.copy_slice %src_i8[%dyn_idx] into %b0[%dyn_idx] count(4)
      : tensor<12xi8> into tensor<20xi8>
  func.return %r2, %b1 : tensor<20xf32>, tensor<20xi8>
}

// CHECK-LABEL: func.func @copy_slice_aligned_args(
// CHECK:         "llvm.intr.memcpy"{{.*}}<{arg_attrs = [{llvm.align = 32 : i64}, {llvm.align = 32 : i64}, {}], isVolatile = false}>
// CHECK:         "llvm.intr.memcpy"{{.*}}<{arg_attrs = [{llvm.align = 4 : i64}, {llvm.align = 32 : i64}, {}], isVolatile = false}>
// CHECK:         "llvm.intr.memcpy"{{.*}}<{arg_attrs = [{llvm.align = 4 : i64}, {llvm.align = 4 : i64}, {}], isVolatile = false}>
// CHECK:         "llvm.intr.memcpy"{{.*}}<{arg_attrs = [{llvm.align = 32 : i64}, {llvm.align = 32 : i64}, {}], isVolatile = false}>
// CHECK:         "llvm.intr.memcpy"{{.*}}<{isVolatile = false}>
