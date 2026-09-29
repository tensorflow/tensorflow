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
// RUN: xla-opt %s --triton-xla-block-barrier | FileCheck %s

// CHECK-LABEL: block_barrier_kernel
// CHECK-SAME: %[[PTR:.+]]: !tt.ptr<i64>,
// CHECK-SAME: %[[SIGNAL_VALUE:.+]]: i32,
// CHECK-SAME: %[[RANK:.+]]: i32
tt.func @block_barrier_kernel(
  %ptr : !tt.ptr<i64>, %signal_value : i32, %rank: i32) {
  // CHECK-NEXT: %[[WORLD_SIZE:.+]] = arith.constant 8 : i32
  // CHECK-NEXT: %[[TID:.+]] = triton_xla.get_tid
  // CHECK-NEXT: %[[PROGRAM_ID:.+]] = tt.get_program_id x
  // CHECK-NEXT: %[[COND:.+]] = arith.cmpi ult, %[[TID]], %[[WORLD_SIZE]]
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   %[[SPLAT_PTR:.+]] = tt.splat %[[PTR]]
  // CHECK-NEXT:   %[[RANGE:.+]] = tt.make_range {end = 8 : i32, start = 0 : i32}
  // CHECK-NEXT:   %[[ADD_PTR:.+]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
  // CHECK-NEXT:   %[[LOAD:.+]] = tt.load %[[ADD_PTR]]
  // CHECK-NEXT:   %[[INT_TO_PTR:.+]] = tt.int_to_ptr %[[LOAD]]
  // CHECK-NEXT:   %[[MULI:.+]] = arith.muli %[[PROGRAM_ID]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[ADDI:.+]] = arith.addi %[[MULI]], %[[RANK]]
  // CHECK-NEXT:   %[[SPLAT_ADDI:.+]] = tt.splat %[[ADDI]]
  // CHECK-NEXT:   %[[ADD_PTR_2:.+]] = tt.addptr %[[INT_TO_PTR]], %[[SPLAT_ADDI]]
  // CHECK-NEXT:   triton_xla.atomic_write sys, release, %[[ADD_PTR_2]], %[[SIGNAL_VALUE]]
  // CHECK-NEXT:   %[[ADD_PTR_3:.+]] = tt.addptr %[[PTR]], %[[RANK]]
  // CHECK-NEXT:   %[[LOAD_2:.+]] = tt.load %[[ADD_PTR_3]]
  // CHECK-NEXT:   %[[INT_TO_PTR_2:.+]] = tt.int_to_ptr %[[LOAD_2]]
  // CHECK-NEXT:   %[[ADD_PTR_4:.+]] = tt.addptr %[[INT_TO_PTR_2]], %[[MULI]]
  // CHECK-NEXT:   %[[SPLAT_ADD_PTR_4:.+]] = tt.splat %[[ADD_PTR_4]]
  // CHECK-NEXT:   %[[ADD_PTR_5:.+]] = tt.addptr %[[SPLAT_ADD_PTR_4]], %[[RANGE]]
  // CHECK-NEXT:   triton_xla.atomic_spin_wait sys, acquire, %[[ADD_PTR_5]], greater_than_or_equal_to, %[[SIGNAL_VALUE]]
  // CHECK-NEXT: }
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.return
  triton_xla.block_barrier %ptr, %rank, %signal_value <world_size = 8> :
    (!tt.ptr<i64>, i32, i32) -> ()
  tt.return
}

// CHECK-LABEL: block_barrier_consumer_symmetric_kernel
// CHECK-SAME: %[[PTR:[^:]+]]: !tt.ptr<i64>,
// CHECK-SAME: %[[SIGNAL_VALUE:[^:]+]]: i32,
// CHECK-SAME: %[[RANK:[^:]+]]: i32,
// CHECK-SAME: %[[SIGNAL_SLOT:[^:]+]]: i32
tt.func @block_barrier_consumer_symmetric_kernel(
  %ptr : !tt.ptr<i64>, %signal_value : i32, %rank: i32, %signal_slot: i32) {
  // CHECK-NEXT: %[[STRIDE:.+]] = arith.constant 4 : i32
  // CHECK-NEXT: %[[WORLD_SIZE:.+]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[TID:.+]] = triton_xla.get_tid
  // CHECK-NEXT: %[[PROGRAM_ID:.+]] = tt.get_program_id x
  // CHECK-NEXT: %[[COND:.+]] = arith.cmpi ult, %[[TID]], %[[WORLD_SIZE]]
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   %[[SPLAT_PTR:.+]] = tt.splat %[[PTR]]
  // CHECK-NEXT:   %[[RANGE:.+]] = tt.make_range {end = 2 : i32, start = 0 : i32}
  // CHECK-NEXT:   %[[ADD_PTR:.+]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
  // CHECK-NEXT:   %[[LOAD:.+]] = tt.load %[[ADD_PTR]]
  // CHECK-NEXT:   %[[INT_TO_PTR:.+]] = tt.int_to_ptr %[[LOAD]]
  // CHECK-NEXT:   %[[MULI:.+]] = arith.muli %[[PROGRAM_ID]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[S_OFFSET:.+]] = arith.muli %[[SIGNAL_SLOT]], %[[STRIDE]]
  // CHECK-NEXT:   %[[BASE_T:.+]] = arith.subi %[[PROGRAM_ID]], %[[S_OFFSET]]
  // CHECK-NEXT:   %[[RANK_OFFSET:.+]] = arith.muli %[[RANK]], %[[STRIDE]]
  // CHECK-NEXT:   %[[WRITE_BLOCK:.+]] = arith.addi %[[BASE_T]], %[[RANK_OFFSET]]
  // CHECK-NEXT:   %[[WRITE_BLOCK_OFFSET:.+]] = arith.muli %[[WRITE_BLOCK]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[WRITE_OFFSET:.+]] = arith.addi %[[WRITE_BLOCK_OFFSET]], %[[SIGNAL_SLOT]]
  // CHECK-NEXT:   %[[SPLAT_WRITE_OFFSET:.+]] = tt.splat %[[WRITE_OFFSET]]
  // CHECK-NEXT:   %[[ADD_PTR_2:.+]] = tt.addptr %[[INT_TO_PTR]], %[[SPLAT_WRITE_OFFSET]]
  // CHECK-NEXT:   triton_xla.atomic_write sys, release, %[[ADD_PTR_2]], %[[SIGNAL_VALUE]]
  // CHECK-NEXT:   %[[ADD_PTR_3:.+]] = tt.addptr %[[PTR]], %[[RANK]]
  // CHECK-NEXT:   %[[LOAD_2:.+]] = tt.load %[[ADD_PTR_3]]
  // CHECK-NEXT:   %[[INT_TO_PTR_2:.+]] = tt.int_to_ptr %[[LOAD_2]]
  // CHECK-NEXT:   %[[ADD_PTR_4:.+]] = tt.addptr %[[INT_TO_PTR_2]], %[[MULI]]
  // CHECK-NEXT:   %[[SPLAT_ADD_PTR_4:.+]] = tt.splat %[[ADD_PTR_4]]
  // CHECK-NEXT:   %[[ADD_PTR_5:.+]] = tt.addptr %[[SPLAT_ADD_PTR_4]], %[[RANGE]]
  // CHECK-NEXT:   triton_xla.atomic_spin_wait sys, acquire, %[[ADD_PTR_5]], greater_than_or_equal_to, %[[SIGNAL_VALUE]]
  // CHECK-NEXT: }
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.return
  triton_xla.block_barrier %ptr, %rank, %signal_value, %signal_slot
    <world_size = 2, signal_stride = 4, barrier_mode = consumer_symmetric> :
    (!tt.ptr<i64>, i32, i32, i32) -> ()
  tt.return
}

// CHECK-LABEL: block_barrier_producer_symmetric_kernel
// CHECK-SAME: %[[PTR:[^:]+]]: !tt.ptr<i64>,
// CHECK-SAME: %[[SIGNAL_VALUE:[^:]+]]: i32,
// CHECK-SAME: %[[RANK:[^:]+]]: i32,
// CHECK-SAME: %[[SIGNAL_SLOT:[^:]+]]: i32
tt.func @block_barrier_producer_symmetric_kernel(
  %ptr : !tt.ptr<i64>, %signal_value : i32, %rank: i32, %signal_slot: i32) {
  // CHECK-NEXT: %[[STRIDE:.+]] = arith.constant 4 : i32
  // CHECK-NEXT: %[[WORLD_SIZE:.+]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[TID:.+]] = triton_xla.get_tid
  // CHECK-NEXT: %[[PROGRAM_ID:.+]] = tt.get_program_id x
  // CHECK-NEXT: %[[COND:.+]] = arith.cmpi ult, %[[TID]], %[[WORLD_SIZE]]
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   %[[SPLAT_PTR:.+]] = tt.splat %[[PTR]]
  // CHECK-NEXT:   %[[RANGE:.+]] = tt.make_range {end = 2 : i32, start = 0 : i32}
  // CHECK-NEXT:   %[[ADD_PTR:.+]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
  // CHECK-NEXT:   %[[LOAD:.+]] = tt.load %[[ADD_PTR]]
  // CHECK-NEXT:   %[[INT_TO_PTR:.+]] = tt.int_to_ptr %[[LOAD]]
  // CHECK-NEXT:   %[[MULI:.+]] = arith.muli %[[PROGRAM_ID]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[S_OFFSET:.+]] = arith.muli %[[SIGNAL_SLOT]], %[[STRIDE]]
  // CHECK-NEXT:   %[[BASE_T:.+]] = arith.subi %[[PROGRAM_ID]], %[[S_OFFSET]]
  // CHECK-NEXT:   %[[RANK_OFFSET:.+]] = arith.muli %[[RANK]], %[[STRIDE]]
  // CHECK-NEXT:   %[[WAIT_BLOCK:.+]] = arith.addi %[[BASE_T]], %[[RANK_OFFSET]]
  // CHECK-NEXT:   %[[WAIT_BLOCK_OFFSET:.+]] = arith.muli %[[WAIT_BLOCK]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[WRITE_OFFSET:.+]] = arith.addi %[[MULI]], %[[RANK]]
  // CHECK-NEXT:   %[[SPLAT_WRITE_OFFSET:.+]] = tt.splat %[[WRITE_OFFSET]]
  // CHECK-NEXT:   %[[ADD_PTR_2:.+]] = tt.addptr %[[INT_TO_PTR]], %[[SPLAT_WRITE_OFFSET]]
  // CHECK-NEXT:   triton_xla.atomic_write sys, release, %[[ADD_PTR_2]], %[[SIGNAL_VALUE]]
  // CHECK-NEXT:   %[[ADD_PTR_3:.+]] = tt.addptr %[[PTR]], %[[RANK]]
  // CHECK-NEXT:   %[[LOAD_2:.+]] = tt.load %[[ADD_PTR_3]]
  // CHECK-NEXT:   %[[INT_TO_PTR_2:.+]] = tt.int_to_ptr %[[LOAD_2]]
  // CHECK-NEXT:   %[[ADD_PTR_4:.+]] = tt.addptr %[[INT_TO_PTR_2]], %[[WAIT_BLOCK_OFFSET]]
  // CHECK-NEXT:   %[[SPLAT_ADD_PTR_4:.+]] = tt.splat %[[ADD_PTR_4]]
  // CHECK-NEXT:   %[[ADD_PTR_5:.+]] = tt.addptr %[[SPLAT_ADD_PTR_4]], %[[RANGE]]
  // CHECK-NEXT:   triton_xla.atomic_spin_wait sys, acquire, %[[ADD_PTR_5]], greater_than_or_equal_to, %[[SIGNAL_VALUE]]
  // CHECK-NEXT: }
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.return
  triton_xla.block_barrier %ptr, %rank, %signal_value, %signal_slot
    <world_size = 2, signal_stride = 4, barrier_mode = producer_symmetric> :
    (!tt.ptr<i64>, i32, i32, i32) -> ()
  tt.return
}
