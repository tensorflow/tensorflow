// RUN: xla-opt %s --triton-xla-block-barrier | FileCheck %s

// CHECK-LABEL: block_barrier_kernel
// CHECK-SAME: %[[PTR:.+]]: !tt.ptr<!tt.ptr<i32>>,
// CHECK-SAME: %[[SIGNAL_VALUE:.+]]: i32,
// CHECK-SAME: %[[RANK:.+]]: i32
tt.func @block_barrier_kernel(
  %ptr : !tt.ptr<!tt.ptr<i32>>, %signal_value : i32, %rank: i32) {
  // CHECK-NEXT: %[[WORLD_SIZE:.+]] = arith.constant 8 : i32
  // CHECK-NEXT: %[[TID:.+]] = triton_xla.get_tid
  // CHECK-NEXT: %[[PROGRAM_ID:.+]] = tt.get_program_id x
  // CHECK-NEXT: %[[COND:.+]] = arith.cmpi ult, %[[TID]], %[[WORLD_SIZE]]
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   %[[BITCAST:.+]] = tt.bitcast %[[PTR]]
  // CHECK-NEXT:   %[[SPLAT_PTR:.+]] = tt.splat %[[BITCAST]]
  // CHECK-NEXT:   %[[RANGE:.+]] = tt.make_range {end = 8 : i32, start = 0 : i32}
  // CHECK-NEXT:   %[[ADD_PTR:.+]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
  // CHECK-NEXT:   %[[LOAD:.+]] = tt.load %[[ADD_PTR]]
  // CHECK-NEXT:   %[[INT_TO_PTR:.+]] = tt.int_to_ptr %[[LOAD]]
  // CHECK-NEXT:   %[[MULI:.+]] = arith.muli %[[PROGRAM_ID]], %[[WORLD_SIZE]]
  // CHECK-NEXT:   %[[ADDI:.+]] = arith.addi %[[MULI]], %[[RANK]]
  // CHECK-NEXT:   %[[SPLAT_ADDI:.+]] = tt.splat %[[ADDI]]
  // CHECK-NEXT:   %[[ADD_PTR_2:.+]] = tt.addptr %[[INT_TO_PTR]], %[[SPLAT_ADDI]]
  // CHECK-NEXT:   triton_xla.atomic_write sys, release, %[[ADD_PTR_2]], %[[SIGNAL_VALUE]]
  // CHECK-NEXT:   %[[ADD_PTR_3:.+]] = tt.addptr %[[BITCAST]], %[[RANK]]
  // CHECK-NEXT:   %[[LOAD_2:.+]] = tt.load %[[ADD_PTR_3]]
  // CHECK-NEXT:   %[[INT_TO_PTR_2:.+]] = tt.int_to_ptr %[[LOAD_2]]
  // CHECK-NEXT:   %[[ADD_PTR_4:.+]] = tt.addptr %[[INT_TO_PTR_2]], %[[MULI]]
  // CHECK-NEXT:   %[[SPLAT_ADD_PTR_4:.+]] = tt.splat %[[ADD_PTR_4]]
  // CHECK-NEXT:   %[[ADD_PTR_5:.+]] = tt.addptr %[[SPLAT_ADD_PTR_4]], %[[RANGE]]
  // CHECK-NEXT:   triton_xla.atomic_spin_wait sys, acquire, %[[ADD_PTR_5]], less_than, %[[SIGNAL_VALUE]]
  // CHECK-NEXT: }
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.return
  triton_xla.block_barrier %ptr, %rank, %signal_value, { world_size = 8 : i32 } :
    (!tt.ptr<!tt.ptr<i32>>, i32, i32) -> ()
  tt.return
}