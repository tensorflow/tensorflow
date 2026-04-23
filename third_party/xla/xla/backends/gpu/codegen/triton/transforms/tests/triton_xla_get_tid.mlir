// RUN: xla-opt %s --triton-xla-get-tid | FileCheck %s

// Test lowering of GetTidOp to tt.extern_elementwise

// CHECK-LABEL: @test_get_tid
tt.func @test_get_tid() -> i32 {
  // CHECK-NOT: triton_xla.get_tid
  // CHECK: [[TID:%.*]] = tt.extern_elementwise
  // CHECK-SAME: pure = true
  // CHECK-SAME: symbol = "xla_getthreadid"
  // CHECK: tt.return [[TID]]
  %tid = triton_xla.get_tid : () -> i32
  tt.return %tid : i32
}

// CHECK-LABEL: @test_get_tid_used_in_computation
tt.func @test_get_tid_used_in_computation(%value: i32) -> i32 {
  // CHECK: [[TID:%.*]] = tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_getthreadid"
  // CHECK: [[RESULT:%.*]] = arith.addi [[TID]], %arg0
  // CHECK: tt.return [[RESULT]]
  %tid = triton_xla.get_tid : () -> i32
  %result = arith.addi %tid, %value : i32
  tt.return %result : i32
}

// CHECK-LABEL: @test_multiple_get_tid_calls
tt.func @test_multiple_get_tid_calls() -> i32 {
  // CHECK: [[TID1:%.*]] = tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_getthreadid"
  // CHECK: [[TID2:%.*]] = tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_getthreadid"
  // CHECK: [[RESULT:%.*]] = arith.addi [[TID1]], [[TID2]]
  // CHECK: tt.return [[RESULT]]
  %tid1 = triton_xla.get_tid : () -> i32
  %tid2 = triton_xla.get_tid : () -> i32
  %result = arith.addi %tid1, %tid2 : i32
  tt.return %result : i32
}
