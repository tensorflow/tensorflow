// RUN: xla-opt %s --triton-xla-remote-access | FileCheck %s

// CHECK-LABEL: module {
// CHECK-LABEL: tt.func @get_rank(%arg0: !tt.ptr<i64>) -> i64 {
tt.func @get_rank(
  %metadata: !tt.ptr<i64>
) -> (i64) {
  // CHECK-NOT: triton_xla.get_rank
  // CHECK: %0 = tt.load %arg0 : !tt.ptr<i64>
  %rank = triton_xla.get_rank %metadata : !tt.ptr<i64> -> i64
  tt.return %rank : i64
}

tt.func @get_peer_ptr(
  %arg0: !tt.ptr<i64>, %peer_id: i64, %metadata: !tt.ptr<i64>
) -> !tt.ptr<i64> {
  // CHECK-NOT: triton_xla.get_peer_ptr
  // Offset to local_buffer_root_ptrs.
  // CHECK: %c72_i64 = arith.constant 72 : i64

  // Byte size of a pointer.
  // CHECK-NEXT: %c8_i64 = arith.constant 8 : i64

  // Load metadata->rank
  // CHECK-NEXT: %0 = tt.load %arg2 : !tt.ptr<i64>

  // Calculate offset to current base pointer.
  // CHECK-NEXT: %1 = arith.muli %0, %c8_i64 : i64

  // Load metadata->local_buffer_root_ptrs[metadata->rank].
  // CHECK-NEXT: %2 = arith.addi %1, %c72_i64 : i64
  // CHECK-NEXT: %3 = tt.addptr %arg2, %2 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %4 = tt.load %3 : !tt.ptr<i64>

  // Calculate offset to address.
  // CHECK-NEXT: %5 = tt.ptr_to_int %arg0 : !tt.ptr<i64> -> i64
  // CHECK-NEXT: %6 = arith.subi %5, %4 : i64

  // Calculate offset to peer base pointer.
  // CHECK-NEXT: %7 = arith.muli %arg1, %c8_i64 : i64
  // CHECK-NEXT: %8 = arith.addi %7, %c72_i64 : i64

  // Load metadata->local_buffer_root_ptrs[peer_id].
  // CHECK-NEXT: %9 = tt.addptr %arg2, %8 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %10 = tt.load %9 : !tt.ptr<i64>

  // Load metadata->local_buffer_root_ptrs[peer_id] + offset.
  // CHECK-NEXT: %11 = arith.addi %10, %6 : i64
  // CHECK-NEXT: %12 = tt.int_to_ptr %11 : i64 -> !tt.ptr<i64>
  // CHECK-NEXT: tt.return %12 : !tt.ptr<i64>
  %peer_ptr = triton_xla.get_peer_ptr %arg0, %peer_id, %metadata : (!tt.ptr<i64>, i64, !tt.ptr<i64>) -> !tt.ptr<i64>
  tt.return %peer_ptr : !tt.ptr<i64>
}