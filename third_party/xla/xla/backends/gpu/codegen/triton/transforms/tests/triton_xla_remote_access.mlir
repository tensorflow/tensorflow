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
  %arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %peer_id: i64, %metadata: !tt.ptr<i64>
) -> !tt.ptr<i64> {
  // CHECK-NOT: triton_xla.get_peer_ptr
  // An offset from the beginning of metadata to the peer pointers for the %arg1
  // offset(param_to_peers) + sizeof(uint64_t) * 2 = 20
  // CHECK: %c24_i64 = arith.constant 24 : i64
  // Size of the uint64_t.
  // CHECK: %c8_i64 = arith.constant 8 : i64

  // Load metadata->rank
  // CHECK-NEXT: %0 = tt.load %arg3 : !tt.ptr<i64>

  // Calculate offset to current base pointer.
  // CHECK-NEXT: %1 = arith.muli %0, %c8_i64 : i64

  // Load metadata->param_to_peers[argument_offset + metadata->rank].
  // Here argument_offset = 0 since %arg0 is the first argument.
  // CHECK-NEXT: %2 = arith.addi %1, %c8_i64 : i64
  // CHECK-NEXT: %3 = tt.addptr %arg3, %2 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %4 = tt.load %3 : !tt.ptr<i64>

  // Calculate offset to address.
  // CHECK-NEXT: %5 = tt.ptr_to_int %arg0 : !tt.ptr<i64> -> i64
  // CHECK-NEXT: %6 = arith.subi %5, %4 : i64

  // Calculate offset to peer base pointer.
  // CHECK-NEXT: %7 = arith.muli %arg2, %c8_i64 : i64
  // CHECK-NEXT: %8 = arith.addi %7, %c8_i64 : i64

  // Load metadata->peer_base_ptrs[argument_offset + peer_id].
  // CHECK-NEXT: %9 = tt.addptr %arg3, %8 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %10 = tt.load %9 : !tt.ptr<i64>

  // Load metadata->buffer_root_ptrs[argument_offset + peer_id] + offset.
  // CHECK-NEXT: %11 = arith.addi %10, %6 : i64
  // CHECK-NEXT: %12 = tt.int_to_ptr %11 : i64 -> !tt.ptr<i64>
  %arg_0_peer_ptr = triton_xla.get_peer_ptr %arg0, %peer_id, %metadata,
     { argument_index = 0 : i32, world_size = 2 : i32 } :
     (!tt.ptr<i64>, i64, !tt.ptr<i64>) -> !tt.ptr<i64>

  // Load metadata->rank
  // CHECK-NEXT: %13 = tt.load %arg3 : !tt.ptr<i64>
  // Calculate offset to current base pointer.
  // CHECK-NEXT: %14 = arith.muli %13, %c8_i64 : i64
  // Load metadata->param_to_peers[argument_offset + metadata->rank].
  // CHECK-NEXT: %15 = arith.addi %14, %c24_i64 : i64
  // CHECK-NEXT: %16 = tt.addptr %arg3, %15 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %17 = tt.load %16 : !tt.ptr<i64>
  // Calculate offset to address.
  // CHECK-NEXT: %18 = tt.ptr_to_int %arg1 : !tt.ptr<i64> -> i64
  // CHECK-NEXT: %19 = arith.subi %18, %17 : i64

  // Calculate offset to peer base pointer.
  // CHECK-NEXT: %20 = arith.muli %arg2, %c8_i64 : i64
  // CHECK-NEXT: %21 = arith.addi %20, %c24_i64 : i64

  // Load metadata->peer_base_ptrs[argument_offset + peer_id].
  // CHECK-NEXT: %22 = tt.addptr %arg3, %21 : !tt.ptr<i64>, i64
  // CHECK-NEXT: %23 = tt.load %22 : !tt.ptr<i64>

  // Load metadata->buffer_root_ptrs[argument_offset + peer_id] + offset.
  // CHECK-NEXT: %24 = arith.addi %23, %19 : i64
  // CHECK-NEXT: %25 = tt.int_to_ptr %24 : i64 -> !tt.ptr<i64>

  %arg_1_peer_ptr = triton_xla.get_peer_ptr %arg1, %peer_id, %metadata,
     { argument_index = 1 : i32, world_size = 2 : i32 } :
     (!tt.ptr<i64>, i64, !tt.ptr<i64>) -> !tt.ptr<i64>
  
  // Avoid optimizing away the get_peer_ptr calls, by returning xor of the two
  // peer pointers.
  // 
  // CHECK-NEXT: %26 = tt.ptr_to_int %12 : !tt.ptr<i64> -> i64
  %int_arg0 = tt.ptr_to_int %arg_0_peer_ptr : !tt.ptr<i64> -> i64
  // CHECK-NEXT: %27 = tt.ptr_to_int %25 : !tt.ptr<i64> -> i64
  %int_arg1 = tt.ptr_to_int %arg_1_peer_ptr : !tt.ptr<i64> -> i64

  // CHECK-NEXT: %28 = arith.ori %26, %27 : i64
  %result_int = arith.ori %int_arg0, %int_arg1 : i64
  // CHECK-NEXT: %29 = tt.int_to_ptr %28 : i64 -> !tt.ptr<i64>
  %result_ptr = tt.int_to_ptr %result_int : i64 -> !tt.ptr<i64>
  // CHECK-NEXT: tt.return %29 : !tt.ptr<i64>
  tt.return %result_ptr : !tt.ptr<i64>
}