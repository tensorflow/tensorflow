// RUN: xla-opt %s -triton-xla-implement-extern-element-wise | FileCheck %s

// Test CUDA implementation of extern_elementwise atomic functions
// This pass operates on LLVM dialect and inlines PTX assembly implementations

// Test unmasked operations
module {
  // CHECK-LABEL: llvm.func @test_get_thread_id
  llvm.func @test_get_thread_id() -> i32 {
    // CHECK-NOT: llvm.call @xla_getthreadid
    // CHECK: [[TID:%.*]] = llvm.inline_asm {{.*}}mov.u32 $0, %tid.x;{{.*}}, "=r" : () -> i32
    // CHECK: llvm.return [[TID]]
    %tid = llvm.call @xla_getthreadid() : () -> i32
    llvm.return %tid : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_write_unmasked
  llvm.func @test_atomic_write_unmasked(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicwrite_release_system_nomask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: st.global.sys.release.u32 [$0], $1;
    // CHECK-SAME: ", "l,r" %arg0, %arg1 : (!llvm.ptr<1>, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_system_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_unmasked
  llvm.func @test_atomic_spin_wait_unmasked(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicspinwait_acquire_system_lt_nomask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: .reg .pred %p<1>;
    // CHECK-SAME: .reg .b32 %r<1>;
    // CHECK-SAME: wait:
    // CHECK-SAME: ld.global.sys.acquire.u32 %r0, [$0];
    // CHECK-SAME: setp.lt.u32 %p0, %r0, $1;
    // CHECK-SAME: @%p0 bra wait;
    // CHECK-SAME: ", "l,r" %arg0, %arg1 : (!llvm.ptr<1>, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_lt_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_eq
  llvm.func @test_atomic_spin_wait_eq(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: setp.eq.u32 %p0, %r0, $1;
    // CHECK-SAME: ", "l,r" %arg0, %arg1 : (!llvm.ptr<1>, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_eq_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_relaxed_ordering
  llvm.func @test_relaxed_ordering(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: st.global.sys.relaxed.u32 [$0], $1;
    // CHECK-SAME: ", "l,r"
    %result = llvm.call @xla_atomicwrite_relaxed_system_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_gpu_scope
  llvm.func @test_gpu_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: st.global.gpu.release.u32 [$0], $1;
    // CHECK-SAME: ", "l,r"
    %result = llvm.call @xla_atomicwrite_release_gpu_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_cta_scope
  llvm.func @test_cta_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: st.global.cta.release.u32 [$0], $1;
    // CHECK-SAME: ", "l,r"
    %result = llvm.call @xla_atomicwrite_release_cta_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_acquire_ordering
  llvm.func @test_acquire_ordering(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: ld.global.sys.acquire.u32 %r0, [$0];
    // CHECK-SAME: ", "l,r"
    %result = llvm.call @xla_atomicspinwait_acquire_system_lt_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // Extern function declarations (will be removed by the pass)
  llvm.func @xla_getthreadid() -> i32
  llvm.func @xla_atomicwrite_release_system_nomask(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomicwrite_relaxed_system_nomask(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomicwrite_release_gpu_nomask(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomicwrite_release_cta_nomask(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_lt_nomask(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_eq_nomask(!llvm.ptr<1>, i32) -> i32
}

// Test masked operations in separate module to avoid function redefinition
module {
  // CHECK-LABEL: llvm.func @test_atomic_write_masked
  llvm.func @test_atomic_write_masked(%ptr: !llvm.ptr<1>, %value: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicwrite_release_system_mask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: .reg .pred %p<>;
    // CHECK-SAME: setp.ne.u32 %p<>, $2, 0;
    // CHECK-SAME: @%p<> st.global.sys.release.u32 [$0], $1;
    // CHECK-SAME: ", "l,r,r" %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i32, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_system_mask(%ptr, %value, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked
  llvm.func @test_atomic_spin_wait_masked(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicspinwait_acquire_system_lt_mask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: .reg .pred %p<2>;
    // CHECK-SAME: .reg .b32 %r<1>;
    // CHECK-SAME: setp.ne.u32 %p0, $2, 0;
    // CHECK-SAME: @%!p0 bra done;
    // CHECK-SAME: wait:
    // CHECK-SAME: ld.global.sys.acquire.u32 %r0, [$0];
    // CHECK-SAME: setp.lt.u32 %p1, %r0, $1;
    // CHECK-SAME: @%p1 bra wait;
    // CHECK-SAME: done:
    // CHECK-SAME: ", "l,r,r" %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i32, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_lt_mask(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked_eq
  llvm.func @test_atomic_spin_wait_masked_eq(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.inline_asm has_side_effects "
    // CHECK-SAME: setp.eq.u32 %p1, %r0, $1;
    // CHECK-SAME: ", "l,r,r" %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i32, i32) -> ()
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_eq_mask(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // Extern function declarations for the masked operations.
  llvm.func @xla_atomicwrite_release_system_mask(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_lt_mask(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_eq_mask(!llvm.ptr<1>, i32, i32) -> i32
}
