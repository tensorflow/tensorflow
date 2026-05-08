// RUN: xla-opt %s -triton-xla-implement-extern-element-wise | FileCheck %s

// Test CUDA implementation of extern_elementwise atomic functions
// This pass operates on LLVM dialect and generates LLVM intrinsics

// Test unmasked operations
module {
  // CHECK-LABEL: llvm.func @test_get_thread_id
  llvm.func @test_get_thread_id() -> i32 {
    // CHECK-NOT: llvm.call @xla_getthreadid
    // CHECK: [[TID:%.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.tid.x"() : () -> i32
    // CHECK: llvm.return [[TID]]
    %tid = llvm.call @xla_getthreadid() : () -> i32
    llvm.return %tid : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_write_unmasked
  llvm.func @test_atomic_write_unmasked(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicwrite_release_system_nomask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.store %arg1, %arg0 atomic release {alignment = 4 : i64} : i32, !llvm.ptr<1>
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_system_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_unmasked
  llvm.func @test_atomic_spin_wait_unmasked(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicspinwait_acquire_system_lt_nomask
    // CHECK: llvm.br ^[[LOOP:.*]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "ult" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT:.*]], ^[[LOOP]]
    // CHECK: ^[[EXIT]]:
    // CHECK:   llvm.return [[LOADED]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_lt_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_eq
  llvm.func @test_atomic_spin_wait_eq(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK: llvm.br ^[[LOOP:.*]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "eq" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT:.*]], ^[[LOOP]]
    // CHECK: ^[[EXIT]]:
    // CHECK:   llvm.return [[LOADED]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_eq_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }

  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_ge
  llvm.func @test_atomic_spin_wait_ge(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK: llvm.br ^[[LOOP:.*]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "uge" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT:.*]], ^[[LOOP]]
    // CHECK: ^[[EXIT]]:
    // CHECK:   llvm.return [[LOADED]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_ge_nomask(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_relaxed_ordering
  llvm.func @test_relaxed_ordering(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.store %arg1, %arg0 atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr<1>
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_relaxed_system_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_gpu_scope
  llvm.func @test_gpu_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.store %arg1, %arg0 atomic syncscope("device") release {alignment = 4 : i64} : i32, !llvm.ptr<1>
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_gpu_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_cta_scope
  llvm.func @test_cta_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: llvm.store %arg1, %arg0 atomic syncscope("block") release {alignment = 4 : i64} : i32, !llvm.ptr<1>
    // CHECK: llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_cta_nomask(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_acquire_ordering
  llvm.func @test_acquire_ordering(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK: llvm.br ^[[LOOP:.*]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "ult" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT:.*]], ^[[LOOP]]
    // CHECK: ^[[EXIT]]:
    // CHECK:   llvm.return [[LOADED]]
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
  llvm.func @xla_atomicspinwait_acquire_system_ge_nomask(!llvm.ptr<1>, i32) -> i32
}

// Test masked operations in separate module to avoid function redefinition
module {
  // CHECK-LABEL: llvm.func @test_atomic_write_masked
  llvm.func @test_atomic_write_masked(%ptr: !llvm.ptr<1>, %value: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicwrite_release_system_mask
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : i32
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[ATOMIC:.*]], ^[[EXIT:.*]]
    // CHECK: ^[[ATOMIC]]:
    // CHECK:   llvm.store %arg1, %arg0 atomic release {alignment = 4 : i64} : i32, !llvm.ptr<1>
    // CHECK:   llvm.br ^[[EXIT]]
    // CHECK: ^[[EXIT]]:
    // CHECK:   llvm.return [[POISON]]
    %result = llvm.call @xla_atomicwrite_release_system_mask(%ptr, %value, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked
  llvm.func @test_atomic_spin_wait_masked(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomicspinwait_acquire_system_lt_mask
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[LOOP:.*]], ^[[EXIT:.*]]([[ZERO]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "ult" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT]]([[LOADED]] : i32), ^[[LOOP]]
    // CHECK: ^[[EXIT]]([[RESULT:%.*]]: i32):
    // CHECK:   llvm.return [[RESULT]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_lt_mask(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked_eq
  llvm.func @test_atomic_spin_wait_masked_eq(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[LOOP:.*]], ^[[EXIT:.*]]([[ZERO]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "eq" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT]]([[LOADED]] : i32), ^[[LOOP]]
    // CHECK: ^[[EXIT]]([[RESULT:%.*]]: i32):
    // CHECK:   llvm.return [[RESULT]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_eq_mask(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }

  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked_ge
  llvm.func @test_atomic_spin_wait_masked_ge(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[LOOP:.*]], ^[[EXIT:.*]]([[ZERO]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "uge" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[EXIT]]([[LOADED]] : i32), ^[[LOOP]]
    // CHECK: ^[[EXIT]]([[RESULT:%.*]]: i32):
    // CHECK:   llvm.return [[RESULT]]
    %result = llvm.call @xla_atomicspinwait_acquire_system_ge_mask(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // Extern function declarations for the masked operations.
  llvm.func @xla_atomicwrite_release_system_mask(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_lt_mask(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_eq_mask(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomicspinwait_acquire_system_ge_mask(!llvm.ptr<1>, i32, i32) -> i32
}
