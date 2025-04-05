// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @conditional_barrier() {
    // CHECK-LABEL: llvm.func @conditional_barrier

    // CHECK:   %[[CMP0:.+]] = llvm.icmp "ne" %3, %1 : i32
    // CHECK:   %[[CMP1:.+]] = llvm.icmp "eq" %3, %1 : i32
    // CHECK:   llvm.cond_br %[[CMP0]], ^bb1, ^bb2
    // CHECK: ^bb1:
    // CHECK:   rocdl.s.barrier
    // CHECK:   llvm.br ^bb2
    // CHECK: ^bb2:
    // CHECK:   llvm.add
    // CHECK:   llvm.cond_br %[[CMP1]], ^bb3, ^bb4
    // CHECK: ^bb3:
    // CHECK:   rocdl.s.barrier
    // CHECK:   llvm.br ^bb4
    // CHECK: ^bb4:
    // CHECK:   llvm.return

    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = rocdl.workitem.id.x : i32
    %1 = arith.divsi %0, %c256_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = arith.cmpi eq, %1, %c0_i32 : i32
    amdgpu.cond_barrier %2
    %4 = arith.addi %0, %c256_i32 : i32
    amdgpu.cond_barrier %3
    tt.return
  }
}
