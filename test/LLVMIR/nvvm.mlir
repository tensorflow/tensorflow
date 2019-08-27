// RUN: mlir-opt %s | FileCheck %s

func @nvvm_special_regs() -> !llvm.i32 {
  // CHECK: nvvm.read.ptx.sreg.tid.x : !llvm.i32
  %0 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.tid.y : !llvm.i32
  %1 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.tid.z : !llvm.i32
  %2 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  %3 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  %4 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  %5 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  %6 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  %7 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  %8 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  %9 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  %10 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  %11 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  llvm.return %0 : !llvm.i32
}

func @llvm.nvvm.barrier0() {
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  llvm.return
}

func @nvvm_shfl(
    %arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32,
    %arg3 : !llvm.i32, %arg4 : !llvm.float) -> !llvm.i32 {
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i32
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 : !llvm.i32
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.float
  %1 = nvvm.shfl.sync.bfly %arg0, %arg4, %arg1, %arg2 : !llvm.float
  llvm.return %0 : !llvm.i32
}

func @nvvm_vote(%arg0 : !llvm.i32, %arg1 : !llvm.i1) -> !llvm.i32 {
  // CHECK: nvvm.vote.ballot.sync %{{.*}}, %{{.*}} : !llvm.i32
  %0 = nvvm.vote.ballot.sync %arg0, %arg1 : !llvm.i32
  llvm.return %0 : !llvm.i32
}
