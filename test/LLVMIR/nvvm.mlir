// RUN: mlir-opt %s | FileCheck %s

func @nvvm_special_regs() -> !llvm.i32 {
  // CHECK: %0 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
  %0 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
  // CHECK: %1 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
  %1 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
  // CHECK: %2 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
  %2 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
  // CHECK: %3 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  %3 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  // CHECK: %4 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  %4 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  // CHECK: %5 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  %5 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  // CHECK: %6 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  %6 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  // CHECK: %7 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  %7 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  // CHECK: %8 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  %8 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  // CHECK: %9 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  %9 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  // CHECK: %10 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  %10 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  // CHECK: %11 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  %11 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  llvm.return %0 : !llvm.i32
}
