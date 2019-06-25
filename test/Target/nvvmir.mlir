// RUN: mlir-translate -mlir-to-nvvmir %s | FileCheck %s

func @nvvm_special_regs() -> !llvm.i32 {
  // CHECK: %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
  // CHECK: %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %2 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
  // CHECK: %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %3 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
  // CHECK: %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  // CHECK: %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %5 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  // CHECK: %6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %6 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  // CHECK: %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  // CHECK: %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  // CHECK: %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %9 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  // CHECK: %10 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %10 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  // CHECK: %11 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %11 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  // CHECK: %12 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %12 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  llvm.return %1 : !llvm.i32
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
func @kernel_func() attributes {gpu.kernel} {
  llvm.return
}

// CHECK:     !nvvm.annotations =
// CHECK-NOT: {i32 ()* @nvvm_special_regs, !"kernel", i32 1}
// CHECK:     {void ()* @kernel_func, !"kernel", i32 1}
