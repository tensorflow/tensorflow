// RUN: mlir-translate -mlir-to-nvvmir %s | FileCheck %s

func @nvvm_special_regs() -> !llvm.i32 {
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = nvvm.read.ptx.sreg.tid.x : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %2 = nvvm.read.ptx.sreg.tid.y : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %3 = nvvm.read.ptx.sreg.tid.z : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %4 = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %5 = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %6 = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %8 = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %9 = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %10 = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %11 = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %12 = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
  llvm.return %1 : !llvm.i32
}

func @llvm.nvvm.barrier0() {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier0
  llvm.return
}

func @nvvm_shfl(
    %0 : !llvm.i32, %1 : !llvm.i32, %2 : !llvm.i32,
    %3 : !llvm.i32, %4 : !llvm.float) -> !llvm.i32 {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync.bfly %0, %3, %1, %2 : !llvm.i32
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync.bfly %0, %4, %1, %2 : !llvm.float
  llvm.return %6 : !llvm.i32
}

func @nvvm_vote(%0 : !llvm.i32, %1 : !llvm.i1) -> !llvm.i32 {
  // CHECK: call i32 @llvm.nvvm.vote.ballot.sync(i32 %{{.*}}, i1 %{{.*}})
  %3 = nvvm.vote.ballot.sync %0, %1 : !llvm.i32
  llvm.return %3 : !llvm.i32
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
func @kernel_func() attributes {gpu.kernel} {
  llvm.return
}

// CHECK:     !nvvm.annotations =
// CHECK-NOT: {i32 ()* @nvvm_special_regs, !"kernel", i32 1}
// CHECK:     {void ()* @kernel_func, !"kernel", i32 1}
