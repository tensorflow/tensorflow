// RUN: mlir-translate -mlir-to-nvvmir %s | FileCheck %s

llvm.func @nvvm_special_regs() -> !llvm.i32 {
  // CHECK: %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
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
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  %13 = nvvm.read.ptx.sreg.warpsize : !llvm.i32
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %14 = nvvm.read.ptx.sreg.laneid : !llvm.i32
  llvm.return %1 : !llvm.i32
}

llvm.func @llvm.nvvm.barrier0() {
  // CHECK: call void @llvm.nvvm.barrier0()
  nvvm.barrier0
  llvm.return
}

llvm.func @nvvm_shfl(
    %0 : !llvm.i32, %1 : !llvm.i32, %2 : !llvm.i32,
    %3 : !llvm.i32, %4 : !llvm.float) -> !llvm.i32 {
  // CHECK: call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync.bfly %0, %3, %1, %2 : !llvm.i32
  // CHECK: call float @llvm.nvvm.shfl.sync.bfly.f32(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync.bfly %0, %4, %1, %2 : !llvm.float
  llvm.return %6 : !llvm.i32
}

llvm.func @nvvm_shfl_pred(
    %0 : !llvm.i32, %1 : !llvm.i32, %2 : !llvm.i32,
    %3 : !llvm.i32, %4 : !llvm.float) -> !llvm<"{ i32, i1 }"> {
  // CHECK: call { i32, i1 } @llvm.nvvm.shfl.sync.bfly.i32p(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %6 = nvvm.shfl.sync.bfly %0, %3, %1, %2 {return_value_and_is_valid} : !llvm<"{ i32, i1 }">
  // CHECK: call { float, i1 } @llvm.nvvm.shfl.sync.bfly.f32p(i32 %{{.*}}, float %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %7 = nvvm.shfl.sync.bfly %0, %4, %1, %2 {return_value_and_is_valid} : !llvm<"{ float, i1 }">
  llvm.return %6 : !llvm<"{ i32, i1 }">
}

llvm.func @nvvm_vote(%0 : !llvm.i32, %1 : !llvm.i1) -> !llvm.i32 {
  // CHECK: call i32 @llvm.nvvm.vote.ballot.sync(i32 %{{.*}}, i1 %{{.*}})
  %3 = nvvm.vote.ballot.sync %0, %1 : !llvm.i32
  llvm.return %3 : !llvm.i32
}

llvm.func @nvvm_mma(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
                    %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
                    %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
                    %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // CHECK: call { float, float, float, float, float, float, float, float } @llvm.nvvm.mma.m8n8k4.row.row.f32.f32
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}

// This function has the "kernel" attribute attached and should appear in the
// NVVM annotations after conversion.
llvm.func @kernel_func() attributes {gpu.kernel} {
  llvm.return
}

// CHECK:     !nvvm.annotations =
// CHECK-NOT: {i32 ()* @nvvm_special_regs, !"kernel", i32 1}
// CHECK:     {void ()* @kernel_func, !"kernel", i32 1}
