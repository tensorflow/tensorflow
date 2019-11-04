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

func @nvvm_shfl_pred(
    %arg0 : !llvm.i32, %arg1 : !llvm.i32, %arg2 : !llvm.i32,
    %arg3 : !llvm.i32, %arg4 : !llvm.float) -> !llvm<"{ i32, i1 }"> {
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"{ i32, i1 }">
  %0 = nvvm.shfl.sync.bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : !llvm<"{ i32, i1 }">
  // CHECK: nvvm.shfl.sync.bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm<"{ float, i1 }">
  %1 = nvvm.shfl.sync.bfly %arg0, %arg4, %arg1, %arg2 {return_value_and_is_valid} : !llvm<"{ float, i1 }">
  llvm.return %0 : !llvm<"{ i32, i1 }">
}

func @nvvm_vote(%arg0 : !llvm.i32, %arg1 : !llvm.i1) -> !llvm.i32 {
  // CHECK: nvvm.vote.ballot.sync %{{.*}}, %{{.*}} : !llvm.i32
  %0 = nvvm.vote.ballot.sync %arg0, %arg1 : !llvm.i32
  llvm.return %0 : !llvm.i32
}

func @nvvm_mma(%a0 : !llvm<"<2 x half>">, %a1 : !llvm<"<2 x half>">,
               %b0 : !llvm<"<2 x half>">, %b1 : !llvm<"<2 x half>">,
               %c0 : !llvm.float, %c1 : !llvm.float, %c2 : !llvm.float, %c3 : !llvm.float,
               %c4 : !llvm.float, %c5 : !llvm.float, %c6 : !llvm.float, %c7 : !llvm.float) {
  // CHECK: nvvm.mma.sync {{.*}} {alayout = "row", blayout = "row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  %0 = nvvm.mma.sync %a0, %a1, %b0, %b1, %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 {alayout="row", blayout="row"} : (!llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm<"<2 x half>">, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float, !llvm.float) -> !llvm<"{ float, float, float, float, float, float, float, float }">
  llvm.return %0 : !llvm<"{ float, float, float, float, float, float, float, float }">
}
