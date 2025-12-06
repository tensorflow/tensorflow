; RUN: ptx_opt  %s --arch=9.0 | FileCheck %s

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

define ptx_kernel void @loop_acos_fusion(ptr noalias align 16 dereferenceable(1024) %0, ptr noalias align 256 dereferenceable(1024) %1) #0 {
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !range !1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %5 = mul i32 %3, 128
  %6 = add i32 %5, %4
  %7 = getelementptr inbounds [256 x float], ptr %0, i32 0, i32 %6
  %8 = load float, ptr %7, align 4, !invariant.load !3
  %9 = call float @__nv_acosf(float %8)
  %10 = getelementptr inbounds [256 x float], ptr %1, i32 0, i32 %6
  store float %9, ptr %10, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

declare float @__nv_acosf(float)

attributes #0 = { "nvvm.reqntid"="128,1,1" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 0, i32 2}
!2 = !{i32 0, i32 128}
!3 = !{}

; CHECK: .target sm_90