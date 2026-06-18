; Simple AMDGPU kernel for testing register spilling detection
; This module has no external dependencies and minimal module flags
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Simple kernel that adds two arrays
define amdgpu_kernel void @simple_add(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx = zext i32 %tid to i64

  %a_ptr = getelementptr float, ptr addrspace(1) %a, i64 %tidx
  %b_ptr = getelementptr float, ptr addrspace(1) %b, i64 %tidx
  %c_ptr = getelementptr float, ptr addrspace(1) %c, i64 %tidx

  %a_val = load float, ptr addrspace(1) %a_ptr, align 4
  %b_val = load float, ptr addrspace(1) %b_ptr, align 4

  %sum = fadd float %a_val, %b_val

  store float %sum, ptr addrspace(1) %c_ptr, align 4
  ret void
}

; Intrinsic declaration
declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone speculatable }

