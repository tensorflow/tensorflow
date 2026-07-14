; AMDGPU kernel with dynamic stack usage (indirect function call)
; Based on real HIP code that uses function pointers
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_40fa47637d275275 = addrspace(1) global i8 0

@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_40fa47637d275275 to ptr)], section "llvm.metadata"

; Kernel that uses indirect function call requiring dynamic stack
define protected amdgpu_kernel void @_Z4TestPDF16bS_S_(ptr addrspace(1) noundef %dst.coerce, ptr addrspace(1) noundef %ptr1.coerce, ptr addrspace(1) noundef %ptr2.coerce) local_unnamed_addr {
entry:
  %0 = ptrtoint ptr addrspace(1) %dst.coerce to i64
  %1 = inttoptr i64 %0 to ptr
  %2 = ptrtoint ptr addrspace(1) %ptr1.coerce to i64
  %3 = inttoptr i64 %2 to ptr
  %4 = ptrtoint ptr addrspace(1) %ptr2.coerce to i64
  %5 = inttoptr i64 %4 to ptr
  %6 = tail call ptr asm "", "=s"() #1
  tail call void %6(ptr noundef %1, ptr noundef %3, ptr noundef %5) #2
  ret void
}

attributes #1 = { nounwind }
attributes #2 = { nounwind }

