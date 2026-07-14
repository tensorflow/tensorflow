; AMDGPU kernel with high SGPR pressure to force scalar register spilling
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Kernel using many scalar operations with limited SGPRs
; We use readfirstlane to force values into SGPRs
define amdgpu_kernel void @sgpr_pressure(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx = zext i32 %tid to i64

  ; Load many scalar values from memory
  ; Using readfirstlane forces values into SGPRs (uniform across wavefront)
  %ptr0 = getelementptr i32, ptr addrspace(1) %in, i64 0
  %v0_vec = load i32, ptr addrspace(1) %ptr0, align 4
  %v0 = call i32 @llvm.amdgcn.readfirstlane(i32 %v0_vec)

  %ptr1 = getelementptr i32, ptr addrspace(1) %in, i64 1
  %v1_vec = load i32, ptr addrspace(1) %ptr1, align 4
  %v1 = call i32 @llvm.amdgcn.readfirstlane(i32 %v1_vec)

  %ptr2 = getelementptr i32, ptr addrspace(1) %in, i64 2
  %v2_vec = load i32, ptr addrspace(1) %ptr2, align 4
  %v2 = call i32 @llvm.amdgcn.readfirstlane(i32 %v2_vec)

  %ptr3 = getelementptr i32, ptr addrspace(1) %in, i64 3
  %v3_vec = load i32, ptr addrspace(1) %ptr3, align 4
  %v3 = call i32 @llvm.amdgcn.readfirstlane(i32 %v3_vec)

  %ptr4 = getelementptr i32, ptr addrspace(1) %in, i64 4
  %v4_vec = load i32, ptr addrspace(1) %ptr4, align 4
  %v4 = call i32 @llvm.amdgcn.readfirstlane(i32 %v4_vec)

  %ptr5 = getelementptr i32, ptr addrspace(1) %in, i64 5
  %v5_vec = load i32, ptr addrspace(1) %ptr5, align 4
  %v5 = call i32 @llvm.amdgcn.readfirstlane(i32 %v5_vec)

  %ptr6 = getelementptr i32, ptr addrspace(1) %in, i64 6
  %v6_vec = load i32, ptr addrspace(1) %ptr6, align 4
  %v6 = call i32 @llvm.amdgcn.readfirstlane(i32 %v6_vec)

  %ptr7 = getelementptr i32, ptr addrspace(1) %in, i64 7
  %v7_vec = load i32, ptr addrspace(1) %ptr7, align 4
  %v7 = call i32 @llvm.amdgcn.readfirstlane(i32 %v7_vec)

  %ptr8 = getelementptr i32, ptr addrspace(1) %in, i64 8
  %v8_vec = load i32, ptr addrspace(1) %ptr8, align 4
  %v8 = call i32 @llvm.amdgcn.readfirstlane(i32 %v8_vec)

  %ptr9 = getelementptr i32, ptr addrspace(1) %in, i64 9
  %v9_vec = load i32, ptr addrspace(1) %ptr9, align 4
  %v9 = call i32 @llvm.amdgcn.readfirstlane(i32 %v9_vec)

  %ptr10 = getelementptr i32, ptr addrspace(1) %in, i64 10
  %v10_vec = load i32, ptr addrspace(1) %ptr10, align 4
  %v10 = call i32 @llvm.amdgcn.readfirstlane(i32 %v10_vec)

  %ptr11 = getelementptr i32, ptr addrspace(1) %in, i64 11
  %v11_vec = load i32, ptr addrspace(1) %ptr11, align 4
  %v11 = call i32 @llvm.amdgcn.readfirstlane(i32 %v11_vec)

  %ptr12 = getelementptr i32, ptr addrspace(1) %in, i64 12
  %v12_vec = load i32, ptr addrspace(1) %ptr12, align 4
  %v12 = call i32 @llvm.amdgcn.readfirstlane(i32 %v12_vec)

  %ptr13 = getelementptr i32, ptr addrspace(1) %in, i64 13
  %v13_vec = load i32, ptr addrspace(1) %ptr13, align 4
  %v13 = call i32 @llvm.amdgcn.readfirstlane(i32 %v13_vec)

  %ptr14 = getelementptr i32, ptr addrspace(1) %in, i64 14
  %v14_vec = load i32, ptr addrspace(1) %ptr14, align 4
  %v14 = call i32 @llvm.amdgcn.readfirstlane(i32 %v14_vec)

  %ptr15 = getelementptr i32, ptr addrspace(1) %in, i64 15
  %v15_vec = load i32, ptr addrspace(1) %ptr15, align 4
  %v15 = call i32 @llvm.amdgcn.readfirstlane(i32 %v15_vec)

  ; Create many scalar computations - chain A
  %a0 = add i32 %v0, %v1
  %a1 = mul i32 %a0, %v2
  %a2 = add i32 %a1, %v3
  %a3 = mul i32 %a2, %v4
  %a4 = add i32 %a3, %v5
  %a5 = mul i32 %a4, %v6
  %a6 = add i32 %a5, %v7
  %a7 = mul i32 %a6, %v8
  %a8 = add i32 %a7, %v9
  %a9 = mul i32 %a8, %v10
  %a10 = add i32 %a9, %v11
  %a11 = mul i32 %a10, %v12
  %a12 = add i32 %a11, %v13
  %a13 = mul i32 %a12, %v14
  %a14 = add i32 %a13, %v15

  ; Chain B - reverse
  %b0 = mul i32 %v15, %v14
  %b1 = add i32 %b0, %v13
  %b2 = mul i32 %b1, %v12
  %b3 = add i32 %b2, %v11
  %b4 = mul i32 %b3, %v10
  %b5 = add i32 %b4, %v9
  %b6 = mul i32 %b5, %v8
  %b7 = add i32 %b6, %v7
  %b8 = mul i32 %b7, %v6
  %b9 = add i32 %b8, %v5
  %b10 = mul i32 %b9, %v4
  %b11 = add i32 %b10, %v3
  %b12 = mul i32 %b11, %v2
  %b13 = add i32 %b12, %v1
  %b14 = mul i32 %b13, %v0

  ; Chain C - subtraction
  %c0 = sub i32 %v0, %v1
  %c1 = mul i32 %c0, %v2
  %c2 = sub i32 %c1, %v3
  %c3 = mul i32 %c2, %v4
  %c4 = sub i32 %c3, %v5
  %c5 = mul i32 %c4, %v6
  %c6 = sub i32 %c5, %v7
  %c7 = mul i32 %c6, %v8
  %c8 = sub i32 %c7, %v9
  %c9 = mul i32 %c8, %v10
  %c10 = sub i32 %c9, %v11
  %c11 = mul i32 %c10, %v12
  %c12 = sub i32 %c11, %v13
  %c13 = mul i32 %c12, %v14
  %c14 = sub i32 %c13, %v15

  ; Chain D - cross dependencies
  %d0 = add i32 %a0, %b0
  %d1 = mul i32 %d0, %c0
  %d2 = add i32 %a1, %b1
  %d3 = mul i32 %d2, %c1
  %d4 = add i32 %a2, %b2
  %d5 = mul i32 %d4, %c2
  %d6 = add i32 %a3, %b3
  %d7 = mul i32 %d6, %c3
  %d8 = add i32 %a4, %b4
  %d9 = mul i32 %d8, %c4
  %d10 = add i32 %a5, %b5
  %d11 = mul i32 %d10, %c5
  %d12 = add i32 %a6, %b6
  %d13 = mul i32 %d12, %c6

  ; Combine all chains
  %r0 = add i32 %a14, %b14
  %r1 = add i32 %r0, %c14
  %r2 = add i32 %r1, %d1
  %r3 = add i32 %r2, %d3
  %r4 = add i32 %r3, %d5
  %r5 = add i32 %r4, %d7
  %r6 = add i32 %r5, %d9
  %r7 = add i32 %r6, %d11
  %result = add i32 %r7, %d13

  %out_ptr = getelementptr i32, ptr addrspace(1) %out, i64 %tidx
  store i32 %result, ptr addrspace(1) %out_ptr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.readfirstlane(i32) #1

; Limit SGPRs to 32, this should force SGPR spilling
attributes #0 = { "amdgpu-num-sgpr"="32" "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind readnone speculatable }
