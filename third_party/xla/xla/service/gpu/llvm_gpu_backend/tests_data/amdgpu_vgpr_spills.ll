; AMDGPU kernel with high register pressure to force spilling
; This uses many vector operations to exhaust available VGPRs
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Kernel with many live values to force register spilling
define amdgpu_kernel void @high_register_pressure(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx = zext i32 %tid to i64

  ; Load many vectors from memory - using volatile to prevent optimization
  %ptr0 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 %tidx
  %v0 = load volatile <4 x float>, ptr addrspace(1) %ptr0, align 16

  %ptr1 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 1
  %v1 = load volatile <4 x float>, ptr addrspace(1) %ptr1, align 16

  %ptr2 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 2
  %v2 = load volatile <4 x float>, ptr addrspace(1) %ptr2, align 16

  %ptr3 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 3
  %v3 = load volatile <4 x float>, ptr addrspace(1) %ptr3, align 16

  %ptr4 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 4
  %v4 = load volatile <4 x float>, ptr addrspace(1) %ptr4, align 16

  %ptr5 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 5
  %v5 = load volatile <4 x float>, ptr addrspace(1) %ptr5, align 16

  %ptr6 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 6
  %v6 = load volatile <4 x float>, ptr addrspace(1) %ptr6, align 16

  %ptr7 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 7
  %v7 = load volatile <4 x float>, ptr addrspace(1) %ptr7, align 16

  %ptr8 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 8
  %v8 = load volatile <4 x float>, ptr addrspace(1) %ptr8, align 16

  %ptr9 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 9
  %v9 = load volatile <4 x float>, ptr addrspace(1) %ptr9, align 16

  %ptr10 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 10
  %v10 = load volatile <4 x float>, ptr addrspace(1) %ptr10, align 16

  %ptr11 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 11
  %v11 = load volatile <4 x float>, ptr addrspace(1) %ptr11, align 16

  %ptr12 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 12
  %v12 = load volatile <4 x float>, ptr addrspace(1) %ptr12, align 16

  %ptr13 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 13
  %v13 = load volatile <4 x float>, ptr addrspace(1) %ptr13, align 16

  %ptr14 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 14
  %v14 = load volatile <4 x float>, ptr addrspace(1) %ptr14, align 16

  %ptr15 = getelementptr <4 x float>, ptr addrspace(1) %in, i64 15
  %v15 = load volatile <4 x float>, ptr addrspace(1) %ptr15, align 16

  ; Create many dependent calculations - chain A
  %a0 = fadd <4 x float> %v0, %v1
  %a1 = fmul <4 x float> %a0, %v2
  %a2 = fadd <4 x float> %a1, %v3
  %a3 = fmul <4 x float> %a2, %v4
  %a4 = fadd <4 x float> %a3, %v5
  %a5 = fmul <4 x float> %a4, %v6
  %a6 = fadd <4 x float> %a5, %v7
  %a7 = fmul <4 x float> %a6, %v8
  %a8 = fadd <4 x float> %a7, %v9
  %a9 = fmul <4 x float> %a8, %v10
  %a10 = fadd <4 x float> %a9, %v11
  %a11 = fmul <4 x float> %a10, %v12
  %a12 = fadd <4 x float> %a11, %v13
  %a13 = fmul <4 x float> %a12, %v14
  %a14 = fadd <4 x float> %a13, %v15

  ; Chain B - reverse direction
  %b0 = fmul <4 x float> %v15, %v14
  %b1 = fadd <4 x float> %b0, %v13
  %b2 = fmul <4 x float> %b1, %v12
  %b3 = fadd <4 x float> %b2, %v11
  %b4 = fmul <4 x float> %b3, %v10
  %b5 = fadd <4 x float> %b4, %v9
  %b6 = fmul <4 x float> %b5, %v8
  %b7 = fadd <4 x float> %b6, %v7
  %b8 = fmul <4 x float> %b7, %v6
  %b9 = fadd <4 x float> %b8, %v5
  %b10 = fmul <4 x float> %b9, %v4
  %b11 = fadd <4 x float> %b10, %v3
  %b12 = fmul <4 x float> %b11, %v2
  %b13 = fadd <4 x float> %b12, %v1
  %b14 = fmul <4 x float> %b13, %v0

  ; Chain C - subtraction chain
  %c0 = fsub <4 x float> %v0, %v1
  %c1 = fmul <4 x float> %c0, %v2
  %c2 = fsub <4 x float> %c1, %v3
  %c3 = fmul <4 x float> %c2, %v4
  %c4 = fsub <4 x float> %c3, %v5
  %c5 = fmul <4 x float> %c4, %v6
  %c6 = fsub <4 x float> %c5, %v7
  %c7 = fmul <4 x float> %c6, %v8
  %c8 = fsub <4 x float> %c7, %v9
  %c9 = fmul <4 x float> %c8, %v10
  %c10 = fsub <4 x float> %c9, %v11
  %c11 = fmul <4 x float> %c10, %v12
  %c12 = fsub <4 x float> %c11, %v13
  %c13 = fmul <4 x float> %c12, %v14
  %c14 = fsub <4 x float> %c13, %v15

  ; Chain D - cross dependencies
  %d0 = fadd <4 x float> %a0, %b0
  %d1 = fmul <4 x float> %d0, %c0
  %d2 = fadd <4 x float> %a1, %b1
  %d3 = fmul <4 x float> %d2, %c1
  %d4 = fadd <4 x float> %a2, %b2
  %d5 = fmul <4 x float> %d4, %c2
  %d6 = fadd <4 x float> %a3, %b3
  %d7 = fmul <4 x float> %d6, %c3
  %d8 = fadd <4 x float> %a4, %b4
  %d9 = fmul <4 x float> %d8, %c4
  %d10 = fadd <4 x float> %a5, %b5
  %d11 = fmul <4 x float> %d10, %c5

  ; Final combination to keep all values live
  %result0 = fadd <4 x float> %a14, %b14
  %result1 = fadd <4 x float> %result0, %c14
  %result2 = fadd <4 x float> %result1, %d1
  %result3 = fadd <4 x float> %result2, %d3
  %result4 = fadd <4 x float> %result3, %d5
  %result5 = fadd <4 x float> %result4, %d7
  %result6 = fadd <4 x float> %result5, %d9
  %result = fadd <4 x float> %result6, %d11

  %out_ptr = getelementptr <4 x float>, ptr addrspace(1) %out, i64 %tidx
  store <4 x float> %result, ptr addrspace(1) %out_ptr, align 16
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

; Limit VGPRs to 64 to force spilling
attributes #0 = { "amdgpu-num-vgpr"="64" "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind readnone speculatable }
