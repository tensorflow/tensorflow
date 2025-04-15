target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

%struct.uint3 = type { i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }

@blockIdx = external addrspace(1) global %struct.uint3
@blockDim = external addrspace(1) global %struct.dim3
@threadIdx = external addrspace(1) global %struct.uint3

; Function Attrs: alwaysinline nounwind readnone
define float @expf(float %f) #0 {
entry:
  %f.addr = alloca float, align 4
  store float %f, float* %f.addr, align 4
  %0 = load float, float* %f.addr, align 4
  %call = call float @__nv_expf(float %0)
  ret float %call
}

declare float @__nv_expf(float) #1

; Function Attrs: nounwind
define void @cuda_saxpy(i32* %n, float* %a, float* %x, float* %y) #2 {
entry:
  %n.addr = alloca i32*, align 8
  %a.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %i = alloca i32, align 4
  store i32* %n, i32** %n.addr, align 8
  store float* %a, float** %a.addr, align 8
  store float* %x, float** %x.addr, align 8
  store float* %y, float** %y.addr, align 8
  %0 = load i32, i32* getelementptr inbounds (%struct.uint3, %struct.uint3* addrspacecast (%struct.uint3 addrspace(1)* @blockIdx to %struct.uint3*), i32 0, i32 0), align 4
  %1 = load i32, i32* getelementptr inbounds (%struct.dim3, %struct.dim3* addrspacecast (%struct.dim3 addrspace(1)* @blockDim to %struct.dim3*), i32 0, i32 0), align 4
  %mul = mul i32 %0, %1
  %2 = load i32, i32* getelementptr inbounds (%struct.uint3, %struct.uint3* addrspacecast (%struct.uint3 addrspace(1)* @threadIdx to %struct.uint3*), i32 0, i32 0), align 4
  %add = add i32 %mul, %2
  store i32 %add, i32* %i, align 4
  %3 = load i32, i32* %i, align 4
  %4 = load i32*, i32** %n.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 0
  %5 = load i32, i32* %arrayidx, align 4
  %cmp = icmp slt i32 %3, %5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %6 = load float*, float** %a.addr, align 8
  %arrayidx1 = getelementptr inbounds float, float* %6, i64 0
  %7 = load float, float* %arrayidx1, align 4
  %8 = load i32, i32* %i, align 4
  %idxprom = sext i32 %8 to i64
  %9 = load float*, float** %x.addr, align 8
  %arrayidx2 = getelementptr inbounds float, float* %9, i64 %idxprom
  %10 = load float, float* %arrayidx2, align 4
  %mul3 = fmul float %7, %10
  %11 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %11 to i64
  %12 = load float*, float** %y.addr, align 8
  %arrayidx5 = getelementptr inbounds float, float* %12, i64 %idxprom4
  %13 = load float, float* %arrayidx5, align 4
  %add6 = fadd float %mul3, %13
  %14 = load i32, i32* %i, align 4
  %idxprom7 = sext i32 %14 to i64
  %15 = load float*, float** %y.addr, align 8
  %arrayidx8 = getelementptr inbounds float, float* %15, i64 %idxprom7
  store float %add6, float* %arrayidx8, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind
define void @cuda_saxpy_s(i32* %n, float* %a, float* %x, float* %y) #2 {
entry:
  %n.addr = alloca i32*, align 8
  %a.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %i = alloca i32, align 4
  store i32* %n, i32** %n.addr, align 8
  store float* %a, float** %a.addr, align 8
  store float* %x, float** %x.addr, align 8
  store float* %y, float** %y.addr, align 8
  %0 = load i32, i32* getelementptr inbounds (%struct.uint3, %struct.uint3* addrspacecast (%struct.uint3 addrspace(1)* @blockIdx to %struct.uint3*), i32 0, i32 0), align 4
  %1 = load i32, i32* getelementptr inbounds (%struct.dim3, %struct.dim3* addrspacecast (%struct.dim3 addrspace(1)* @blockDim to %struct.dim3*), i32 0, i32 0), align 4
  %mul = mul i32 %0, %1
  %2 = load i32, i32* getelementptr inbounds (%struct.uint3, %struct.uint3* addrspacecast (%struct.uint3 addrspace(1)* @threadIdx to %struct.uint3*), i32 0, i32 0), align 4
  %add = add i32 %mul, %2
  store i32 %add, i32* %i, align 4
  call void @llvm.cuda.syncthreads()
  %3 = load i32, i32* %i, align 4
  %4 = load i32*, i32** %n.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 0
  %5 = load i32, i32* %arrayidx, align 4
  %cmp = icmp slt i32 %3, %5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %6 = load float*, float** %a.addr, align 8
  %arrayidx1 = getelementptr inbounds float, float* %6, i64 0
  %7 = load float, float* %arrayidx1, align 4
  %8 = load i32, i32* %i, align 4
  %idxprom = sext i32 %8 to i64
  %9 = load float*, float** %x.addr, align 8
  %arrayidx2 = getelementptr inbounds float, float* %9, i64 %idxprom
  %10 = load float, float* %arrayidx2, align 4
  %mul3 = fmul float %7, %10
  %11 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %11 to i64
  %12 = load float*, float** %y.addr, align 8
  %arrayidx5 = getelementptr inbounds float, float* %12, i64 %idxprom4
  %13 = load float, float* %arrayidx5, align 4
  %add6 = fadd float %mul3, %13
  %14 = load i32, i32* %i, align 4
  %idxprom7 = sext i32 %14 to i64
  %15 = load float*, float** %y.addr, align 8
  %arrayidx8 = getelementptr inbounds float, float* %15, i64 %idxprom7
  store float %add6, float* %arrayidx8, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind
declare void @llvm.cuda.syncthreads() #3

attributes #0 = { alwaysinline nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!nvvm.annotations = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{void (i32*, float*, float*, float*)* @cuda_saxpy, !"kernel", i32 1}
!1 = !{void (i32*, float*, float*, float*)* @cuda_saxpy_s, !"kernel", i32 1}
!2 = !{!"clang version xla-trunk (trunk r203011)"}
