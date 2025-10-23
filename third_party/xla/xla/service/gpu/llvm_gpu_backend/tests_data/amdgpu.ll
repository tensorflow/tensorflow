; ModuleID = 'example.hip'
source_filename = "example.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_dbc6fc0be16cd677 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_dbc6fc0be16cd677 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(read, argmem: readwrite)
define protected amdgpu_kernel void @_Z3fooPfS_i(ptr addrspace(1) noundef writeonly captures(none) %dst.coerce, ptr addrspace(1) noundef readonly captures(none) %src.coerce, i32 noundef %limit) local_unnamed_addr #0 {
entry:
  %call.i12 = tail call i64 @__ockl_get_group_id(i32 noundef 0) #3
  %conv.i = trunc i64 %call.i12 to i32
  %call.i = tail call i64 @__ockl_get_local_size(i32 noundef 0) #3
  %conv.i13 = trunc i64 %call.i to i32
  %mul = mul i32 %conv.i13, %conv.i
  %call.i14 = tail call i64 @__ockl_get_local_id(i32 noundef 0) #3
  %conv.i15 = trunc i64 %call.i14 to i32
  %add = add i32 %mul, %conv.i15
  %cmp.not = icmp slt i32 %add, %limit
  br i1 %cmp.not, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64
  %arrayidx7 = getelementptr inbounds float, ptr addrspace(1) %dst.coerce, i64 %idxprom
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %src.coerce, i64 %idxprom
  %0 = load float, ptr addrspace(1) %arrayidx, align 4, !tbaa !5
  %call5 = tail call contract float @__ocml_exp_f32(float noundef %0) #4
  store float %call5, ptr addrspace(1) %arrayidx7, align 4, !tbaa !5
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare hidden float @__ocml_exp_f32(float noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_group_id(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_local_size(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare hidden i64 @__ockl_get_local_id(i32 noundef) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn memory(read, argmem: readwrite) "amdgpu-flat-work-group-size"="1,1024" "amdgpu-waves-per-eu"="8,16" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1200" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1200" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1200" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f32,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot10-insts,+dot11-insts,+dot12-insts,+dot7-insts,+dot8-insts,+dot9-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-insts,+gfx8-insts,+gfx9-insts,+wavefrontsize32" "uniform-work-group-size"="false" }
attributes #3 = { convergent nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{!"clang version 22.0.0git"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
