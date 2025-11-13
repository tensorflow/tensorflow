; RUN: ir-compiler-opt %s | FileCheck %s

; Test that CppGenIntrinsics replace and vectorize as expected.
; This file is a simple loop that calls tanh n times, loading and storing
; each element individually from input and output float pointers.
; void simple_tanh_loop(float* in, float* out, int64_t n) 
;     for (int64_t i = 0; i < n; i++) {
;         out[i] = tanh_f32(in[i]);
; We expect this loop to be automatically vectorized by default by LLVM in 
; IrCompiler and for calls to xla.tanh.f32 to be inlined and replaced by 
; vectorized math tanh code.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare float @local_xla.tanh.f32(float)

; --- Main Function ---
; Defines a function that takes an input pointer (%in), output pointer (%out),
; and a loop count (%n).
; We use 'noalias' to promise LLVM that the input and output pointers
; do not overlap, which is a critical requirement for vectorization.
; 'readonly' and 'writeonly' provide even stronger guarantees.
define void @simple_tanh_loop(ptr noalias noundef readonly %in, ptr noalias noundef writeonly %out, i64 %n) {
entry:
  %cmp.end = icmp eq i64 %n, 0
  br i1 %cmp.end, label %loop.exit, label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]

  %cmp.loop = icmp slt i64 %iv, %n
  br i1 %cmp.loop, label %loop.body, label %loop.exit

loop.body:
  %in.gep = getelementptr inbounds float, ptr %in, i64 %iv
  %val = load float, ptr %in.gep, align 4
  %tanh.val = call fast float @local_xla.tanh.f32(float %val)
; check that all calls have been inlined
; CHECK-NOT: {{.*}}call{{.*}}xla.tanh
; Check that we see inline float vectorized here automatically - 4x on ARM, 8x on x86.
; CHECK: fm{{.*}}<{{[48]}} x float>{{.*}}0x3E
; CHECK-NOT: define{{.*}}xla.tanh
  %out.gep = getelementptr inbounds float, ptr %out, i64 %iv
  store float %tanh.val, ptr %out.gep, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  br label %loop.header

loop.exit:
  ret void
}