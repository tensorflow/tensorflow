; RUN: ir-compiler-opt %s | FileCheck %s

; Test that the IR compiler's vectorization transforms simple loops

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Simple loop that should be vectorized - processes arrays element by element
define void @simple_add_loop(ptr %a, ptr %b, ptr %c, i32 %n) {
; CHECK: define void @simple_add_loop
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.preheader, label %exit

loop.preheader:
  br label %loop

loop:
  %i = phi i32 [ 0, %loop.preheader ], [ %i.next, %loop ]
  %a.gep = getelementptr float, ptr %a, i32 %i
  %b.gep = getelementptr float, ptr %b, i32 %i
  %c.gep = getelementptr float, ptr %c, i32 %i
  %a.load = load float, ptr %a.gep, align 4
  %b.load = load float, ptr %b.gep, align 4
  %sum = fadd float %a.load, %b.load
  store float %sum, ptr %c.gep, align 4
  %i.next = add nsw i32 %i, 1
  %exit.cond = icmp eq i32 %i.next, %n
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

; Check for vectorized version - should have vector loads, operations, and stores
; CHECK: vector.body:
; CHECK: load <{{[0-9]+}} x float>, ptr
; CHECK: load <{{[0-9]+}} x float>, ptr
; CHECK: fadd <{{[0-9]+}} x float>
; CHECK: store <{{[0-9]+}} x float>

; Also check that original scalar loop still exists (for remainder iterations)
; CHECK: load float, ptr
; CHECK: fadd float
; CHECK: store float