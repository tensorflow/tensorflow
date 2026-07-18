; RUN: ir-compiler-opt %s | FileCheck %s
; Checks that when we emit xla.exp, it gets inlined and vectorized.

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @func(double* %0, double* %1, i32 %2) local_unnamed_addr #0 {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %7

5:
  %6 = zext nneg i32 %2 to i64
  br label %8

7:
  ret void

8:
  %9 = phi i64 [ 0, %5 ], [ %15, %8 ]
  %10 = getelementptr inbounds nuw double, ptr %1, i64 %9
  %11 = load double, ptr %10, align 8
  %13 = tail call double @xla.exp.f64(double %11) #1
  %14 = getelementptr inbounds nuw double, ptr %0, i64 %9
  store double %13, ptr %14, align 8
  %15 = add nuw nsw i64 %9, 1
  %16 = icmp eq i64 %15, %6
  br i1 %16, label %7, label %8
}

; Check that we have vectorized and inlined the call to ldexp:
; CHECK: vector.body:
; Check that the call to ldexp is inlined:
; CHECK: %b{{.*}} = ashr <{{[0-9]+}} x i64> {{.*}}splat (i64 2)
; CHECK-NOT: {{.*}}call{{.*}}ldexp
; vectorized minimum:
; CHECK call <{{[0-9]+}} x double> @llvm.minimum.v4f64{{.+}}0x40862E42FEFA39EF
; CHECK-NOT: {{.*}}call{{.*}}ldexp

; Check that the loop epilogue does an unvectorized minimum.
; CHECK: scalar.ph:
; CHECK: call double @llvm.minimum.f64{{.+}}0x40862E42FEFA39EF

declare double @xla.exp.f64(double) #1

attributes #0 = {  mustprogress nofree norecurse nounwind memory(argmem: readwrite) uwtable }
attributes #1 = {  mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = {  nounwind }