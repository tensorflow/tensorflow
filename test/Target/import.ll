; RUN: mlir-translate -import-llvm %s | FileCheck %s

%struct.t = type {}
%struct.s = type { %struct.t, i64 }

; CHECK: llvm.mlir.global @g1() : !llvm<"{ {}, i64 }">
@g1 = external global %struct.s, align 8
; CHECK: llvm.mlir.global @g2() : !llvm.double
@g2 = external global double, align 8
; CHECK: llvm.mlir.global @g3("string")
@g3 = internal global [6 x i8] c"string"

; CHECK: llvm.mlir.global @g5() : !llvm<"<8 x i32>">
@g5 = external global <8 x i32>

@g4 = external global i32, align 8
; CHECK: llvm.mlir.global constant @int_gep() : !llvm<"i32*"> {
; CHECK-DAG:   %[[addr:[0-9]+]] = llvm.mlir.addressof @g4 : !llvm<"i32*">
; CHECK-DAG:   %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : !llvm.i32
; CHECK-NEXT:  %[[gepinit:[0-9]+]] = llvm.getelementptr %[[addr]][%[[c2]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
; CHECK-NEXT:  llvm.return %[[gepinit]] : !llvm<"i32*">
; CHECK-NEXT: }
@int_gep = internal constant i32* getelementptr (i32, i32* @g4, i32 2)

; CHECK: llvm.func @fe(!llvm.i32) -> !llvm.float
declare float @fe(i32)

; FIXME: function attributes.
; CHECK-LABEL: llvm.func @f1(%arg0: !llvm.i64) -> !llvm.i32 {
; CHECK-DAG: %[[c2:[0-9]+]] = llvm.mlir.constant(2 : i32) : !llvm.i32
; CHECK-DAG: %[[c42:[0-9]+]] = llvm.mlir.constant(42 : i32) : !llvm.i32
; CHECK-DAG: %[[c1:[0-9]+]] = llvm.mlir.constant(1 : i1) : !llvm.i1
; CHECK-DAG: %[[c43:[0-9]+]] = llvm.mlir.constant(43 : i32) : !llvm.i32
define internal dso_local i32 @f1(i64 %a) norecurse {
entry:
; CHECK: %{{[0-9]+}} = llvm.inttoptr %arg0 : !llvm.i64 to !llvm<"i64*">
  %aa = inttoptr i64 %a to i64*
; CHECK: %[[addrof:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm<"double*">
; CHECK: %{{[0-9]+}} = llvm.ptrtoint %[[addrof]] : !llvm<"double*"> to !llvm.i64
  %bb = ptrtoint double* @g2 to i64
; CHECK-DAG: %[[addrof2:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm<"double*">
; CHECK: %{{[0-9]+}} = llvm.getelementptr %[[addrof2]][%[[c2]]] : (!llvm<"double*">, !llvm.i32) -> !llvm<"double*">
  %cc = getelementptr double, double* @g2, i32 2
; CHECK: %[[b:[0-9]+]] = llvm.trunc %arg0 : !llvm.i64 to !llvm.i32
  %b = trunc i64 %a to i32
; CHECK: %[[c:[0-9]+]] = llvm.call @fe(%[[b]]) : (!llvm.i32) -> !llvm.float
  %c = call float @fe(i32 %b)
; CHECK: %[[d:[0-9]+]] = llvm.fptosi %[[c]] : !llvm.float to !llvm.i32
  %d = fptosi float %c to i32
; FIXME: icmp should return i1.
; CHECK: %[[e:[0-9]+]] = llvm.icmp "ne" %[[d]], %[[c2]] : !llvm.i32
  %e = icmp ne i32 %d, 2
; CHECK: llvm.cond_br %[[e]], ^bb1, ^bb2
  br i1 %e, label %if.then, label %if.end

; CHECK: ^bb1:
if.then:
; CHECK: llvm.return %[[c42]] : !llvm.i32
  ret i32 42
  
; CHECK: ^bb2:
if.end:
; CHECK: %[[orcond:[0-9]+]] = llvm.or %[[e]], %[[c1]] : !llvm.i1
  %or.cond = or i1 %e, 1
; CHECK: llvm.return %[[c43]]
  ret i32 43
}

; Test that instructions that dominate can be out of sequential order.
; CHECK-LABEL: llvm.func @f2(%arg0: !llvm.i64) -> !llvm.i64 {
; CHECK-DAG: %[[c3:[0-9]+]] = llvm.mlir.constant(3 : i64) : !llvm.i64
define i64 @f2(i64 %a) noduplicate {
entry:
; CHECK: llvm.br ^bb2
  br label %next

; CHECK: ^bb1:
end:
; CHECK: llvm.return %1
  ret i64 %b

; CHECK: ^bb2:
next:
; CHECK: %1 = llvm.add %arg0, %[[c3]] : !llvm.i64
  %b = add i64 %a, 3
; CHECK: llvm.br ^bb1
  br label %end
}

; Test arguments/phis.
; CHECK-LABEL: llvm.func @f2_phis(%arg0: !llvm.i64) -> !llvm.i64 {
; CHECK-DAG: %[[c3:[0-9]+]] = llvm.mlir.constant(3 : i64) : !llvm.i64
define i64 @f2_phis(i64 %a) noduplicate {
entry:
; CHECK: llvm.br ^bb2
  br label %next

; CHECK: ^bb1(%1: !llvm.i64):
end:
  %c = phi i64 [ %b, %next ]
; CHECK: llvm.return %1
  ret i64 %c

; CHECK: ^bb2:
next:
; CHECK: %2 = llvm.add %arg0, %[[c3]] : !llvm.i64
  %b = add i64 %a, 3
; CHECK: llvm.br ^bb1
  br label %end
}

; CHECK-LABEL: llvm.func @f3() -> !llvm<"i32*">
define i32* @f3() {
; CHECK: %[[c:[0-9]+]] = llvm.mlir.addressof @g2 : !llvm<"double*">
; CHECK: %[[b:[0-9]+]] = llvm.bitcast %[[c]] : !llvm<"double*"> to !llvm<"i32*">
; CHECK: llvm.return %[[b]] : !llvm<"i32*">
  ret i32* bitcast (double* @g2 to i32*)
}

; CHECK-LABEL: llvm.func @f4() -> !llvm<"i32*">
define i32* @f4() {
; CHECK: %[[b:[0-9]+]] = llvm.mlir.null : !llvm<"i32*">
; CHECK: llvm.return %[[b]] : !llvm<"i32*">
  ret i32* bitcast (double* null to i32*)
}

; CHECK-LABEL: llvm.func @f5
define void @f5(i32 %d) {
; FIXME: icmp should return i1.
; CHECK: = llvm.icmp "eq"
  %1 = icmp eq i32 %d, 2
; CHECK: = llvm.icmp "slt"
  %2 = icmp slt i32 %d, 2
; CHECK: = llvm.icmp "sle"
  %3 = icmp sle i32 %d, 2
; CHECK: = llvm.icmp "sgt"
  %4 = icmp sgt i32 %d, 2
; CHECK: = llvm.icmp "sge"
  %5 = icmp sge i32 %d, 2
; CHECK: = llvm.icmp "ult"
  %6 = icmp ult i32 %d, 2
; CHECK: = llvm.icmp "ule"
  %7 = icmp ule i32 %d, 2
; CHECK: = llvm.icmp "ugt"
  %8 = icmp ugt i32 %d, 2
  ret void
}

; CHECK-LABEL: llvm.func @f6(%arg0: !llvm<"void (i16)*">)
define void @f6(void (i16) *%fn) {
; CHECK: %[[c:[0-9]+]] = llvm.mlir.constant(0 : i16) : !llvm.i16
; CHECK: llvm.call %arg0(%[[c]])
  call void %fn(i16 0)
  ret void
}