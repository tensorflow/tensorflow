; RUN: triton-llvm-opt -break-struct-phi-nodes %s | FileCheck %s

; CHECK-LABEL: struct
define {i32, i32} @struct(i1 %c) {
; CHECK: br i1 %{{.*}}, label [[TRUE:%.*]], label [[FALSE:%.*]]
  br i1 %c, label %true, label %false

true:
  %s.1 = insertvalue {i32, i32} undef, i32 20, 0
  %s.2 = insertvalue {i32, i32} %s.1, i32 200, 1

; CHECK-DAG: [[E0:%.*]] = extractvalue { i32, i32 } %{{.*}}, 0
; CHECK-DAG: [[E1:%.*]] = extractvalue { i32, i32 } %{{.*}}, 1
; CHECK: br
  br label %exit

false:
  %s.3 = insertvalue {i32, i32} undef, i32 30, 0
  %s.4 = insertvalue {i32, i32} %s.3, i32 300, 1
; CHECK-DAG: [[E2:%.*]] = extractvalue { i32, i32 } %{{.*}}, 0
; CHECK-DAG: [[E3:%.*]] = extractvalue { i32, i32 } %{{.*}}, 1
; CHECK: br
  br label %exit

exit:
; CHECK-DAG: [[PHI0:%.*]] = phi i32 [ [[E0]], [[TRUE]] ], [ [[E2]], [[FALSE]] ]
; CHECK-DAG: [[PHI1:%.*]] = phi i32 [ [[E1]], [[TRUE]] ], [ [[E3]], [[FALSE]] ]
; CHECK: [[S0:%.*]] = insertvalue { i32, i32 } undef, i32 [[PHI0]], 0
; CHECK: [[S1:%.*]] = insertvalue { i32, i32 } [[S0]], i32 [[PHI1]], 1
; CHECK: ret { i32, i32 } [[S1]]
  %r = phi {i32, i32} [ %s.2, %true], [ %s.4, %false ]
  ret {i32, i32} %r
}
