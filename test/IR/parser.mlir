; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - | FileCheck %s


; CHECK: extfunc @foo()
extfunc @foo()

; CHECK: extfunc @bar()
extfunc @bar()

; CHECK: extfunc @baz()
extfunc @baz()

