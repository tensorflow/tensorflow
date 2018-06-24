; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - -check-parser-errors 2>&1 | FileCheck %s

; Check different error cases.
; TODO(jpienaar): This is checking the errors by simplify verifying the output.
; -----

; CHECK: expected type
; CHECK-NEXT: illegaltype
extfunc @illegaltype(i42)

; -----
; CHECK: expected type
; CHECK-NEXT: nestedtensor
extfunc @nestedtensor(tensor<tensor<i8>>) -> ()

; -----
; CHECK: expected '{' in CFG function
cfgfunc @foo()
cfgfunc @bar()

; -----
; CHECK: expected a function identifier like
; CHECK-NEXT: missingsigil
extfunc missingsigil() -> (i1, int, f32)


; -----

cfgfunc @bad_branch() {
bb42:
  br missing  ; CHECK: error: reference to an undefined basic block 'missing'
}

; -----

cfgfunc @block_redef() {
bb42:
  return
bb42:        ; CHECK: error: redefinition of block 'bb42'
  return
}
