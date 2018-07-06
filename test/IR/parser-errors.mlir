; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - -check-parser-errors

; Check different error cases.
; -----

extfunc @illegaltype(i) ; expected-error {{expected type}}

; -----

extfunc @nestedtensor(tensor<tensor<i8>>) -> () ; expected-error {{expected type}}

; -----

cfgfunc @foo()
cfgfunc @bar() ; expected-error {{expected '{' in CFG function}}

; -----

extfunc missingsigil() -> (i1, affineint, f32) ; expected-error {{expected a function identifier like}}


; -----

cfgfunc @bad_branch() {
bb42:
  br missing  ; expected-error {{reference to an undefined basic block 'missing'}}
}

; -----

cfgfunc @block_redef() {
bb42:
  return
bb42:        ; expected-error {{redefinition of block 'bb42'}}
  return
}

; -----

cfgfunc @no_terminator() {
bb40:
  return
bb41:
bb42:        ; expected-error {{expected operation name}}
  return
}

; -----

mlfunc @foo()
mlfunc @bar() ; expected-error {{expected '{' in ML function}}

; -----

mlfunc @no_return() {
}        ; expected-error {{ML function must end with return statement}}

; -----

"       ; expected-error {{expected}}
"

; -----

"       ; expected-error {{expected}}

; -----

cfgfunc @no_terminator() {
bb40:
  "foo"()
  ""()   ; expected-error {{empty operation name is invalid}}
  return
}

; -----

extfunc @illegaltype(i0) ; expected-error {{invalid integer width}}

; -----

mlfunc @incomplete_for() {
  for
}        ; expected-error {{expected '{' before statement list}}

; -----

mlfunc @non_statement() {
  asd   ; expected-error {{expected statement}}
}

; -----

cfgfunc @malformed_dim() {
bb42:
  "dim"(){index: "xyz"}  ; expected-error {{'dim' op requires an integer attribute named 'index'}}
  return
}

; -----
