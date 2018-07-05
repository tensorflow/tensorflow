;
; RUN: %S/../../mlir-opt %s -o - -check-parser-errors

; Check different error cases.
; -----

#hello_world1 = (i, j) -> ((), j) ; expected-error {{no expression inside parentheses}}

; -----
#hello_world2 (i, j) [s0] -> (i, j) ; expected-error {{expected '=' in affine map outlined definition}}

; -----
#hello_world3a = (i, j) [s0] -> (2*i*, 3*j*i*2 + 5) ; expected-error {{missing right operand of multiply op}}

; -----
#hello_world3b = (i, j) [s0] -> (i+, i+j+2 + 5) ; expected-error {{missing right operand of add op}}

; -----
#hello_world4 = (i, j) [s0] -> ((s0 + i, j) ; expected-error {{expected ')'}}

; -----
#hello_world5 = (i, j) [s0] -> ((s0 + i, j) ; expected-error {{expected ')'}}

; -----
#hello_world6 = (i, j) [s0] -> (((s0 + (i + j) + 5), j) ; expected-error {{expected ')'}}

; -----
#hello_world8 = (i, j) [s0] -> i + s0, j) ; expected-error {{expected '(' at start of affine map range}}

; -----
#hello_world9 = (i, j) [s0] -> (x) ; expected-error {{identifier is neither dimensional nor symbolic}}

; -----
#hello_world10 = (i, j, i) [s0] -> (i) ; expected-error {{dimensional identifier name reused}}

; -----
#hello_world11 = (i, j) [s0, s1, s0] -> (i) ; expected-error {{symbolic identifier name reused}}

; -----
#hello_world12 = (i, j) [i, s0] -> (j) ; expected-error {{dimensional identifier name reused}}

; -----
#hello_world13 = (i, j) [s0, s1] -> () ; expected-error {{expected list element}}

; -----
#hello_world14 = (i, j) [s0, s1] -> (+i, j) ; expected-error {{left operand of binary op missing}}

; -----
#hello_world15 = (i, j) [s0, s1] -> (i, *j+5) ; expected-error {{left operand of binary op missing}}

; FIXME(bondhugula) This one leads to two errors: the first on identifier being
; neither dimensional nor symbolic and then the right operand missing.
;-----
; #hello_world22 = (i, j) -> (i, 3*d0 + j)

; TODO(bondhugula): Add more tests; coverage of error messages emitted not complete


; -----

#ABC = (i,j) -> (i+j)
#ABC = (i,j) -> (i+j)  // expected-error {{redefinition of affine map id 'ABC'}}

