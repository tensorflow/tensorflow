// RUN: mlir-opt %s -split-input-file -verify

// Check different error cases.
// -----
#hello_world = (i, j) -> ((), j) // expected-error {{no expression inside parentheses}}

// -----
#hello_world = (i, j) -> (->, j) // expected-error {{expected affine expression}}

// -----
#hello_world = (i, j) -> (:) // expected-error {{expected affine expression}}

// -----
#hello_world = (i, j) -> (, j) // expected-error {{expected affine expression}}

// -----
#hello_world (i, j) [s0] -> (i, j) // expected-error {{expected '=' in attribute alias definition}}

// -----
#hello_world = (i, j) [s0] -> (2*i*, 3*j*i*2 + 5) // expected-error {{missing right operand of binary op}}

// -----
#hello_world = (i, j) [s0] -> (i+, i+j+2 + 5) // expected-error {{missing right operand of binary op}}

// -----
#hello_world = (i, j) [s0] -> ((s0 + i, j) // expected-error {{expected ')'}}

// -----
#hello_world = (i, j) [s0] -> (((s0 + (i + j) + 5), j) // expected-error {{expected ')'}}

// -----
#hello_world = (i, j) [s0] -> i + s0, j) // expected-error {{expected '(' at start of affine map range}}

// -----
#hello_world = (i, j) [s0] -> (x) // expected-error {{use of undeclared identifier}}

// -----
#hello_world = (i, j, i) [s0] -> (i) // expected-error {{redefinition of identifier 'i'}}

// -----
#hello_world = (i, j) [s0, s1, s0] -> (i) // expected-error {{redefinition of identifier 's0'}}

// -----
#hello_world = (i, j) [i, s0] -> (j) // expected-error {{redefinition of identifier 'i'}}

// -----
#hello_world = (i, j) [s0, s1] -> () // expected-error {{expected list element}}

// -----
#hello_world = (i, j) [s0, s1] -> (+i, j) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, *j) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (floordiv i 2, j) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (ceildiv i 2, j) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (mod i 2, j) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (-(), j)
// expected-error@-1 {{no expression inside parentheses}}
// expected-error@-2 {{missing operand of negation}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, *j+5) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, floordiv j+5) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, ceildiv j+5) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, mod j+5) // expected-error {{missing left operand of binary op}}

// -----
#hello_world = (i, j) [s0, s1] -> (i*j, j) // expected-error {{non-affine expression: at least one of the multiply operands has to be either a constant or symbolic}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, j + j ceildiv 128 mod 16 * i - 4) // expected-error {{non-affine expression: at least one of the multiply operands has to be either a constant or symbolic}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, j floordiv i) // expected-error {{non-affine expression: right operand of floordiv has to be either a constant or symbolic}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, i*2 ceildiv j*5) // expected-error {{non-affine expression: right operand of ceildiv has to be either a constant or symbolic}}

// -----
#hello_world = (i, j) [s0, s1] -> (i, i mod (2+i)) // expected-error {{non-affine expression: right operand of mod has to be either a constant or symbolic}}

// -----
#hello_world = (i, j) [s0, s1] -> (-1*i j, j) // expected-error {{expected ',' or ')'}}

// -----
#hello_world = (i, j) -> (i, 3*d0 + ) // expected-error {{use of undeclared identifier}}

// -----
#hello_world = (i, j) -> (i, j) size (10, x) // expected-error {{use of undeclared identifier}}

// -----
#hello_world = (i, j) [M] -> (i, j) size (10, j) // expected-error {{size expressions cannot refer to dimension values}}

// -----
#hello_world = (i, j) [M] -> (i, j) size (10, M+i) // expected-error {{size expressions cannot refer to dimension values}}

// -----
#hello_world = (i, j) -> (i, j) size (10) // expected-error {{fewer range sizes than range expressions}}

// -----
#hello_world = (i, j) -> (i, j) size (10, 20, 30) // expected-error {{more range sizes than range expressions}}

// TODO(bondhugula): Add more tests; coverage of error messages emitted not complete

// -----
#ABC = (i,j) -> (i+j)
#ABC = (i,j) -> (i+j)  // expected-error {{redefinition of attribute alias id 'ABC'}}
