// RUN: mlir-opt %s -split-input-file -verify

// Check different error cases.
// -----

extfunc @illegaltype(i) // expected-error {{expected type}}

// -----

mlfunc @illegaltype() {
  %0 = constant splat<<vector 4 x f32>, 0> : vector<4 x f32> // expected-error {{expected type}}
}

// -----

extfunc @nestedtensor(tensor<tensor<i8>>) -> () // expected-error {{invalid tensor element type}}

// -----

extfunc @indexvector(vector<4 x index>) -> () // expected-error {{vector elements must be int or float type}}

// -----

extfunc @indexmemref(memref<? x index>) -> () // expected-error {{invalid memref element type}}

// -----

extfunc @indextensor(tensor<4 x index>) -> () // expected-error {{invalid tensor element type}}

// -----
// Test no map in memref type.
extfunc @memrefs(memref<2x4xi8, >) // expected-error {{expected list element}}

// -----
// Test non-existent map in memref type.
extfunc @memrefs(memref<2x4xi8, #map7>) // expected-error {{undefined affine map id 'map7'}}

// -----
// Test non hash identifier in memref type.
extfunc @memrefs(memref<2x4xi8, %map7>) // expected-error {{expected '(' at start of dimensional identifiers list}}

// -----
// Test non-existent map in map composition of memref type.
#map0 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, #map8>) // expected-error {{undefined affine map id 'map8'}}

// -----
// Test multiple memory space error.
#map0 = (d0, d1) -> (d0, d1)
extfunc @memrefs(memref<2x4xi8, #map0, 1, 2>) // expected-error {{multiple memory spaces specified in memref type}}

// -----
// Test affine map after memory space.
#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, 1, #map1>) // expected-error {{affine map after memory space in memref type}}

// -----
// Test dimension mismatch between memref and layout map.
// The error must be emitted even for the trivial identity layout maps that are
// dropped in type creation.
#map0 = (d0, d1) -> (d0, d1)
extfunc @memrefs(memref<42xi8, #map0>) // expected-error {{memref affine map dimension mismatch}}

// -----

#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0) -> (d0)
extfunc @memrefs(memref<42x42xi8, #map0, #map1>) // expected-error {{memref affine map dimension mismatch}}

// -----

extfunc @illegalattrs() -> () attributes { key } // expected-error {{expected ':' in attribute list}}

// -----

cfgfunc @foo()
cfgfunc @bar() // expected-error {{expected '{' in function}}

// -----

extfunc missingsigil() -> (i1, index, f32) // expected-error {{expected a function identifier like}}


// -----

cfgfunc @bad_branch() {
bb12:
  br missing  // expected-error {{reference to an undefined basic block}}
}

// -----

cfgfunc @block_redef() {
bb42:
  return
bb42:        // expected-error {{redefinition of block 'bb42'}}
  return
}

// -----

cfgfunc @no_terminator() {
bb40:
  return
bb41:
bb42:        // expected-error {{custom op 'bb42' is unknown}}
  return
}

// -----

cfgfunc @block_no_rparen() {
bb42 (%bb42 : i32: // expected-error {{expected ')' to end argument list}}
  return
}

// -----

cfgfunc @block_arg_no_ssaid() {
bb42 (i32): // expected-error {{expected SSA operand}}
  return
}

// -----

cfgfunc @block_arg_no_type() {
bb42 (%0): // expected-error {{expected ':' and type for SSA operand}}
  return
}

// -----

cfgfunc @block_arg_no_close_paren() {
bb42:
  br bb2( // expected-error@+1 {{expected ')' to close argument list}}
  return
}

// -----

cfgfunc @block_first_has_predecessor() {
// expected-error@-1 {{first block of cfgfunc must not have predecessors}}
bb42:
  br bb43
bb43:
  br bb42
}

// -----

cfgfunc @illegalattrs() -> ()
  attributes { key } { // expected-error {{expected ':' in attribute list}}
bb42:
  return
}

// -----

mlfunc @foo()
mlfunc @bar() // expected-error {{expected '{' in function}}

// -----

mlfunc @empty() {
} // expected-error {{function must have a body}}

// -----

mlfunc @illegalattrs() -> ()
  attributes { key } { // expected-error {{expected ':' in attribute list}}
bb42:
  return
}

// -----

mlfunc @no_return() {  // expected-error {{ML function must end with return instruction}}
  "foo"() : () -> ()
}

// -----

"       // expected-error {{expected}}
"

// -----

"       // expected-error {{expected}}

// -----

cfgfunc @bad_op_type() {
bb40:
  "foo"() : i32  // expected-error {{expected function type}}
  return
}
// -----

cfgfunc @no_terminator() {
bb40:
  "foo"() : ()->()
  ""() : ()->()  // expected-error {{empty operation name is invalid}}
  return
}

// -----

extfunc @illegaltype(i0) // expected-error {{invalid integer width}}

// -----

mlfunc @malformed_for_percent() {
  for i = 1 to 10 { // expected-error {{expected SSA identifier for the loop variable}}

// -----

mlfunc @malformed_for_equal() {
  for %i 1 to 10 { // expected-error {{expected '='}}

// -----

mlfunc @malformed_for_to() {
  for %i = 1 too 10 { // expected-error {{expected 'to' between bounds}}
  }
}

// -----

mlfunc @incomplete_for() {
  for %i = 1 to 10 step 2
}        // expected-error {{expected '{' before instruction list}}

// -----

mlfunc @nonconstant_step(%1 : i32) {
  for %2 = 1 to 5 step %1 { // expected-error {{expected integer}}

// -----

mlfunc @for_negative_stride() {
  for %i = 1 to 10 step -1
}        // expected-error {{step has to be a positive integer}}

// -----

mlfunc @non_instruction() {
  asd   // expected-error {{custom op 'asd' is unknown}}
}

// -----

mlfunc @invalid_if_conditional1() {
  for %i = 1 to 10 {
    if () { // expected-error {{expected ':' or '['}}
  }
}

// -----

mlfunc @invalid_if_conditional2() {
  for %i = 1 to 10 {
    if (i)[N] : (i >= )  // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

mlfunc @invalid_if_conditional3() {
  for %i = 1 to 10 {
    if (i)[N] : (i == 1) // expected-error {{expected '0' after '=='}}
  }
}

// -----

mlfunc @invalid_if_conditional4() {
  for %i = 1 to 10 {
    if (i)[N] : (i >= 2) // expected-error {{expected '0' after '>='}}
  }
}

// -----

mlfunc @invalid_if_conditional5() {
  for %i = 1 to 10 {
    if (i)[N] : (i <= 0 ) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

mlfunc @invalid_if_conditional6() {
  for %i = 1 to 10 {
    if (i) : (i) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----
// TODO (support if (1)?
mlfunc @invalid_if_conditional7() {
  for %i = 1 to 10 {
    if (i) : (1) // expected-error {{expected '== 0' or '>= 0' at end of affine constraint}}
  }
}

// -----

#map = (d0) -> (%  // expected-error {{invalid SSA name}}

// -----

cfgfunc @test() {
bb40:
  %1 = "foo"() : (i32)->i64 // expected-error {{expected 0 operand types but had 1}}
  return
}

// -----

cfgfunc @redef() {
bb42:
  %x = "xxx"(){index: 0} : ()->i32 // expected-error {{previously defined here}}
  %x = "xxx"(){index: 0} : ()->i32 // expected-error {{redefinition of SSA value '%x'}}
  return
}

// -----

cfgfunc @undef() {
bb42:
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value}}
  return
}

// -----

mlfunc @missing_rbrace() {
  return
mlfunc @d() {return} // expected-error {{expected '}' after instruction list}}

// -----

mlfunc @malformed_type(%a : intt) { // expected-error {{expected type}}
}

// -----

cfgfunc @resulterror() -> i32 {
bb42:
  return    // expected-error {{'return' op has 0 operands, but enclosing function returns 1}}
}

// -----

mlfunc @mlfunc_resulterror() -> i32 {
  return // expected-error {{'return' op has 0 operands, but enclosing function returns 1}}
}

// -----

cfgfunc @argError() {
bb1(%a: i64):  // expected-error {{previously defined here}}
  br bb2
bb2(%a: i64):  // expected-error{{redefinition of SSA value '%a'}}
  return
}

// -----

cfgfunc @bbargMismatch(i32, f32) { // expected-error {{first block of cfgfunc must have 2 arguments to match function signature}}
bb42(%0: f32):
  return
}

// -----

cfgfunc @br_mismatch() {
bb0:
  %0 = "foo"() : () -> (i1, i17)
  // expected-error @+1 {{branch has 2 operands, but target block has 1}}
  br bb1(%0#1, %0#0 : i17, i1)

bb1(%x: i17):
  return
}

// -----

// Test no nested vector.
extfunc @vectors(vector<1 x vector<1xi32>>, vector<2x4xf32>)
// expected-error@-1 {{vector elements must be int or float type}}

// -----

cfgfunc @condbr_notbool() {
bb0:
  %a = "foo"() : () -> i32 // expected-error {{prior use here}}
  cond_br %a, bb0, bb0 // expected-error {{use of value '%a' expects different type than prior uses}}
// expected-error@-1 {{expected condition type was boolean (i1)}}
}

// -----

cfgfunc @condbr_badtype() {
bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cond_br %c, bb0(%a, %a : i32, bb0) // expected-error {{expected type}}
}

// -----

cfgfunc @condbr_a_bb_is_not_a_type() {
bb0:
  %c = "foo"() : () -> i1
  %a = "foo"() : () -> i32
  cond_br %c, bb0(%a, %a : i32, i32), i32 // expected-error {{expected basic block name}}
}

// -----

cfgfunc @undef() {
bb0:
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undeclared SSA value name}}
  return
}

// -----

mlfunc @undef() {
  %x = "xxx"(%y) : (i32)->i32   // expected-error {{use of undefined SSA value %y}}
  return
}

// -----

mlfunc @duplicate_induction_var() {
  for %i = 1 to 10 {   // expected-error {{previously defined here}}
    for %i = 1 to 10 { // expected-error {{redefinition of SSA value '%i'}}
    }
  }
  return
}

// -----

mlfunc @dominance_failure() {
  for %i = 1 to 10 {
  }
  "xxx"(%i) : (index)->()   // expected-error {{operand #0 does not dominate this use}}
  return
}

// -----

cfgfunc @dominance_failure() {
bb0:
  "foo"(%x) : (i32) -> ()    // expected-error {{operand #0 does not dominate this use}}
  br bb1
bb1:
  %x = "bar"() : () -> i32    // expected-note {{operand defined here}}
  return
}

// -----

mlfunc @return_type_mismatch() -> i32 {
  %0 = "foo"() : ()->f32
  return %0 : f32  // expected-error {{type of return operand 0 doesn't match function result type}}
}

// -----

mlfunc @return_inside_loop() -> i8 {
  for %i = 1 to 100 {
    %a = "foo"() : ()->i8
    return %a : i8
    // expected-error@-1 {{'return' op must be the last instruction in the ML function}}
  }
}

// -----

extfunc @redef()
extfunc @redef()  // expected-error {{redefinition of function named 'redef'}}

// -----

cfgfunc @foo() {
bb0:
  %x = constant @foo : (i32) -> ()  // expected-error {{reference to function with mismatched type}}
  return
}

// -----

cfgfunc @undefined_function() {
bb0:
  %x = constant @bar : (i32) -> ()  // expected-error {{reference to undefined function 'bar'}}
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @bound_symbol_mismatch(%N : index) {
  for %i = #map1(%N) to 100 {
  // expected-error@-1 {{symbol operand count and affine map symbol count must match}}
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @bound_dim_mismatch(%N : index) {
  for %i = #map1(%N, %N)[%N] to 100 {
  // expected-error@-1 {{dim operand count and affine map dim count must match}}
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @invalid_dim_nested(%N : index) {
  for %i = 1 to 100 {
    %a = "foo"(%N) : (index)->(index)
    for %j = 1 to #map1(%a)[%i] {
    // expected-error@-1 {{value '%a' cannot be used as a dimension id}}
    }
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @invalid_dim_affine_apply(%N : index) {
  for %i = 1 to 100 {
    %a = "foo"(%N) : (index)->(index)
    %w = affine_apply (i)->(i+1) (%a)
    for %j = 1 to #map1(%w)[%i] {
    // expected-error@-1 {{value '%w' cannot be used as a dimension id}}
    }
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @invalid_symbol_iv(%N : index) {
  for %i = 1 to 100 {
    %a = "foo"(%N) : (index)->(index)
    for %j = 1 to #map1(%N)[%i] {
    // expected-error@-1 {{value '%i' cannot be used as a symbol}}
    }
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @invalid_symbol_nested(%N : index) {
  for %i = 1 to 100 {
    %a = "foo"(%N) : (index)->(index)
    for %j = 1 to #map1(%N)[%a] {
    // expected-error@-1 {{value '%a' cannot be used as a symbol}}
    }
  }
  return
}

// -----

#map1 = (i)[j] -> (i+j)

mlfunc @invalid_symbol_affine_apply(%N : index) {
  for %i = 1 to 100 {
    %w = affine_apply (i)->(i+1) (%i)
    for %j = 1 to #map1(%i)[%w] {
    // expected-error@-1 {{value '%w' cannot be used as a symbol}}
    }
  }
  return
}

// -----

mlfunc @large_bound() {
  for %i = 1 to 9223372036854775810 {
  // expected-error@-1 {{bound or step is too large for index}}
  }
  return
}

// -----

mlfunc @max_in_upper_bound(%N : index) {
  for %i = 1 to max (i)->(N, 100) { //expected-error {{expected SSA operand}}
  }
  return
}

// -----

mlfunc @step_typo() {
  for %i = 1 to 100 step -- 1 { //expected-error {{expected integer}}
  }
  return
}

// -----

mlfunc @invalid_bound_map(%N : i32) {
  for %i = 1 to (i)->(j)(%N) { //expected-error {{use of undeclared identifier}}
  }
  return
}

// -----
#set0 = (i)[N] : (i >= 0, N - i >= 0)

mlfunc @invalid_if_operands1(%N : index) {
  for %i = 1 to 10 {
    if #set0(%i) {
    // expected-error@-1 {{symbol operand count and integer set symbol count must match}}

// -----
#set0 = (i)[N] : (i >= 0, N - i >= 0)

mlfunc @invalid_if_operands2(%N : index) {
  for %i = 1 to 10 {
    if #set0()[%N] {
    // expected-error@-1 {{dim operand count and integer set dim count must match}}

// -----
#set0 = (i)[N] : (i >= 0, N - i >= 0)

mlfunc @invalid_if_operands3(%N : index) {
  for %i = 1 to 10 {
    if #set0(%i)[%i] {
    // expected-error@-1 {{value '%i' cannot be used as a symbol}}

// -----
// expected-error@+1 {{expected '"' in string literal}}
"J// -----
mlfunc @calls(%arg0 : i32) {
  // expected-error@+1 {{expected type}}
  %z = "casdasda"(%x) : (ppop32) -> i32
}
// -----
// expected-error@+2 {{expected SSA operand}}
cfgfunc@n(){b(
// -----

cfgfunc @elementsattr_non_tensor_type() -> () {
bb0:
  "foo"(){bar: dense<i32, [4]>} : () -> () // expected-error {{expected elements literal has a tensor or vector type}}
}

// -----

cfgfunc @elementsattr_non_ranked() -> () {
bb0:
  "foo"(){bar: dense<tensor<?xi32>, [4]>} : () -> () // expected-error {{tensor literals must be ranked and have static shape}}
}

// -----

cfgfunc @elementsattr_shape_mismatch() -> () {
bb0:
  "foo"(){bar: dense<tensor<5xi32>, [4]>} : () -> () // expected-error {{inferred shape of elements literal ([1]) does not match type ([5])}}
}

// -----

cfgfunc @elementsattr_invalid() -> () {
bb0:
  "foo"(){bar: dense<tensor<2xi32>, [4, [5]]>} : () -> () // expected-error {{tensor literal is invalid; ranks are not consistent between elements}}
}

// -----

cfgfunc @elementsattr_badtoken() -> () {
bb0:
  "foo"(){bar: dense<tensor<1xi32>, [tf_opaque]>} : () -> () // expected-error {{expected '[' or scalar constant inside tensor literal}}
}

// -----

cfgfunc @elementsattr_floattype1() -> () {
bb0:
  // expected-error@+1 {{floating point value not valid for specified type}}
  "foo"(){bar: dense<tensor<1xi32>, [4.0]>} : () -> ()
}

// -----

cfgfunc @elementsattr_floattype2() -> () {
bb0:
  // expected-error@+1 {{integer value not valid for specified type}}
  "foo"(){bar: dense<tensor<1xf32>, [4]>} : () -> ()
}

// -----

cfgfunc @elementsattr_toolarge1() -> () {
bb0:
  "foo"(){bar: dense<tensor<1xi8>, [777]>} : () -> () // expected-error {{integer constant out of range for attribute}}
}

// -----

cfgfunc @elementsattr_toolarge2() -> () {
bb0:
  "foo"(){bar: dense<tensor<1xi8>, [-777]>} : () -> () // expected-error {{integer constant out of range for attribute}}
}

// -----

cfgfunc @elementsattr_malformed_opaque() -> () {
bb0:
  "foo"(){bar: opaque<tensor<1xi8>, "0xQZz123">} : () -> () // expected-error {{opaque string only contains hex digits}}
}

// -----

cfgfunc @elementsattr_malformed_opaque1() -> () {
bb0:
  "foo"(){bar: opaque<tensor<1xi8>, "00abc">} : () -> () // expected-error {{opaque string should start with '0x'}}
}
