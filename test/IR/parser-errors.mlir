// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt %s -o - -check-parser-errors

// Check different error cases.
// -----

extfunc @illegaltype(i) // expected-error {{expected type}}

// -----

extfunc @nestedtensor(tensor<tensor<i8>>) -> () // expected-error {{expected type}}

// -----
// Test no comma in memref type.
// TODO(andydavis) Fix this test if we decide to allow empty affine map to
// imply identity affine map.
extfunc @memrefs(memref<2x4xi8>) ; expected-error {{expected ',' in memref type}}

// -----
// Test no map in memref type.
extfunc @memrefs(memref<2x4xi8, >) ; expected-error {{expected list element}}

// -----
// Test non-existent map in memref type.
extfunc @memrefs(memref<2x4xi8, #map7>) ; expected-error {{undefined affine map id 'map7'}}

// -----
// Test non hash identifier in memref type.
extfunc @memrefs(memref<2x4xi8, %map7>) ; expected-error {{expected '(' at start of dimensional identifiers list}}

// -----
// Test non-existent map in map composition of memref type.
#map0 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, #map8>) ; expected-error {{undefined affine map id 'map8'}}

// -----
// Test multiple memory space error.
#map0 = (d0, d1) -> (d0, d1)
extfunc @memrefs(memref<2x4xi8, #map0, 1, 2>) ; expected-error {{multiple memory spaces specified in memref type}}

// -----
// Test affine map after memory space.
#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0, d1) -> (d0, d1)

extfunc @memrefs(memref<2x4xi8, #map0, 1, #map1>) ; expected-error {{affine map after memory space in memref type}}

// -----
// Test no memory space error.
#map0 = (d0, d1) -> (d0, d1)
extfunc @memrefs(memref<2x4xi8, #map0>) ; expected-error {{expected memory space in memref type}}

// -----

cfgfunc @foo()
cfgfunc @bar() // expected-error {{expected '{' in CFG function}}

// -----

extfunc missingsigil() -> (i1, affineint, f32) // expected-error {{expected a function identifier like}}


// -----

cfgfunc @bad_branch() {
bb42:
  br missing  // expected-error {{reference to an undefined basic block 'missing'}}
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
bb42:        // expected-error {{expected operation name}}
  return
}

// -----

mlfunc @foo()
mlfunc @bar() // expected-error {{expected '{' in ML function}}

// -----

mlfunc @no_return() {
}        // expected-error {{ML function must end with return statement}}

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

mlfunc @incomplete_for() {
  for
}        // expected-error {{expected '{' before statement list}}

// -----

mlfunc @non_statement() {
  asd   // expected-error {{expected operation name in quotes}}
}

// -----

cfgfunc @malformed_dim() {
bb42:
  "dim"(){index: "xyz"} : ()->i32 // expected-error {{'dim' op requires an integer attribute named 'index'}}
  return
}

// -----

#map = (d0) -> (%  // expected-error {{invalid SSA name}}

// -----
