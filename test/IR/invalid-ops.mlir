// TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
// statements (perhaps through using lit config substitutions).
//
// RUN: %S/../../mlir-opt %s -o - -check-parser-errors

cfgfunc @dim(tensor<1xf32>) {
bb(%0: tensor<1xf32>):
  "dim"(%0){index: "xyz"} : (tensor<1xf32>)->i32 // expected-error {{'dim' op requires an integer attribute named 'index'}}
  return
}

// -----

cfgfunc @dim2(tensor<1xf32>) {
bb(%0: tensor<1xf32>):
  "dim"(){index: "xyz"} : ()->i32 // expected-error {{'dim' op requires a single operand}}
  return
}

// -----

cfgfunc @dim3(tensor<1xf32>) {
bb(%0: tensor<1xf32>):
  "dim"(%0){index: 1} : (tensor<1xf32>)->i32 // expected-error {{'dim' op index is out of range}}
  return
}

// -----

cfgfunc @constant() {
bb:
  %x = "constant"(){value: "xyz"} : () -> i32 // expected-error {{'constant' op requires 'value' to be an integer for an integer result type}}
  return
}

// -----

cfgfunc @affine_apply_no_map() {
bb0:
  // TODO Make constant work with affineint.
  %i = "const"() {value: 0} : () -> affineint
  %x = "affine_apply" (%i) { } : (affineint) -> (affineint) //  expected-error {{'affine_apply' op requires an affine map.}}
  return
}
