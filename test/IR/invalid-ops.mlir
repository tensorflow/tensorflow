// RUN: mlir-opt %s -split-input-file -verify

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
  %i = "constant"() {value: 0} : () -> index
  %x = "affine_apply" (%i) { } : (index) -> (index) //  expected-error {{'affine_apply' op requires an affine map}}
  return
}

// -----

cfgfunc @affine_apply_wrong_operand_count() {
bb0:
  %i = "constant"() {value: 0} : () -> index
  %x = "affine_apply" (%i) {map: (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index) -> (index) //  expected-error {{'affine_apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

cfgfunc @affine_apply_wrong_result_count() {
bb0:
  %i = "constant"() {value: 0} : () -> index
  %j = "constant"() {value: 1} : () -> index
  %x = "affine_apply" (%i, %j) {map: (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index,index) -> (index) //  expected-error {{'affine_apply' op result count and affine map result count must match}}
  return
}

// -----

cfgfunc @unknown_custom_op() {
bb0:
  %i = crazyThing() {value: 0} : () -> index  // expected-error {{custom op 'crazyThing' is unknown}}
  return
}

// -----

cfgfunc @bad_alloc_wrong_dynamic_dim_count() {
bb0:
  %0 = "constant"() {value: 7} : () -> index
  // Test alloc with wrong number of dynamic dimensions.
  %1 = alloc(%0)[%1] : memref<2x4xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{custom op 'alloc' dimension operand count does not equal memref dynamic dimension count}}
  return
}

// -----

cfgfunc @bad_alloc_wrong_symbol_count() {
bb0:
  %0 = "constant"() {value: 7} : () -> index
  // Test alloc with wrong number of symbols
  %1 = alloc(%0) : memref<2x?xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{operand count does not equal dimension plus symbol operand count}}
  return
}

// -----

cfgfunc @test_store_zero_results() {
bb0:
  %0 = alloc() : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>
  %1 = "constant"() {value: 0} : () -> index
  %2 = "constant"() {value: 1} : () -> index
  %3 = load %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>
  // Test that store returns zero results.
  %4 = store %3, %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1> // expected-error {{cannot name an operation with no results}}
  return
}

// -----

mlfunc @test_store_zero_results2(%x: i32, %p: memref<i32>) {
  "store"(%x,%p) : (i32, memref<i32>) -> i32  // expected-error {{'store' op requires zero results}}
  return
}

// -----

cfgfunc @test_alloc_memref_map_rank_mismatch() {
bb0:
  %0 = alloc() : memref<1024x64xf32, (d0) -> (d0), 1> // expected-error {{memref affine map dimension mismatch}}
  return
}

// -----

cfgfunc @intlimit2() {
bb:
  %0 = "constant"() {value: 0} : () -> i4096
  %1 = "constant"() {value: 1} : () -> i4097 // expected-error {{integer bitwidth is limited to 4096 bits}}
  return
}

// -----

mlfunc @mlfunc_constant() {
  %x = "constant"(){value: "xyz"} : () -> i32 // expected-error {{'constant' op requires 'value' to be an integer for an integer result type}}
  return
}

// -----

mlfunc @calls(%arg0 : i32) {
  %x = call @calls() : () -> i32  // expected-error {{reference to function with mismatched type}}
  return
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
bb0(%a : f32):
  %sf = addf %a, %a, %a : f32  // expected-error {{custom op 'addf' expected 2 operands}}
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
bb0(%a : f32):
  %sf = addf(%a, %a) : f32  // expected-error {{unexpected delimiter}}
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
bb0(%a : f32):
  %sf = addf{%a, %a} : f32  // expected-error {{invalid operand}}
}


// -----

cfgfunc @cfgfunc_with_ops(i32) {
bb0(%a : i32):
  %sf = addf %a, %a : i32  // expected-error {{'addf' op requires a floating point type}}
}
