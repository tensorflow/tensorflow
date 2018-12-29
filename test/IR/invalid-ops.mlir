// RUN: mlir-opt %s -split-input-file -verify

cfgfunc @dim(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "dim"(%0){index: "xyz"} : (tensor<1xf32>)->i32 // expected-error {{'dim' op requires an integer attribute named 'index'}}
  return
}

// -----

cfgfunc @dim2(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "dim"(){index: "xyz"} : ()->i32 // expected-error {{'dim' op requires a single operand}}
  return
}

// -----

cfgfunc @dim3(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "dim"(%0){index: 1} : (tensor<1xf32>)->i32 // expected-error {{'dim' op index is out of range}}
  return
}

// -----

cfgfunc @constant() {
^bb:
  %x = "constant"(){value: "xyz"} : () -> i32 // expected-error {{'constant' op requires 'value' to be an integer for an integer result type}}
  return
}

// -----

cfgfunc @constant_out_of_range() {
^bb:
  %x = "constant"(){value: 100} : () -> i1 // expected-error {{'constant' op requires 'value' to be an integer within the range of the integer result type}}
  return
}

// -----

cfgfunc @affine_apply_no_map() {
^bb0:
  %i = "constant"() {value: 0} : () -> index
  %x = "affine_apply" (%i) { } : (index) -> (index) //  expected-error {{'affine_apply' op requires an affine map}}
  return
}

// -----

cfgfunc @affine_apply_wrong_operand_count() {
^bb0:
  %i = "constant"() {value: 0} : () -> index
  %x = "affine_apply" (%i) {map: (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index) -> (index) //  expected-error {{'affine_apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

cfgfunc @affine_apply_wrong_result_count() {
^bb0:
  %i = "constant"() {value: 0} : () -> index
  %j = "constant"() {value: 1} : () -> index
  %x = "affine_apply" (%i, %j) {map: (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index,index) -> (index) //  expected-error {{'affine_apply' op result count and affine map result count must match}}
  return
}

// -----

cfgfunc @unknown_custom_op() {
^bb0:
  %i = crazyThing() {value: 0} : () -> index  // expected-error {{custom op 'crazyThing' is unknown}}
  return
}

// -----

cfgfunc @bad_alloc_wrong_dynamic_dim_count() {
^bb0:
  %0 = "constant"() {value: 7} : () -> index
  // Test alloc with wrong number of dynamic dimensions.
  %1 = alloc(%0)[%1] : memref<2x4xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{custom op 'alloc' dimension operand count does not equal memref dynamic dimension count}}
  return
}

// -----

cfgfunc @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = "constant"() {value: 7} : () -> index
  // Test alloc with wrong number of symbols
  %1 = alloc(%0) : memref<2x?xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{operand count does not equal dimension plus symbol operand count}}
  return
}

// -----

cfgfunc @test_store_zero_results() {
^bb0:
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
^bb0:
  %0 = alloc() : memref<1024x64xf32, (d0) -> (d0), 1> // expected-error {{memref affine map dimension mismatch}}
  return
}

// -----

cfgfunc @intlimit2() {
^bb:
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

mlfunc @mlfunc_constant_out_of_range() {
  %x = "constant"(){value: 100} : () -> i1 // expected-error {{'constant' op requires 'value' to be an integer within the range of the integer result type}}
  return
}

// -----

mlfunc @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{reference to function with mismatched type}}
  return
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf %a, %a, %a : f32  // expected-error {{custom op 'addf' expected 2 operands}}
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf(%a, %a) : f32  // expected-error {{unexpected delimiter}}
}

// -----

cfgfunc @cfgfunc_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf{%a, %a} : f32  // expected-error {{invalid operand}}
}


// -----

cfgfunc @cfgfunc_with_ops(i32) {
^bb0(%a : i32):
  %sf = addf %a, %a : i32  // expected-error {{'addf' op requires a floating point type}}
}

// -----

cfgfunc @cfgfunc_with_ops(i32) {
^bb0(%a : i32):
  // expected-error@+1 {{'predicate' attribute value out of range}}
  %r = "cmpi"(%a, %b) {predicate: 42} : (i32, i32) -> i1
}

// -----

// Comparison are defined for arguments of the same type.
cfgfunc @cfgfunc_with_ops(i32, i64) {
^bb0(%a : i32, %b : i64): // expected-error {{prior use here}}
  %r = cmpi "eq", %a, %b : i32 // expected-error {{use of value '%b' expects different type than prior uses}}
}

// -----

// Comparisons must have the "predicate" attribute.
cfgfunc @cfgfunc_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = cmpi %a, %b : i32 // expected-error {{expected type}}
}

// -----

// Integer comparisons are not recognized for float types.
cfgfunc @cfgfunc_with_ops(f32, f32) {
^bb0(%a : f32, %b : f32):
  %r = cmpi "eq", %a, %b : f32 // expected-error {{op requires an integer or index type}}
}

// -----

// Result type must be boolean like.
cfgfunc @cfgfunc_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = "cmpi"(%a, %b) {predicate: 0} : (i32, i32) -> i32 // expected-error {{op requires a bool result type}}
}

// -----

cfgfunc @cfgfunc_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  // expected-error@+1 {{requires an integer attribute named 'predicate'}}
  %r = "cmpi"(%a, %b) {foo: 1} : (i32, i32) -> i1
}

// -----

cfgfunc @cfgfunc_with_ops() {
^bb0:
  %c = constant splat<vector<42 x i32>, 0> : vector<42 x i32>
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %r = "cmpi"(%c, %c) {predicate: 0} : (vector<42 x i32>, vector<42 x i32>) -> vector<41 x i1>
}

// -----

cfgfunc @cfgfunc_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+2 {{different type than prior uses}}
  // expected-error@-2 {{prior use here}}
  %r = select %cond, %t, %f : i32
}

// -----

cfgfunc @cfgfunc_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+1 {{elemental type i1}}
  %r = "select"(%cond, %t, %f) : (i32, i32, i32) -> i32
}

// -----

cfgfunc @cfgfunc_with_ops(i1, i32, i64) {
^bb0(%cond : i1, %t : i32, %f : i64):
  // expected-error@+1 {{'true' and 'false' arguments to be of the same type}}
  %r = "select"(%cond, %t, %f) : (i1, i32, i64) -> i32
}

// -----

cfgfunc @cfgfunc_with_ops(i1, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : i1, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{requires the condition to have the same shape as arguments}}
  %r = "select"(%cond, %t, %f) : (i1, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

cfgfunc @cfgfunc_with_ops(i1, tensor<42xi32>, tensor<?xi32>) {
^bb0(%cond : i1, %t : tensor<42xi32>, %f : tensor<?xi32>):
  // expected-error@+1 {{'true' and 'false' arguments to be of the same type}}
  %r = "select"(%cond, %t, %f) : (i1, tensor<42xi32>, tensor<?xi32>) -> tensor<42xi32>
}

// -----

cfgfunc @cfgfunc_with_ops(tensor<?xi1>, tensor<42xi32>, tensor<42xi32>) {
^bb0(%cond : tensor<?xi1>, %t : tensor<42xi32>, %f : tensor<42xi32>):
  // expected-error@+1 {{requires the condition to have the same shape as arguments}}
  %r = "select"(%cond, %t, %f) : (tensor<?xi1>, tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{expected 4 operand types but had 3}}
  %0 = "vector_transfer_read"(%arg0, %c3, %c3, %c3) : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires 3 operands}}
  %0 = vector_transfer_read %arg0, %c3, %c3, %c3 : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  %0 = vector_transfer_read %arg0, %c3, %c3 : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  %0 = vector_transfer_read %arg0, %c3, %c3 {perm: (d0)->(d0)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0)->(d0)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0, d1)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0 + d1)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
}

// -----

cfgfunc @test_vector_transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector_transfer_read %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0 + 1)} : (memref<?x?xf32>, index, index) -> vector<128xf32>
}
// -----

cfgfunc @test_vector_transfer_read(memref<?x?x?xf32>) {
^bb0(%arg0: memref<?x?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  %0 = vector_transfer_read %arg0, %c3, %c3, %c3 {permutation_map: (d0, d1, d2)->(d0, d0)} : (memref<?x?x?xf32>, index, index, index) -> vector<3x7xf32>
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{expected 5 operand types but had 4}}
  %0 = "vector_transfer_write"(%cst, %arg0, %c3, %c3, %c3) : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires number of operands and input types to match}}
  vector_transfer_write %cst, %arg0, %c3, %c3, %c3 : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  vector_transfer_write %cst, %arg0, %c3, %c3 : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  vector_transfer_write %cst, %arg0, %c3, %c3 {perm: (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  vector_transfer_write %cst, %arg0, %c3, %c3 {permutation_map: (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  vector_transfer_write %cst, %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0, d1)} : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector_transfer_write %cst, %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0 + d1)} : vector<128xf32>, memref<?x?xf32>, index, index
}

// -----

cfgfunc @test_vector_transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<128 x f32>, 3.0>  : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector_transfer_write %cst, %arg0, %c3, %c3 {permutation_map: (d0, d1)->(d0 + 1)} : vector<128xf32>, memref<?x?xf32>, index, index
}
// -----

cfgfunc @test_vector_transfer_write(memref<?x?x?xf32>) {
^bb0(%arg0: memref<?x?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant splat<vector<3 x 7 x f32>, 3.0>  : vector<3 x 7 x f32>
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  vector_transfer_write %cst, %arg0, %c3, %c3, %c3 {permutation_map: (d0, d1, d2)->(d0, d0)} : vector<3x7xf32>, memref<?x?x?xf32>, index, index, index
}



