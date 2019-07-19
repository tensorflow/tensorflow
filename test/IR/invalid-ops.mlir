// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @dim(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(%0){index = "xyz"} : (tensor<1xf32>)->index // expected-error {{attribute 'index' failed to satisfy constraint: arbitrary integer attribute}}
  return
}

// -----

func @dim2(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(){index = "xyz"} : ()->index // expected-error {{'std.dim' op requires a single operand}}
  return
}

// -----

func @dim3(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(%0){index = 1} : (tensor<1xf32>)->index // expected-error {{'std.dim' op index is out of range}}
  return
}

// -----

func @rank(f32) {
^bb(%0: f32):
  "std.rank"(%0): (f32)->index // expected-error {{'std.rank' op operand #0 must be tensor of any type values}}
  return
}

// -----

func @constant() {
^bb:
  %x = "std.constant"(){value = "xyz"} : () -> i32 // expected-error {{requires a result type that aligns with the 'value' attribute}}
  return
}

// -----

func @constant_out_of_range() {
^bb:
  %x = "std.constant"(){value = 100} : () -> i1 // expected-error {{requires attribute's type ('i64') to match op's return type ('i1')}}
  return
}

// -----

func @constant_wrong_type() {
^bb:
  %x = "std.constant"(){value = 10.} : () -> f32 // expected-error {{requires attribute's type ('f64') to match op's return type ('f32')}}
  return
}

// -----
func @affine_apply_no_map() {
^bb0:
  %i = constant 0 : index
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{'affine.apply' op requires an affine map}}
  return
}

// -----

func @affine_apply_wrong_operand_count() {
^bb0:
  %i = constant 0 : index
  %x = "affine.apply" (%i) {map = (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index) -> (index) //  expected-error {{'affine.apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

func @affine_apply_wrong_result_count() {
^bb0:
  %i = constant 0 : index
  %j = constant 1 : index
  %x = "affine.apply" (%i, %j) {map = (d0, d1) -> ((d0 + 1), (d1 + 2))} : (index,index) -> (index) //  expected-error {{'affine.apply' op mapping must produce one value}}
  return
}

// -----

func @unknown_custom_op() {
^bb0:
  %i = crazyThing() {value = 0} : () -> index  // expected-error {{custom op 'crazyThing' is unknown}}
  return
}

// -----

func @unknown_std_op() {
  // expected-error@+1 {{unregistered operation 'std.foo_bar_op' found in dialect ('std') that does not allow unknown operations}}
  %0 = "std.foo_bar_op"() : () -> index
  return
}

// -----

func @bad_alloc_wrong_dynamic_dim_count() {
^bb0:
  %0 = constant 7 : index
  // Test alloc with wrong number of dynamic dimensions.
  %1 = alloc(%0)[%1] : memref<2x4xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{custom op 'alloc' dimension operand count does not equal memref dynamic dimension count}}
  return
}

// -----

func @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = constant 7 : index
  // Test alloc with wrong number of symbols
  %1 = alloc(%0) : memref<2x?xf32, (d0, d1)[s0] -> ((d0 + s0), d1), 1> // expected-error {{operand count does not equal dimension plus symbol operand count}}
  return
}

// -----

func @test_store_zero_results() {
^bb0:
  %0 = alloc() : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>
  %1 = constant 0 : index
  %2 = constant 1 : index
  %3 = load %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1>
  // Test that store returns zero results.
  %4 = store %3, %0[%1, %2] : memref<1024x64xf32, (d0, d1) -> (d0, d1), 1> // expected-error {{cannot name an operation with no results}}
  return
}

// -----

func @test_store_zero_results2(%x: i32, %p: memref<i32>) {
  "std.store"(%x,%p) : (i32, memref<i32>) -> i32  // expected-error {{'std.store' op requires zero results}}
  return
}

// -----

func @test_alloc_memref_map_rank_mismatch() {
^bb0:
  %0 = alloc() : memref<1024x64xf32, (d0) -> (d0), 1> // expected-error {{memref affine map dimension mismatch}}
  return
}

// -----

func @intlimit2() {
^bb:
  %0 = "std.constant"() {value = 0} : () -> i4096
  %1 = "std.constant"() {value = 1} : () -> i4097 // expected-error {{integer bitwidth is limited to 4096 bits}}
  return
}

// -----

func @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{incorrect number of operands for callee}}
  return
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf %a, %a, %a : f32  // expected-error {{custom op 'addf' expected 2 operands}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf(%a, %a) : f32  // expected-error {{unexpected delimiter}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf{%a, %a} : f32  // expected-error {{invalid operand}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  // expected-error@+1 {{'std.addi' op operand #0 must be integer-like}}
  %sf = addi %a, %a : f32
}

// -----

func @func_with_ops(i32) {
^bb0(%a : i32):
  %sf = addf %a, %a : i32  // expected-error {{'std.addf' op operand #0 must be floating-point-like}}
}

// -----

func @func_with_ops(i32) {
^bb0(%a : i32):
  // expected-error@+1 {{'predicate' attribute value out of range}}
  %r = "std.cmpi"(%a, %a) {predicate = 42} : (i32, i32) -> i1
}

// -----

// Comparison are defined for arguments of the same type.
func @func_with_ops(i32, i64) {
^bb0(%a : i32, %b : i64): // expected-note {{prior use here}}
  %r = cmpi "eq", %a, %b : i32 // expected-error {{use of value '%b' expects different type than prior uses}}
}

// -----

// Comparisons must have the "predicate" attribute.
func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = cmpi %a, %b : i32 // expected-error {{expected non-function type}}
}

// -----

// Integer comparisons are not recognized for float types.
func @func_with_ops(f32, f32) {
^bb0(%a : f32, %b : f32):
  %r = cmpi "eq", %a, %b : f32 // expected-error {{operand #0 must be integer-like}}
}

// -----

// Result type must be boolean like.
func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = "std.cmpi"(%a, %b) {predicate = 0} : (i32, i32) -> i32 // expected-error {{op result #0 must be bool-like}}
}

// -----

func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  // expected-error@+1 {{requires an integer attribute named 'predicate'}}
  %r = "std.cmpi"(%a, %b) {foo = 1} : (i32, i32) -> i1
}

// -----

func @func_with_ops() {
^bb0:
  %c = constant dense<0> : vector<42 x i32>
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %r = "std.cmpi"(%c, %c) {predicate = 0} : (vector<42 x i32>, vector<42 x i32>) -> vector<41 x i1>
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+2 {{different type than prior uses}}
  // expected-note@-2 {{prior use here}}
  %r = select %cond, %t, %f : i32
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+1 {{op operand #0 must be bool-like}}
  %r = "std.select"(%cond, %t, %f) : (i32, i32, i32) -> i32
}

// -----

func @func_with_ops(i1, i32, i64) {
^bb0(%cond : i1, %t : i32, %f : i64):
  // expected-error@+1 {{'true' and 'false' arguments to be of the same type}}
  %r = "std.select"(%cond, %t, %f) : (i1, i32, i64) -> i32
}

// -----

func @func_with_ops(i1, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : i1, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %r = "std.select"(%cond, %t, %f) : (i1, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

func @func_with_ops(i1, tensor<42xi32>, tensor<?xi32>) {
^bb0(%cond : i1, %t : tensor<42xi32>, %f : tensor<?xi32>):
  // expected-error@+1 {{ op requires the same shape for all operands and results}}
  %r = "std.select"(%cond, %t, %f) : (i1, tensor<42xi32>, tensor<?xi32>) -> tensor<42xi32>
}

// -----

func @func_with_ops(tensor<?xi1>, tensor<42xi32>, tensor<42xi32>) {
^bb0(%cond : tensor<?xi1>, %t : tensor<42xi32>, %f : tensor<42xi32>):
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %r = "std.select"(%cond, %t, %f) : (tensor<?xi1>, tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{expected 2 types}}
  %0 = vector.transfer_read %arg0[%c3, %c3] : memref<?x?xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{expected 2 indices to the memref}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3] : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  %0 = vector.transfer_read %arg0[%c3, %c3] : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  %0 = vector.transfer_read %arg0[%c3, %c3] {perm = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  %0 = vector.transfer_read %arg0[%c3, %c3] {permutation_map = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  %0 = vector.transfer_read %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0, d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + 1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(memref<?x?x?xf32>) {
^bb0(%arg0: memref<?x?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3] {permutation_map = (d0, d1, d2)->(d0, d0)} : memref<?x?x?xf32>, vector<3x7xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{expected 5 operand types but had 4}}
  %0 = "vector.transfer_write"(%cst, %arg0, %c3, %c3, %c3) : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{expects 4 operands (of which 2 indices)}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  vector.transfer_write %cst, %arg0[%c3, %c3] : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires an AffineMapAttr named 'permutation_map'}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {perm = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0, d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(memref<?x?xf32>) {
^bb0(%arg0: memref<?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + 1)} : vector<128xf32>, memref<?x?xf32>
}
// -----

func @test_vector.transfer_write(memref<?x?x?xf32>) {
^bb0(%arg0: memref<?x?x?xf32>):
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<3 x 7 x f32>
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = (d0, d1, d2)->(d0, d0)} : vector<3x7xf32>, memref<?x?x?xf32>
}

// -----

func @invalid_select_shape(%cond : i1, %idx : () -> ()) {
  // expected-error@+1 {{expected type with valid i1 shape}}
  %sel = select %cond, %idx, %idx : () -> ()

// -----

func @invalid_cmp_shape(%idx : () -> ()) {
  // expected-error@+1 {{expected type with valid i1 shape}}
  %cmp = cmpi "eq", %idx, %idx : () -> ()

// -----

func @dma_no_src_memref(%m : f32, %tag : f32, %c0 : index) {
  // expected-error@+1 {{expected source to be of memref type}}
  dma_start %m[%c0], %m[%c0], %c0, %tag[%c0] : f32, f32, f32
}

// -----

func @dma_no_dst_memref(%m : f32, %tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected destination to be of memref type}}
  dma_start %mref[%c0], %m[%c0], %c0, %tag[%c0] : memref<8 x f32>, f32, f32
}

// -----

func @dma_no_tag_memref(%tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected tag to be of memref type}}
  dma_start %mref[%c0], %mref[%c0], %c0, %tag[%c0] : memref<8 x f32>, memref<8 x f32>, f32
}

// -----

func @dma_wait_no_tag_memref(%tag : f32, %c0 : index) {
  // expected-error@+1 {{expected tag to be of memref type}}
  dma_wait %tag[%c0], %arg0 : f32
}

// -----

func @invalid_cmp_attr(%idx : i32) {
  // expected-error@+1 {{expected string comparison predicate attribute}}
  %cmp = cmpi i1, %idx, %idx : i32

// -----

func @cmpf_generic_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{'predicate' attribute value out of range}}
  %r = "std.cmpf"(%a, %a) {predicate = 42} : (f32, f32) -> i1
}

// -----

func @cmpf_canonical_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "foo"}}
  %r = cmpf "foo", %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_signed(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "sge"}}
  %r = cmpf "sge", %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_no_order(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "eq"}}
  %r = cmpf "eq", %a, %a : f32
}

// -----

func @cmpf_canonical_no_predicate_attr(%a : f32, %b : f32) {
  %r = cmpf %a, %b : f32 // expected-error {{}}
}

// -----

func @cmpf_generic_no_predicate_attr(%a : f32, %b : f32) {
  // expected-error@+1 {{requires an integer attribute named 'predicate'}}
  %r = "std.cmpf"(%a, %b) {foo = 1} : (f32, f32) -> i1
}

// -----

func @cmpf_wrong_type(%a : i32, %b : i32) {
  %r = cmpf "oeq", %a, %b : i32 // expected-error {{operand #0 must be floating-point-like}}
}

// -----

func @cmpf_generic_wrong_result_type(%a : f32, %b : f32) {
  // expected-error@+1 {{result #0 must be bool-like}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (f32, f32) -> f32
}

// -----

func @cmpf_canonical_wrong_result_type(%a : f32, %b : f32) -> f32 {
  %r = cmpf "oeq", %a, %b : f32 // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%r' expects different type than prior uses}}
  return %r : f32
}

// -----

func @cmpf_result_shape_mismatch(%a : vector<42xf32>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %r = "std.cmpf"(%a, %a) {predicate = 0} : (vector<42 x f32>, vector<42 x f32>) -> vector<41 x i1>
}

// -----

func @cmpf_operand_shape_mismatch(%a : vector<42xf32>, %b : vector<41xf32>) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (vector<42 x f32>, vector<41 x f32>) -> vector<42 x i1>
}

// -----

func @cmpf_generic_operand_type_mismatch(%a : f32, %b : f64) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (f32, f64) -> i1
}

// -----

func @cmpf_canonical_type_mismatch(%a : f32, %b : f64) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = cmpf "oeq", %a, %b : f32
}

// -----

func @extract_element_no_operands() {
  // expected-error@+1 {{op expected 1 or more operands}}
  %0 = "std.extract_element"() : () -> f32
  return
}

// -----

func @extract_element_no_indices(%v : vector<3xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%v) : (vector<3xf32>) -> f32
  return
}

// -----

func @extract_element_invalid_index_type(%v : vector<3xf32>, %i : i32) {
  // expected-error@+1 {{operand #1 must be index}}
  %0 = "std.extract_element"(%v, %i) : (vector<3xf32>, i32) -> f32
  return
}

// -----

func @extract_element_element_result_type_mismatch(%v : vector<3xf32>, %i : index) {
  // expected-error@+1 {{result type must match element type of aggregate}}
  %0 = "std.extract_element"(%v, %i) : (vector<3xf32>, index) -> f64
  return
}

// -----

func @extract_element_vector_too_many_indices(%v : vector<3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%v, %i, %i) : (vector<3xf32>, index, index) -> f32
  return
}

// -----

func @extract_element_tensor_too_many_indices(%t : tensor<2x3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%t, %i, %i, %i) : (tensor<2x3xf32>, index, index, index) -> f32
  return
}

// -----

func @extract_element_tensor_too_few_indices(%t : tensor<2x3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%t, %i) : (tensor<2x3xf32>, index) -> f32
  return
}

// -----

func @index_cast_index_to_index(%arg0: index) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0: index to index
  return
}

// -----

func @index_cast_float(%arg0: index, %arg1: f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0 : index to f32
  return
}

// -----

func @index_cast_float_to_index(%arg0: f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0 : f32 to index
  return
}

// -----

func @return_not_in_function() {
  "foo.region"() ({
    // expected-error@+1 {{must be nested within a 'func' region}}
    return
  }): () -> ()
  return
}
