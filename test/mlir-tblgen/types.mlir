// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: @complex_f64_success
func @complex_f64_success() {
  "test.complex_f64"() : () -> (complex<f64>)
  return
}

// -----

// CHECK-LABEL: @complex_f64_tensor_success
func @complex_f64_tensor_success() {
  "test.complex_f64_tensor"() : () -> (tensor<complex<f64>>)
  return
}

// -----

func @complex_f64_failure() {
  // expected-error@+1 {{must be complex type with 64-bit float elements}}
  "test.complex_f64"() : () -> (f64)
  return
}

// -----

// CHECK-LABEL: @tuple_success
func @tuple_success() {
  "test.tuple_32_bit"() : () -> (tuple<i32>)
  return
}

// -----

// CHECK-LABEL: @tuple_mixed_success
func @tuple_mixed_success() {
  "test.tuple_32_bit"() : () -> (tuple<i32, f32>)
  return
}

// -----

func @tuple_empty_success() {
  "test.tuple_32_bit"() : () -> (tuple<>)
  return
}

// -----

func @tuple_wrong_type_scalar() {
  // expected-error@+1 {{must be tuple with any combination of 32-bit integer or 32-bit float values}}
  "test.tuple_32_bit"() : () -> (tuple<i64>)
  return
}

// -----

func @tuple_wrong_type_tensor() {
  // expected-error@+1 {{must be tuple with any combination of 32-bit integer or 32-bit float values}}
  "test.tuple_32_bit"() : () -> (tuple<tensor<i32>>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_empty_success
func @nested_tuple_empty_success() {
  "test.nested_tuple_32_bit"() : () -> (tuple<>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_one_level_success
func @nested_tuple_one_level_success() {
  "test.nested_tuple_32_bit"() : () -> (tuple<i32>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_multi_level_success
func @nested_tuple_multi_level_success() {
  "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<i32, tuple<i32>>>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_multi_level_mixed_success
func @nested_tuple_multi_level_mixed_success() {
  "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<f32, tuple<i32>>>)
  return
}

// -----

func @nested_tuple_multi_level_wrong_type() {
  // expected-error@+1 {{must be nested tuple with any combination of 32-bit integer or 32-bit float values}}
  "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<i32, tuple<i64>>>)
  return
}

// -----

// CHECK-LABEL: func @rank_less_than_2_I8_F32_memref_success
func @rank_less_than_2_I8_F32_memref_success() {
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<i8>)
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<3xi8>)
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<f32>)
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<1xf32>)
  return
}

// -----

func @rank_less_than_2_I8_F32_memref_bad_type() {
  // expected-error@+1 {{must be 0D/1D memref of 8-bit integer or 32-bit float values}}
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<i16>)
  return
}

// -----

func @rank_less_than_2_I8_F32_memref_bad_rank() {
  // expected-error@+1 {{must be 0D/1D memref of 8-bit integer or 32-bit float values}}
  "test.rank_less_than_2_I8_F32_memref"() : () -> (memref<1x2xi8>)
  return
}

// -----

func @nd_tensor_of_success(%arg0: tensor<f32>, %arg1: tensor<10xf32>, %arg2: tensor<20x30xi16>, %arg3: tensor<40x50x60xi16>, %arg4: tensor<70x80x90x100xi16>) {
  "test.nd_tensor_of"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<f32>, tensor<10xf32>, tensor<20x30xi16>, tensor<40x50x60xi16>, tensor<70x80x90x100xi16>) -> ()
  return
}

// -----

func @nd_tensor_of_success_wrong_type_0d(%arg0: tensor<f32>, %arg1: tensor<10xf32>, %arg2: tensor<20x30xi16>, %arg3: tensor<40x50x60xi16>, %arg4: tensor<70x80x90x100xi32>) {
  // expected-error @+1 {{'test.nd_tensor_of' op operand #0 must be 0D tensor of 32-bit float values}}
  "test.nd_tensor_of"(%arg1, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<10xf32>, tensor<20x30xi16>, tensor<40x50x60xi16>, tensor<70x80x90x100xi32>) -> ()
  return
}

// -----

func @nd_tensor_of_success_wrong_type_4d(%arg0: tensor<f32>, %arg1: tensor<10xf32>, %arg2: tensor<20x30xi16>, %arg3: tensor<40x50x60xi16>, %arg4: tensor<70x80x90x100xi32>) {
  // expected-error @+1 {{'test.nd_tensor_of' op operand #4 must be 4D tensor of 16-bit integer values}}
  "test.nd_tensor_of"(%arg0, %arg1, %arg2, %arg3, %arg3) : (tensor<f32>, tensor<10xf32>, tensor<20x30xi16>, tensor<40x50x60xi16>, tensor<40x50x60xi16>) -> ()
  return
}

// -----

func @ranked_tensor_success(%arg0: tensor<i8>, %arg1: tensor<1xi32>, %arg2: tensor<?x?xf32>) {
  "test.ranked_tensor_op"(%arg0) : (tensor<i8>) -> ()
  "test.ranked_tensor_op"(%arg1) : (tensor<1xi32>) -> ()
  "test.ranked_tensor_op"(%arg2) : (tensor<?x?xf32>) -> ()
  return
}

// -----

func @ranked_tensor_success(%arg0: tensor<*xf32>) {
  // expected-error @+1 {{must be ranked tensor of any type values}}
  "test.ranked_tensor_op"(%arg0) : (tensor<*xf32>) -> ()
  return
}

// -----

func @ranked_tensor_success(%arg0: vector<2xf32>) {
  // expected-error @+1 {{must be ranked tensor of any type values}}
  "test.ranked_tensor_op"(%arg0) : (vector<2xf32>) -> ()
  return
}

// -----

func @multi_tensor_rank_of_success(%arg0: tensor<i8>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi32>, %arg5: tensor<1xf32>) {
  "test.multi_tensor_rank_of"(%arg0) : (tensor<i8>) -> ()
  "test.multi_tensor_rank_of"(%arg1) : (tensor<i32>) -> ()
  "test.multi_tensor_rank_of"(%arg2) : (tensor<f32>) -> ()
  "test.multi_tensor_rank_of"(%arg3) : (tensor<1xi8>) -> ()
  "test.multi_tensor_rank_of"(%arg4) : (tensor<1xi32>) -> ()
  "test.multi_tensor_rank_of"(%arg5) : (tensor<1xf32>) -> ()
  return
}

// -----

func @multi_tensor_rank_of_wrong_unranked_type(%arg0: tensor<2x2xi8>) {
  // expected-error @+1 {{'test.multi_tensor_rank_of' op operand #0 must be 0D/1D tensor of 8-bit integer or 32-bit integer or 32-bit float values}}
  "test.multi_tensor_rank_of"(%arg0) : (tensor<2x2xi8>) -> ()
  return
}

// -----

func @multi_tensor_rank_of_wrong_element_type(%arg0: tensor<2xi16>) {
  // expected-error @+1 {{'test.multi_tensor_rank_of' op operand #0 must be 0D/1D tensor of 8-bit integer or 32-bit integer or 32-bit float values}}
  "test.multi_tensor_rank_of"(%arg0) : (tensor<2xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: @fixed_element_types
func @fixed_element_types(%ti32: tensor<* x i32>, %tf32: tensor<* x f32>, %mi32 : memref<2x3xi32>, %vf32 : vector<2xf32>) {
  "test.arg_and_res_have_fixed_element_types"(%ti32, %tf32) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i16>
  "test.arg_and_res_have_fixed_element_types"(%mi32, %vf32) : (memref<2x3xi32>, vector<2xf32>) -> memref<1x2xi16>
  return
}

// -----

func @fixed_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{'res' is 16-bit integer}}
  "test.arg_and_res_have_fixed_element_types"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i32>
  return
}

// -----

func @fixed_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{fixed type combination}}
  "test.arg_and_res_have_fixed_element_types"(%arg1, %arg0) : (tensor<* x f32>, tensor<* x i32>) -> tensor<* x i16>
  return
}

// -----

// CHECK-LABEL: same_element_types_success
func @same_element_types_success(%ti32: tensor<* x i32>, %i32 : i32, %mi32 : memref<2x3xi32>) {
  "test.operands_have_same_element_type"(%ti32, %ti32): (tensor<* x i32>, tensor<* x i32>) -> ()
  "test.operands_have_same_element_type"(%i32, %ti32): (i32, tensor<* x i32>) -> ()
  "test.operands_have_same_element_type"(%i32, %mi32): (i32, memref<2x3xi32>) -> ()
  return
}


// -----

func @same_element_types_failure(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{verify that all of {x, y} have same element type}}
  "test.operands_have_same_element_type"(%arg1, %arg0): (tensor<* x f32>, tensor<* x i32>) -> ()
  return
}

// -----

// CHECK-LABEL: same_element_types_success
func @same_element_types_success(%ti32: tensor<* x i32>, %tf32: tensor<* x f32>) {
  "test.operand0_and_result_have_same_element_type"(%tf32, %ti32) : (tensor<* x f32>, tensor<* x i32>) -> tensor<* x f32>
  "test.operand0_and_result_have_same_element_type"(%tf32, %ti32) : (tensor<* x f32>, tensor<* x i32>) -> memref<2x3xf32>
  "test.operand0_and_result_have_same_element_type"(%tf32, %ti32) : (tensor<* x f32>, tensor<* x i32>) -> f32
  return
}

// -----

func @same_element_types_failure(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, res} have same element type}}
  "test.operand0_and_result_have_same_element_type"(%arg1, %arg0) : (tensor<* x f32>, tensor<* x i32>) -> tensor<* x i32>
  return
}

// -----

// CHECK-LABEL: same_types
func @same_types(%ti32: tensor<* x i32>, %tf32: tensor<* x f32>) {
  "test.operands_have_same_type"(%ti32, %ti32) : (tensor<* x i32>, tensor<* x i32>) -> ()
  "test.operand0_and_result_have_same_type"(%ti32, %tf32) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i32>
  return
}

// -----

func @same_types_failure(%ti32: tensor<* x i32>, %i32: i32) {
  // expected-error@+1 {{all of {x, y} have same type}}
  "test.operands_have_same_type"(%ti32, %i32) : (tensor<* x i32>, i32) -> ()
  return
}

// -----

func @same_types_element_mismatch(%ti32: tensor<* x i32>, %tf32: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, y} have same type}}
  "test.operands_have_same_type"(%ti32, %tf32) : (tensor<* x i32>, tensor<* x f32>) -> ()
  return
}

// -----

func @same_types_shape_mismatch(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) {
  // expected-error@+1 {{all of {x, y} have same type}}
  "test.operands_have_same_type"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: same_rank_success
func @same_rank_success(%t1xi : tensor<1xi32>, %t2xf : tensor<2xf32>, %m3xi : memref<3xi32>, %t1x2xf : tensor<1x2xf32>, %t1x2xi : tensor<1x2xi32>) {
  "test.operands_have_same_rank"(%t1xi, %t2xf) : (tensor<1xi32>, tensor<2xf32>) -> ()
  "test.operands_have_same_rank"(%t1xi, %m3xi) : (tensor<1xi32>, memref<3xi32>) -> ()
  "test.operand0_and_result_have_same_rank"(%t1xi, %t1x2xf) : (tensor<1xi32>, tensor<1x2xf32>) -> (tensor<3xf32>)
  "test.operand0_and_result_have_same_rank"(%t1x2xi, %t1x2xf) : (tensor<1x2xi32>, tensor<1x2xf32>) -> (tensor<3x3xf64>)
  return
}

// -----

func @same_rank_failure(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) {
  // expected-error@+1 {{all of {x, y} have same rank}}
  "test.operands_have_same_rank"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> ()
  return
}

// -----

func @same_rank_failure(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) {
  // expected-error@+1 {{all of {x, res} have same rank}}
  "test.operand0_and_result_have_same_rank"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> (tensor<i32>)
  return
}

// -----

func @same_rank_failure(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xf32>) {
  // expected-error@+1 {{all of {x, res} have same rank}}
  "test.operand0_and_result_have_same_rank"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<1x2xf32>) -> (tensor<3xi32>)
  return
}

// -----

// CHECK-LABEL: same_shape_success
func @same_shape_success(%t2x3: tensor<2x3xi32>, %m2x3: memref<2x3xf32>, %v2x3 : vector<2x3xi32>, %t4x5 : tensor<4x5xi32>) {
  "test.operand0_and_result_have_same_shape"(%t2x3, %t4x5) : (tensor<2x3xi32>, tensor<4x5xi32>) -> (tensor<2x3xf32>)
  "test.operand0_and_result_have_same_shape"(%t2x3, %t4x5) : (tensor<2x3xi32>, tensor<4x5xi32>) -> (memref<2x3xf32>)
  "test.operand0_and_result_have_same_shape"(%t2x3, %t4x5) : (tensor<2x3xi32>, tensor<4x5xi32>) -> (vector<2x3xf32>)
  return
}

// -----

func @same_shape_failure(%t2x3: tensor<2x3xi32>, %t4x5 : tensor<4x5xi32>) {
  // expected-error@+1 {{all of {x, res} have same shape}}
  "test.operand0_and_result_have_same_shape"(%t2x3, %t4x5) : (tensor<2x3xi32>, tensor<4x5xi32>) -> (tensor<1x3xf32>)
  return
}

// -----

// CHECK-LABEL: same_element_count_success
func @same_element_count_success(%arg0: tensor<36xi32>, %arg1: tensor<1x2xf32>, %arg3: tensor<f32>) {
  "test.operand0_and_result_have_same_element_count"(%arg0, %arg1) : (tensor<36xi32>, tensor<1x2xf32>) -> (tensor<3x4x3xf32>)
  "test.operand0_and_result_have_same_element_count"(%arg0, %arg1) : (tensor<36xi32>, tensor<1x2xf32>) -> (tensor<3x12xf64>)
  "test.operand0_and_result_have_same_element_count"(%arg3, %arg1) : (tensor<f32>, tensor<1x2xf32>) -> (memref<1x1x1xi32>)
  return
}

// -----

func @same_element_count_failure(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) {
  // expected-error@+1 {{all of {x, res} have same element count}}
  "test.operand0_and_result_have_same_element_count"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> (tensor<2xi32>)
  return
}

// -----

func @four_equals_five() {
  // expected-error@+1 {{failed to verify that 4 equals 5}}
  "test.four_equals_five"() : () -> ()
  return
}

// -----

func @operand_rank_equals_result_size_success(%arg : tensor<1x2x3x4xi32>) {
  "test.operand_rank_equals_result_size"(%arg) : (tensor<1x2x3x4xi32>) -> tensor<4xi32>
  "test.operand_rank_equals_result_size"(%arg) : (tensor<1x2x3x4xi32>) -> memref<2x2xf32>
  return
}

// -----

func @operand_rank_equals_result_size_failure(%arg : tensor<1x2x3x4xi32>) {
  // expected-error@+1 {{failed to verify that operand rank equals result size}}
  "test.operand_rank_equals_result_size"(%arg) : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  return
}

// -----

func @same_types_element_mismatch(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, res} have same type}}
  "test.operand0_and_result_have_same_type"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x f32>
  return
}

// -----

func @same_types_shape_mismatch(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) {
  // expected-error@+1 {{all of {x, res} have same type}}
  "test.operand0_and_result_have_same_type"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
  return
}

// -----

func @does_not_have_i32(%arg0: tensor<1x2xi32>, %arg1: none) {
  // expected-error@+1 {{either both none type operands or first is not none}}
  "test.if_first_operand_is_none_then_so_is_second"(%arg1, %arg0) : (none, tensor<1x2xi32>) -> ()
  return
}

// -----

func @does_not_have_static_memref(%arg0: memref<?xi32>) {
  // expected-error@+1 {{'test.takes_static_memref' op operand #0 must be statically shaped memref of any type values}}
  "test.takes_static_memref"(%arg0) : (memref<?xi32>) -> ()
}

// -----

func @elements_attr_not_i32_f32() {
  // expected-error@+1 {{32-bit integer elements attribute}}
  "test.i32ElementsAttr"() {attr = dense<[1.0, 20.0]>:tensor<2xf32>} : () -> ()
  return
}

// -----

func @elements_attr_not_i32_i64() {
  // expected-error@+1 {{32-bit integer elements attribute}}
  "test.i32ElementsAttr"() {attr = dense<[1, 20]>:tensor<2xi64>} : () -> ()
  return
}


// -----

func @elements_attr_i32(%arg0: tensor<1x2xi32>) {
  "test.i32ElementsAttr"() {attr = dense<[1, 2]>:tensor<2xi32>} : () -> ()
  return
}
