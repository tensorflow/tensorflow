// RUN: mlir-test-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: @tuple_success
func @tuple_success() {
  %0 = "test.tuple_32_bit"() : () -> (tuple<i32>)
  return
}

// -----

// CHECK-LABEL: @tuple_mixed_success
func @tuple_mixed_success() {
  %0 = "test.tuple_32_bit"() : () -> (tuple<i32, f32>)
  return
}

// -----

func @tuple_empty_success() {
  %0 = "test.tuple_32_bit"() : () -> (tuple<>)
  return
}

// -----

func @tuple_wrong_type_scalar() {
  // expected-error@+1 {{must be tuple with any combination of 32-bit integer or 32-bit float values}}
  %0 = "test.tuple_32_bit"() : () -> (tuple<i64>)
  return
}

// -----

func @tuple_wrong_type_tensor() {
  // expected-error@+1 {{must be tuple with any combination of 32-bit integer or 32-bit float values}}
  %0 = "test.tuple_32_bit"() : () -> (tuple<tensor<i32>>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_empty_success
func @nested_tuple_empty_success() {
  %0 = "test.nested_tuple_32_bit"() : () -> (tuple<>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_one_level_success
func @nested_tuple_one_level_success() {
  %0 = "test.nested_tuple_32_bit"() : () -> (tuple<i32>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_multi_level_success
func @nested_tuple_multi_level_success() {
  %0 = "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<i32, tuple<i32>>>)
  return
}

// -----

// CHECK-LABEL: @nested_tuple_multi_level_mixed_success
func @nested_tuple_multi_level_mixed_success() {
  %0 = "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<f32, tuple<i32>>>)
  return
}

// -----

func @nested_tuple_multi_level_wrong_type() {
  // expected-error@+1 {{must be nested tuple with any combination of 32-bit integer or 32-bit float values}}
  %0 = "test.nested_tuple_32_bit"() : () -> (tuple<i32, tuple<i32, tuple<i64>>>)
  return
}

// -----

// CHECK-LABEL: @fixed_element_types
func @fixed_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  %0 = "test.arg_and_res_have_fixed_element_types"(%arg0, %arg1) {attr: ""} : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i16>
  return
}

// -----

func @fixed_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{'res' is 16-bit integer}}
  %0 = "test.arg_and_res_have_fixed_element_types"(%arg0, %arg1) {attr: ""}: (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i32>
  return
}

// -----

func @fixed_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{fixed type combination}}
  %0 = "test.arg_and_res_have_fixed_element_types"(%arg1, %arg0) {attr: ""}: (tensor<* x f32>, tensor<* x i32>) -> tensor<* x i16>
  return
}

// -----

func @same_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{verify that all of {x, y} have same element type}}
  "test.operands_have_same_element_type"(%arg1, %arg0): (tensor<* x f32>, tensor<* x i32>) -> ()
  return
}

// -----

// CHECK-LABEL: same_element_types
func @same_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  %0 = "test.operand_one_and_result_have_same_element_type"(%arg1, %arg0) : (tensor<* x f32>, tensor<* x i32>) -> tensor<* x f32>
  return
}

// -----

func @same_element_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, res} have same element type}}
  %0 = "test.operand_one_and_result_have_same_element_type"(%arg1, %arg0) : (tensor<* x f32>, tensor<* x i32>) -> tensor<* x i32>
  return
}

// -----

// CHECK-LABEL: same_types
func @same_types(%arg0: tensor<* x i32>, %arg1: tensor<* x i32>) {
  "test.operands_have_same_type"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x i32>) -> ()
  return
}

// -----

func @same_types_element_mismatch(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, y} have same type}}
  "test.operands_have_same_type"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x f32>) -> ()
  return
}

// -----

func @same_types_shape_mismatch(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) {
  // expected-error@+1 {{all of {x, y} have same type}}
  "test.operands_have_same_type"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: same_types
func @same_types(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  %0 = "test.operand_one_and_result_have_same_type"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x i32>
  return
}

// -----

func @same_types_element_mismatch(%arg0: tensor<* x i32>, %arg1: tensor<* x f32>) {
  // expected-error@+1 {{all of {x, res} have same type}}
  %0 = "test.operand_one_and_result_have_same_type"(%arg0, %arg1) : (tensor<* x i32>, tensor<* x f32>) -> tensor<* x f32>
  return
}

// -----

func @same_types_shape_mismatch(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) {
  // expected-error@+1 {{all of {x, res} have same type}}
  %0 = "test.operand_one_and_result_have_same_type"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
  return
}
