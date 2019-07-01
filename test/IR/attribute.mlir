// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test TypeArrayAttr
//===----------------------------------------------------------------------===//

func @correct_type_array_attr_pass() {
  // CHECK: test.type_array_attr
  "test.type_array_attr"() {attr = [i32, f32]} : () -> ()
  return
}

// -----

func @non_type_in_type_array_attr_fail() {
  // expected-error @+1 {{'attr' failed to satisfy constraint: type array attribute}}
  "test.type_array_attr"() {attr = [i32, 5 : i64]} : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test StringAttr with custom type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @string_attr_custom_type
func @string_attr_custom_type() {
  // CHECK: "string_data" : !foo.string
  test.string_attr_with_type "string_data"
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test StrEnumAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @allowed_cases_pass
func @allowed_cases_pass() {
  // CHECK: test.str_enum_attr
  %0 = "test.str_enum_attr"() {attr = "A"} : () -> i32
  // CHECK: test.str_enum_attr
  %1 = "test.str_enum_attr"() {attr = "B"} : () -> i32
  return
}

// -----

func @disallowed_case_fail() {
  // expected-error @+1 {{allowed string cases: 'A', 'B'}}
  %0 = "test.str_enum_attr"() {attr = 7: i32} : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test I32EnumAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @allowed_cases_pass
func @allowed_cases_pass() {
  // CHECK: test.i32_enum_attr
  %0 = "test.i32_enum_attr"() {attr = 5: i32} : () -> i32
  // CHECK: test.i32_enum_attr
  %1 = "test.i32_enum_attr"() {attr = 10: i32} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 32-bit integer cases: 5, 10}}
  %0 = "test.i32_enum_attr"() {attr = 7: i32} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 32-bit integer cases: 5, 10}}
  %0 = "test.i32_enum_attr"() {attr = 5: i64} : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test I64EnumAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @allowed_cases_pass
func @allowed_cases_pass() {
  // CHECK: test.i64_enum_attr
  %0 = "test.i64_enum_attr"() {attr = 5: i64} : () -> i32
  // CHECK: test.i64_enum_attr
  %1 = "test.i64_enum_attr"() {attr = 10: i64} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 64-bit integer cases: 5, 10}}
  %0 = "test.i64_enum_attr"() {attr = 7: i64} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 64-bit integer cases: 5, 10}}
  %0 = "test.i64_enum_attr"() {attr = 5: i32} : () -> i32
  return
}
