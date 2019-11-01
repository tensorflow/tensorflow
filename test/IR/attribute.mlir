// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test Non-negative Int Attr
//===----------------------------------------------------------------------===//

func @non_negative_int_attr_pass() {
  // CHECK: test.non_negative_int_attr
  "test.non_negative_int_attr"() {i32attr = 5 : i32, i64attr = 10 : i64} : () -> ()
  // CHECK: test.non_negative_int_attr
  "test.non_negative_int_attr"() {i32attr = 0 : i32, i64attr = 0 : i64} : () -> ()
  return
}

// -----

func @negative_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: non-negative 32-bit integer attribute}}
  "test.non_negative_int_attr"() {i32attr = -5 : i32, i64attr = 10 : i64} : () -> ()
  return
}

// -----

func @negative_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: non-negative 64-bit integer attribute}}
  "test.non_negative_int_attr"() {i32attr = 5 : i32, i64attr = -10 : i64} : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test Positive Int Attr
//===----------------------------------------------------------------------===//

func @positive_int_attr_pass() {
  // CHECK: test.positive_int_attr
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = 10 : i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: positive 32-bit integer attribute}}
  "test.positive_int_attr"() {i32attr = 0 : i32, i64attr = 5: i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: positive 64-bit integer attribute}}
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = 0: i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: positive 32-bit integer attribute}}
  "test.positive_int_attr"() {i32attr = -10 : i32, i64attr = 5 : i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: positive 64-bit integer attribute}}
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = -10 : i64} : () -> ()
  return
}

// -----

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

// -----

//===----------------------------------------------------------------------===//
// Test FloatElementsAttr
//===----------------------------------------------------------------------===//

func @correct_type_pass() {
  "test.float_elements_attr"() {
    // CHECK: scalar_f32_attr = dense<5.000000e+00> : tensor<2xf32>
    // CHECK: tensor_f64_attr = dense<6.000000e+00> : tensor<4x8xf64>
    scalar_f32_attr = dense<5.0> : tensor<2xf32>,
    tensor_f64_attr = dense<6.0> : tensor<4x8xf64>
  } : () -> ()
  return
}

// -----

func @wrong_element_type_pass() {
  // expected-error @+1 {{failed to satisfy constraint: 32-bit float elements attribute of shape [2]}}
  "test.float_elements_attr"() {
    scalar_f32_attr = dense<5.0> : tensor<2xf64>,
    tensor_f64_attr = dense<6.0> : tensor<4x8xf64>
  } : () -> ()
  return
}

// -----

func @correct_type_pass() {
  // expected-error @+1 {{failed to satisfy constraint: 64-bit float elements attribute of shape [4, 8]}}
  "test.float_elements_attr"() {
    scalar_f32_attr = dense<5.0> : tensor<2xf32>,
    tensor_f64_attr = dense<6.0> : tensor<4xf64>
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test SymbolRefAttr
//===----------------------------------------------------------------------===//

func @fn() { return }

// CHECK: test.symbol_ref_attr
"test.symbol_ref_attr"() {symbol = @fn} : () -> ()

// -----

// expected-error @+1 {{referencing to a 'FuncOp' symbol}}
"test.symbol_ref_attr"() {symbol = @foo} : () -> ()
