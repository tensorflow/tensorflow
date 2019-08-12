// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.global @global(42 : i64) : !llvm.i64
llvm.global @global(42 : i64) : !llvm.i64

// CHECK: llvm.global constant @constant(3.700000e+01 : f64) : !llvm.float
llvm.global constant @constant(37.0) : !llvm.float

// CHECK: llvm.global constant @string("foobar")
llvm.global constant @string("foobar") : !llvm<"[6 x i8]">

// CHECK: llvm.global @string_notype("1234567")
llvm.global @string_notype("1234567")

// CHECK-LABEL: references
func @references() {
  // CHECK: llvm.addressof @global : !llvm<"i64*">
  %0 = llvm.addressof @global : !llvm<"i64*">

  // CHECK: llvm.addressof @string : !llvm<"[6 x i8]*">
  %1 = llvm.addressof @string : !llvm<"[6 x i8]*">

  llvm.return
}

// -----

// expected-error @+1 {{op requires attribute 'sym_name'}}
"llvm.global"() {type = !llvm.i64, constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{op requires attribute 'type'}}
"llvm.global"() {sym_name = "foo", constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{op requires attribute 'value'}}
"llvm.global"() {sym_name = "foo", type = !llvm.i64, constant} : () -> ()

// -----

// expected-error @+1 {{expects type to be a valid element type for an LLVM pointer}}
llvm.global constant @constant(37.0) : !llvm<"label">

// -----

func @foo() {
  // expected-error @+1 {{must appear at the module level}}
  llvm.global @bar(42) : !llvm.i32
}

// -----

// expected-error @+1 {{requires an i8 array type of the length equal to that of the string}}
llvm.global constant @string("foobar") : !llvm<"[42 x i8]">

// -----

// expected-error @+1 {{type can only be omitted for string globals}}
llvm.global @i64_needs_type(0: i64)

// -----

// expected-error @+1 {{expected zero or one type}}
llvm.global @more_than_one_type(0) : !llvm.i64, !llvm.i32

// -----

llvm.global @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+2{{expected ':'}}
  llvm.addressof @foo
}

// -----

func @foo() {
  // The attribute parser will consume the first colon-type, so we put two of
  // them to trigger the attribute type mismatch error.
  // expected-error @+1 {{expected symbol reference}}
  llvm.addressof "foo" : i64 : !llvm<"void ()*">
}

// -----

func @foo() {
  // expected-error @+1 {{must reference a global defined by 'llvm.global'}}
  llvm.addressof @foo : !llvm<"void ()*">
}

// -----

llvm.global @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referred global}}
  llvm.addressof @foo : !llvm<"i64*">
}
