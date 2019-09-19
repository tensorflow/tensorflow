// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.mlir.global @global(42 : i64) : !llvm.i64
llvm.mlir.global @global(42 : i64) : !llvm.i64

// CHECK: llvm.mlir.global constant @constant(3.700000e+01 : f64) : !llvm.float
llvm.mlir.global constant @constant(37.0) : !llvm.float

// CHECK: llvm.mlir.global constant @string("foobar")
llvm.mlir.global constant @string("foobar") : !llvm<"[6 x i8]">

// CHECK: llvm.mlir.global @string_notype("1234567")
llvm.mlir.global @string_notype("1234567")

// CHECK-LABEL: references
func @references() {
  // CHECK: llvm.mlir.addressof @global : !llvm<"i64*">
  %0 = llvm.mlir.addressof @global : !llvm<"i64*">

  // CHECK: llvm.mlir.addressof @string : !llvm<"[6 x i8]*">
  %1 = llvm.mlir.addressof @string : !llvm<"[6 x i8]*">

  llvm.return
}

// -----

// expected-error @+1 {{op requires attribute 'sym_name'}}
"llvm.mlir.global"() {type = !llvm.i64, constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{op requires attribute 'type'}}
"llvm.mlir.global"() {sym_name = "foo", constant, value = 42 : i64} : () -> ()

// -----

// expected-error @+1 {{op requires attribute 'value'}}
"llvm.mlir.global"() {sym_name = "foo", type = !llvm.i64, constant} : () -> ()

// -----

// expected-error @+1 {{expects type to be a valid element type for an LLVM pointer}}
llvm.mlir.global constant @constant(37.0) : !llvm<"label">

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: non-negative 32-bit integer}}
"llvm.mlir.global"() {sym_name = "foo", type = !llvm.i64, value = 42 : i64, addr_space = -1 : i32} : () -> ()

// -----

// expected-error @+1 {{'addr_space' failed to satisfy constraint: non-negative 32-bit integer}}
"llvm.mlir.global"() {sym_name = "foo", type = !llvm.i64, value = 42 : i64, addr_space = 1.0 : f32} : () -> ()

// -----

func @foo() {
  // expected-error @+1 {{must appear at the module level}}
  llvm.mlir.global @bar(42) : !llvm.i32
}

// -----

// expected-error @+1 {{requires an i8 array type of the length equal to that of the string}}
llvm.mlir.global constant @string("foobar") : !llvm<"[42 x i8]">

// -----

// expected-error @+1 {{type can only be omitted for string globals}}
llvm.mlir.global @i64_needs_type(0: i64)

// -----

// expected-error @+1 {{expected zero or one type}}
llvm.mlir.global @more_than_one_type(0) : !llvm.i64, !llvm.i32

// -----

llvm.mlir.global @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+2{{expected ':'}}
  llvm.mlir.addressof @foo
}

// -----

func @foo() {
  // The attribute parser will consume the first colon-type, so we put two of
  // them to trigger the attribute type mismatch error.
  // expected-error @+1 {{expected symbol reference}}
  llvm.mlir.addressof "foo" : i64 : !llvm<"void ()*">
}

// -----

func @foo() {
  // expected-error @+1 {{must reference a global defined by 'llvm.mlir.global'}}
  llvm.mlir.addressof @foo : !llvm<"void ()*">
}

// -----

llvm.mlir.global @foo(0: i32) : !llvm.i32

func @bar() {
  // expected-error @+1 {{the type must be a pointer to the type of the referred global}}
  llvm.mlir.addressof @foo : !llvm<"i64*">
}
