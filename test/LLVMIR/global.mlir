// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: llvm.global @global(42 : i64) : !llvm.i64
llvm.global @global(42 : i64) : !llvm.i64

// CHECK: llvm.global constant @constant(3.700000e+01 : f64) : !llvm.float
llvm.global constant @constant(37.0) : !llvm.float

// CHECK: llvm.global constant @string("foobar") : !llvm<"[6 x i8]">
llvm.global constant @string("foobar") : !llvm<"[6 x i8]">

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
