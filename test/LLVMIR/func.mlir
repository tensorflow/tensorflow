// RUN: mlir-opt -split-input-file -verify-diagnostics %s | mlir-opt | FileCheck %s
// RUN: mlir-opt -split-input-file -verify-diagnostics -mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

module {
  // GENERIC: "llvm.func"
  // GENERIC: sym_name = "foo"
  // GENERIC-SAME: type = !llvm<"void ()">
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @foo()
  "llvm.func"() ({
  }) {sym_name = "foo", type = !llvm<"void ()">} : () -> ()

  // GENERIC: "llvm.func"
  // GENERIC: sym_name = "bar"
  // GENERIC-SAME: type = !llvm<"i64 (i64, i64)">
  // GENERIC-SAME: () -> ()
  // CHECK: llvm.func @bar(!llvm.i64, !llvm.i64) -> !llvm.i64
  "llvm.func"() ({
  }) {sym_name = "bar", type = !llvm<"i64 (i64, i64)">} : () -> ()

  // GENERIC: "llvm.func"
  // CHECK: llvm.func @baz(%{{.*}}: !llvm.i64) -> !llvm.i64
  "llvm.func"() ({
  // GENERIC: ^bb0
  ^bb0(%arg0: !llvm.i64):
    // GENERIC: llvm.return
    llvm.return %arg0 : !llvm.i64

  // GENERIC: sym_name = "baz"
  // GENERIC-SAME: type = !llvm<"i64 (i64)">
  // GENERIC-SAME: () -> ()
  }) {sym_name = "baz", type = !llvm<"i64 (i64)">} : () -> ()

  // CHECK: llvm.func @qux(!llvm<"i64*"> {llvm.noalias = true}, !llvm.i64)
  // CHECK-NEXT: attributes  {xxx = {yyy = 42 : i64}}
  "llvm.func"() ({
  }) {sym_name = "qux", type = !llvm<"void (i64*, i64)">,
      arg0 = {llvm.noalias = true}, xxx = {yyy = 42}} : () -> ()

  // CHECK: llvm.func @roundtrip1()
  llvm.func @roundtrip1()

  // CHECK: llvm.func @roundtrip2(!llvm.i64, !llvm.float) -> !llvm.double
  llvm.func @roundtrip2(!llvm.i64, !llvm.float) -> !llvm.double

  // CHECK: llvm.func @roundtrip3(!llvm.i32, !llvm.i1)
  llvm.func @roundtrip3(%a: !llvm.i32, %b: !llvm.i1)

  // CHECK: llvm.func @roundtrip4(%{{.*}}: !llvm.i32, %{{.*}}: !llvm.i1) {
  llvm.func @roundtrip4(%a: !llvm.i32, %b: !llvm.i1) {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip5()
  // CHECK-NEXT: attributes  {baz = 42 : i64, foo = "bar"}
  llvm.func @roundtrip5() attributes {foo = "bar", baz = 42}

  // CHECK: llvm.func @roundtrip6()
  // CHECK-NEXT: attributes  {baz = 42 : i64, foo = "bar"}
  llvm.func @roundtrip6() attributes {foo = "bar", baz = 42} {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip7() {
  llvm.func @roundtrip7() attributes {} {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip8() -> !llvm.i32
  llvm.func @roundtrip8() -> !llvm.i32 attributes {}

  // CHECK: llvm.func @roundtrip9(!llvm<"i32*"> {llvm.noalias = true})
  llvm.func @roundtrip9(!llvm<"i32*"> {llvm.noalias = true})

  // CHECK: llvm.func @roundtrip10(!llvm<"i32*"> {llvm.noalias = true})
  llvm.func @roundtrip10(%arg0: !llvm<"i32*"> {llvm.noalias = true})

  // CHECK: llvm.func @roundtrip11(%{{.*}}: !llvm<"i32*"> {llvm.noalias = true}) {
  llvm.func @roundtrip11(%arg0: !llvm<"i32*"> {llvm.noalias = true}) {
    llvm.return
  }

  // CHECK: llvm.func @roundtrip12(%{{.*}}: !llvm<"i32*"> {llvm.noalias = true})
  // CHECK-NEXT: attributes  {foo = 42 : i32}
  llvm.func @roundtrip12(%arg0: !llvm<"i32*"> {llvm.noalias = true})
  attributes {foo = 42 : i32} {
    llvm.return
  }
}

// -----

module {
  // expected-error@+1 {{expects one region}}
  "llvm.func"() {sym_name = "no_region", type = !llvm<"void ()">} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires a type attribute 'type'}}
  "llvm.func"() ({}) {sym_name = "missing_type"} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires 'type' attribute of wrapped LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_llvm_type", type = i64} : () -> ()
}

// -----

module {
  // expected-error@+1 {{requires 'type' attribute of wrapped LLVM function type}}
  "llvm.func"() ({}) {sym_name = "non_function_type", type = !llvm<"i64">} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block must have 0 arguments}}
  "llvm.func"() ({
  ^bb0(%arg0: !llvm.i64):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm<"void ()">} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block argument #0 is not of LLVM type}}
  "llvm.func"() ({
  ^bb0(%arg0: i64):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm<"void (i64)">} : () -> ()
}

// -----

module {
  // expected-error@+1 {{entry block argument #0 does not match the function signature}}
  "llvm.func"() ({
  ^bb0(%arg0: !llvm.i32):
    llvm.return
  }) {sym_name = "wrong_arg_number", type = !llvm<"void (i64)">} : () -> ()
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected LLVM type for function arguments}}
  llvm.func @foo(i64)
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected LLVM type for function results}}
  llvm.func @foo() -> i64
}

// -----

module {
  // expected-error@+1 {{failed to construct function type: expected zero or one function result}}
  llvm.func @foo() -> (!llvm.i64, !llvm.i64)
}
