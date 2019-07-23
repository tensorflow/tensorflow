// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK: "llvm.func"
  // CHECK: sym_name = "foo"
  // CHECK-SAME: type = !llvm<"void ()">
  // CHECK-SAME: () -> ()
  "llvm.func"() ({
  }) {sym_name = "foo", type = !llvm<"void ()">} : () -> ()

  // CHECK: "llvm.func"
  // CHECK: sym_name = "bar"
  // CHECK-SAME: type = !llvm<"i64 (i64, i64)">
  // CHECK-SAME: () -> ()
  "llvm.func"() ({
  }) {sym_name = "bar", type = !llvm<"i64 (i64, i64)">} : () -> ()

  // CHECK: "llvm.func"
  "llvm.func"() ({
  // CHECK: ^bb0
  ^bb0(%arg0: !llvm.i64):
    // CHECK: llvm.return
    llvm.return %arg0 : !llvm.i64

  // CHECK: sym_name = "baz"
  // CHECK-SAME: type = !llvm<"i64 (i64)">
  // CHECK-SAME: () -> ()
  }) {sym_name = "baz", type = !llvm<"i64 (i64)">} : () -> ()
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
