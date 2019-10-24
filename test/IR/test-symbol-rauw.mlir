// RUN: mlir-opt %s -test-symbol-rauw -split-input-file | FileCheck %s

// Symbol references to the module itself don't affect uses of symbols within
// its table.
// CHECK: module
// CHECK-SAME: @symbol_foo
module attributes {sym.outside_use = @symbol_foo } {
  // CHECK: func @replaced_foo
  func @symbol_foo() attributes {sym.new_name = "replaced_foo" }

  // CHECK: func @symbol_bar
  // CHECK-NEXT: @replaced_foo
  func @symbol_bar() attributes {sym.use = @symbol_foo} {
    // CHECK: foo.op
    // CHECK-SAME: non_symbol_attr,
    // CHECK-SAME: use = [{nested_symbol = [@replaced_foo], other_use = @symbol_bar, z_use = @replaced_foo}],
    // CHECK-SAME: z_non_symbol_attr_3
    "foo.op"() {
      non_symbol_attr,
      use = [{nested_symbol = [@symbol_foo], other_use = @symbol_bar, z_use = @symbol_foo}],
      z_non_symbol_attr_3
    } : () -> ()
  }

  // CHECK: module attributes {test.reference = @replaced_foo}
  module attributes {test.reference = @symbol_foo} {
    // CHECK: foo.op
    // CHECK-SAME: @symbol_foo
    "foo.op"() {test.nested_reference = @symbol_foo} : () -> ()
  }
}

// -----

// Check that the replacement fails for potentially unknown symbol tables.
module {
  // CHECK: func @failed_repl
  func @failed_repl() attributes {sym.new_name = "replaced_name" }

  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}
