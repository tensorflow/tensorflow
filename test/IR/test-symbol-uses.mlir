// RUN: mlir-opt %s -test-symbol-uses -split-input-file -verify-diagnostics

// Symbol references to the module itself don't affect uses of symbols within
// its table.
module attributes {sym.outside_use = @symbol_foo } {
  // expected-remark@+1 {{function has 2 uses}}
  func @symbol_foo()

  // expected-remark@below {{function has no uses}}
  // expected-remark@below {{found use of function : @symbol_foo}}
  // expected-remark@below {{function contains 2 nested references}}
  func @symbol_bar() attributes {sym.use = @symbol_foo} {
    // expected-remark@+1 {{found use of function : @symbol_foo}}
    "foo.op"() {
      non_symbol_attr,
      use = [{ nested_symbol = [@symbol_foo]}],
      z_other_non_symbol_attr
    } : () -> ()
  }

  // expected-remark@+1 {{function has 1 use}}
  func @symbol_baz()

  // expected-remark@+1 {{found use of function : @symbol_baz}}
  module attributes {test.reference = @symbol_baz} {
    "foo.op"() {test.nested_reference = @symbol_baz} : () -> ()
  }
}

// -----

// expected-remark@+1 {{contains an unknown nested operation that 'may' define a new symbol table}}
func @symbol_bar() {
  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}
