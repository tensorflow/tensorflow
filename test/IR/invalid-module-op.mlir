// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @module_op() {
  // expected-error@+1 {{expected body region to have a single block}}
  module {
  ^bb1:
    "module_terminator"() : () -> ()
  ^bb2:
    "module_terminator"() : () -> ()
  }
  return
}

// -----

func @module_op() {
  // expected-error@+1 {{expected body to have no arguments}}
  module {
  ^bb1(%arg: i32):
    "module_terminator"() : () -> ()
  }
  return
}

// -----

func @module_op() {
  // expected-error@+2 {{expects region to end with 'module_terminator'}}
  // expected-note@+1 {{the absence of terminator implies 'module_terminator'}}
  module {
    return
  }
  return
}

// -----

func @module_op() {
  // expected-error@+1 {{is expected to terminate a 'module' operation}}
  "module_terminator"() : () -> ()
}

// -----

module {
// expected-error@-1 {{may not contain operations that produce results}}
// expected-note@+1 {{see offending operation defined here}}
  %result = "foo.op"() : () -> (i32)
}
