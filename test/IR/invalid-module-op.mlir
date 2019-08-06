// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @module_op() {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
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
  // expected-error@+2 {{expects regions to end with 'module_terminator'}}
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
