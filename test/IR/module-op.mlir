// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @module_parsing
func @module_parsing() {
  // CHECK-NEXT: module {
  module {
  }

  // CHECK: module {
  // CHECK-NEXT: }
  module {
    "module_terminator"() : () -> ()
  }

  // CHECK: module attributes {foo.attr: true} {
  module attributes {foo.attr: true} {
  }

  return
}
