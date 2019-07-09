// RUN: mlir-opt %s | FileCheck %s

// CHECK: module {
module {
}

// CHECK: module {
// CHECK-NEXT: }
module {
  "module_terminator"() : () -> ()
}

// CHECK: module attributes {foo.attr = true} {
module attributes {foo.attr = true} {
}

// CHECK: module {
module {
  // CHECK-NEXT: "foo.result_op"() : () -> i32
  %result = "foo.result_op"() : () -> i32
}
