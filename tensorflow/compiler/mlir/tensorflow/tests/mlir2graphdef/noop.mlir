// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
^bb0:
  "_tf.NoOp"() {} : () -> () loc("noop")
  return
}

// CHECK: node {
// CHECK-NEXT:  name: "noop"
// CHECK-NEXT:  op: "NoOp"
// CHECK-NEXT:  experimental_debug_info {
// CHECK-NEXT:  }
// CHECK-NEXT: }
