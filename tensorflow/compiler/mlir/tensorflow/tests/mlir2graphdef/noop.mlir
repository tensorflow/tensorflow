// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
    tf_executor.island wraps "tf.NoOp"() {} : () -> () loc("noop")
    tf_executor.fetch
  }
  return
}

// CHECK: node {
// CHECK-NEXT:  name: "noop"
// CHECK-NEXT:  op: "NoOp"
// CHECK-NEXT:  experimental_debug_info {
// CHECK-NEXT:  }
// CHECK-NEXT: }
